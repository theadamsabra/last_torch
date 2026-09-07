"""Train a character transducer on Free Spoken Digit Dataset recordings."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import last_torch
from last_torch.examples.fsdd import TorchFSDD


class PadOrTrim:
    """Pad/clip [features, time], returning the valid (clipped) length."""
    def __init__(self, max_num_frames):
        if max_num_frames < 1:
            raise ValueError("max_num_frames must be positive")
        self.max_num_frames = max_num_frames

    def __call__(self, x):
        length = min(x.shape[-1], self.max_num_frames)
        return torch.nn.functional.pad(
            x[..., :length], (0, self.max_num_frames - length)), length


def make_text_labels(int_label, max_num_labels=5):
    words = ("zero", "one", "two", "three", "four",
             "five", "six", "seven", "eight", "nine")
    if not 0 <= int_label < len(words):
        raise ValueError("digit must be in [0, 9]")
    word = words[int_label]
    if len(word) > max_num_labels:
        raise ValueError("max_num_labels is too short")
    labels = torch.zeros(max_num_labels, dtype=torch.int64)
    labels[:len(word)] = torch.tensor([ord(c) - ord("a") + 1 for c in word])
    return labels, len(word)


class AudioFeatures:
    def __init__(self, sample_rate, n_mfcc, n_fft, hop_length, max_num_frames):
        from torchaudio.transforms import MFCC
        self.mfcc = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc,
                         melkwargs=dict(n_fft=n_fft, hop_length=hop_length,
                                        n_mels=max(40, n_mfcc), pad_mode="constant"))
        self.pad = PadOrTrim(max_num_frames)

    def __call__(self, waveform):
        features, length = self.pad(self.mfcc(waveform))
        return features.transpose(0, 1).contiguous(), length


def slice_and_process_dataset(files, sample_rate=8000, n_mfcc=20,
                              n_fft=512, hop_length=160, max_num_frames=64):
    """Use utterances 0--4 for evaluation and >=5 for training, per speaker/digit."""
    dataset = TorchFSDD(
        sorted(files), sample_rate=sample_rate, label_transform=make_text_labels,
        transforms=AudioFeatures(sample_rate, n_mfcc, n_fft, hop_length, max_num_frames))
    train, test = [], []
    for i, file in enumerate(dataset.files):
        utterance = int(file.stem.rsplit("_", 1)[1])
        if utterance < 0:
            raise ValueError(f"Invalid utterance index: {file}")
        (test if utterance < 5 else train).append(i)
    if not train or not test:
        raise ValueError("Need recordings with utterance indices 0--4 and >=5")
    return Subset(dataset, train), Subset(dataset, test)

class Encoder(nn.Module):
    """Batch-first unidirectional LSTM, initialized from the first input."""
    def __init__(self, hidden_size, num_layers, device="cpu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = None

    def forward(self, xs):
        if self.lstm is None:
            self.lstm = nn.LSTM(xs.shape[-1], self.hidden_size, self.num_layers,
                                batch_first=True, device=xs.device, dtype=xs.dtype)
        return self.lstm(xs)[0]


class Model(nn.Module):
    """LSTM encoder and character lattice with a shared recurrent context cache."""
    def __init__(self, locally_normalize=False, hidden_size=32,
                 num_encoder_layers=1, vocab_size=26, context_size=1, device="cpu"):
        super().__init__()
        self.device = device
        self.encoder = Encoder(hidden_size, num_encoder_layers, device)

        def cacher(context):
            return last_torch.weight_fns.SharedRNNCacher(
                vocab_size=vocab_size, context_size=context_size,
                rnn_size=hidden_size, rnn_embedding_size=hidden_size, device=device)

        def weights(context):
            fn = last_torch.weight_fns.JointWeightFn(
                vocab_size=vocab_size, hidden_size=hidden_size, device=device)
            if locally_normalize:
                fn = last_torch.weight_fns.LocallyNormalizedWeightFn(fn, device=device)
            return fn

        self.lattice = last_torch.RecognitionLattice(
            context=last_torch.contexts.FullNGram(vocab_size, context_size, device=device),
            alignment=last_torch.alignments.FrameDependent(),
            weight_fn_cacher_factory=cacher, weight_fn_factory=weights, device=device)

    def forward(self, input_data, num_frames, labels, num_labels):
        return self.lattice(self.encoder(input_data), num_frames, labels, num_labels).mean()

    def decode(self, input_tensor, num_frames):
        return self.lattice.shortest_path(self.encoder(input_tensor), num_frames)[0]


def remove_blank_labels(labels):
    """Compact each row independently; retain repeated lexical labels."""
    if labels.ndim != 2:
        raise ValueError("Expected [batch, time] labels")
    result = torch.zeros_like(labels)
    for i, row in enumerate(labels):
        values = row[row != 0]
        result[i, :values.numel()] = values
    return result


def sequence_accuracy(ref_labels, hyp_labels):
    ref_labels, hyp_labels = map(remove_blank_labels, (ref_labels, hyp_labels))
    length = max(ref_labels.shape[1], hyp_labels.shape[1])
    ref_labels = torch.nn.functional.pad(ref_labels, (0, length-ref_labels.shape[1]))
    hyp_labels = torch.nn.functional.pad(hyp_labels, (0, length-hyp_labels.shape[1]))
    return (ref_labels == hyp_labels).all(dim=-1).float().mean()


def move_batch(batch, device):
    (frames, num_frames), labels, num_labels = batch
    if torch.any(num_frames < num_labels):
        raise ValueError("FrameDependent requires at least one frame per target label")
    return tuple(x.to(device) for x in (frames, num_frames, labels, num_labels))


@torch.no_grad()
def eval_step(model, test_set):
    was_training = model.training
    model.eval()
    total_loss = total_correct = count = 0
    try:
        for batch in test_set:
            args = move_batch(batch, next(model.parameters()).device)
            loss = model(*args)
            hypotheses = model.decode(*args[:2])
            size = args[0].shape[0]
            total_loss += loss.item() * size
            total_correct += sequence_accuracy(args[2], hypotheses).item() * size
            count += size
    finally:
        model.train(was_training)
    if not count:
        raise ValueError("Evaluation dataset is empty")
    return {"loss": total_loss/count, "sequence_accuracy": total_correct/count,
            "examples": count}


def training_loop(test_batch, train_batches, model, optim, batch_size=8,
                  num_steps=100, num_steps_per_eval=100, device="cpu"):
    """Run exactly num_steps optimizer updates; report example-weighted evaluation."""
    if min(batch_size, num_steps, num_steps_per_eval) < 1 or not len(train_batches):
        raise ValueError("Training sizes and intervals must be positive")
    train_set = DataLoader(train_batches, batch_size=batch_size, shuffle=True)
    test_set = DataLoader(test_batch, batch_size=batch_size)
    # Initialize all lazy parameters and verify that the supplied optimizer owns them.
    with torch.no_grad():
        model(*move_batch(next(iter(train_set)), device))
    owned = {id(p) for group in optim.param_groups for p in group["params"]}
    if any(p.requires_grad and id(p) not in owned for p in model.parameters()):
        raise ValueError("Run a warmup forward before constructing the optimizer")
    step = 0
    while step < num_steps:
        model.train()
        for batch in train_set:
            optim.zero_grad(set_to_none=True)
            loss = model(*move_batch(batch, device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optim.step()
            step += 1
            if step == 1 or step % num_steps_per_eval == 0 or step == num_steps:
                metrics = eval_step(model, test_set)
                print(json.dumps(dict(step=step, train_loss=loss.item(),
                                      grad_norm=norm.item(), evaluation=metrics)), flush=True)
            if step == num_steps:
                break
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--context-size", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--locally-normalize", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train, test = slice_and_process_dataset(
        list(args.recordings.glob("*.wav")), n_mfcc=args.n_mfcc,
        max_num_frames=args.max_frames)
    print(json.dumps(dict(train_examples=len(train), evaluation_examples=len(test),
                          config={k: str(v) if isinstance(v, Path) else v
                                  for k, v in vars(args).items()})), flush=True)
    model = Model(device=args.device, hidden_size=args.hidden_size,
                  context_size=args.context_size, locally_normalize=args.locally_normalize)
    with torch.no_grad():
        model(*move_batch(next(iter(DataLoader(train, batch_size=args.batch_size))), args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    metrics = training_loop(test, train, model, optimizer, args.batch_size,
                            args.steps, args.eval_every, args.device)
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        config=config, step=args.steps, metrics=metrics), args.checkpoint)


if __name__ == "__main__":
    main()
