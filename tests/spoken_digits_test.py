"""Regression tests for researcher-facing training and decoding evaluation."""
import torch
import pytest
from torch.utils.data import DataLoader

from last_torch.examples.spoken_digits import (
    Encoder, Model, PadOrTrim, eval_step, make_text_labels,
    remove_blank_labels, sequence_accuracy, slice_and_process_dataset,
    training_loop, move_batch,
)
from last_torch.examples.fsdd import TorchFSDD


def test_padding_and_repeated_letters():
    features, length = PadOrTrim(3)(torch.ones(2, 5))
    assert features.shape == (2, 3) and length == 3
    labels, length = make_text_labels(3)
    assert labels.tolist() == [20, 8, 18, 5, 5] and length == 5
    hyp = torch.tensor([[0, 20, 8, 18, 0, 5, 5], [1, 0, 2, 0, 0, 0, 0]])
    assert remove_blank_labels(hyp).tolist()[1] == [1, 2, 0, 0, 0, 0, 0]
    assert sequence_accuracy(labels[None], hyp[:1]).item() == 1
    with pytest.raises(ValueError):
        make_text_labels(3, 4)


def test_encoder_batch_independence():
    torch.manual_seed(0)
    encoder = Encoder(4, 1)
    x = torch.randn(3, 6, 2)
    combined = encoder(x)
    separate = torch.cat([encoder(row[None]) for row in x])
    torch.testing.assert_close(combined, separate)


def test_eval_weighting_and_mode():
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.))

        def forward(self, frames, *unused):
            assert not self.training
            return frames.mean()

        def decode(self, frames, lengths):
            return torch.ones((frames.shape[0], 1), dtype=torch.long)

    data = [((torch.tensor([[float(i)]]), 1), torch.tensor([1 if i < 2 else 2]), 1)
            for i in range(3)]
    model = Dummy()
    metrics = eval_step(model, DataLoader(data, batch_size=2))
    assert metrics == dict(loss=1., sequence_accuracy=2/3, examples=3)
    assert model.training
    model.eval()
    eval_step(model, DataLoader(data))
    assert not model.training


def test_audio_loading_split_and_training(tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("torchaudio")
    import numpy as np
    files = []
    for digit in (0, 1):
        for utterance in (0, 5, 6):
            file = tmp_path / f"{digit}_speaker_{utterance}.wav"
            # Exercise resampling and stereo-to-mono conversion.
            t = np.arange(1600) / 16000
            mono = np.sin(2 * np.pi * (300 + digit*100) * t).astype("float32")
            sf.write(file, np.stack([mono, mono], axis=1), 16000)
            files.append(file)
    cached = TorchFSDD(files, load_all=True)
    lazy = TorchFSDD(files)
    torch.testing.assert_close(cached[0][0], lazy[0][0])
    assert cached[0][0].shape == (800,)
    train, test = slice_and_process_dataset(files[::-1], n_mfcc=4, max_num_frames=6)
    assert len(train) == 4 and len(test) == 2
    (frames, length), _, _ = train[0]
    assert frames.shape == (6, 4) and length <= 6
    model = Model(hidden_size=4, context_size=0)
    batch = move_batch(next(iter(DataLoader(train, batch_size=2))), "cpu")
    with torch.no_grad():
        model(*batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {k: v.clone() for k, v in model.named_parameters()}
    metrics = training_loop(test, train, model, optimizer, batch_size=2,
                            num_steps=2, num_steps_per_eval=1)
    assert metrics["examples"] == 2
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    for prefix in ("encoder.", "lattice."):
        assert any(not torch.equal(before[k], v) for k, v in model.named_parameters()
                   if k.startswith(prefix))
    assert next(iter(optimizer.state.values()))["step"] == 2
    uninitialized = Model(hidden_size=4, context_size=0)
    premature_optimizer = torch.optim.Adam(uninitialized.parameters())
    with pytest.raises(ValueError, match="warmup forward"):
        training_loop(test, train, uninitialized, premature_optimizer,
                      batch_size=2, num_steps=1)


@pytest.mark.parametrize("mask_kind", ["blank", "lexical", "both"])
def test_alignment_masks_preserve_batch_and_alignment_states(mask_kind):
    import last_torch
    # One frame, one context and one lexical label: enumerate paths of length 0,1,2.
    lattice = last_torch.RecognitionLattice(
        context=last_torch.contexts.FullNGram(1, 0),
        alignment=last_torch.alignments.FrameLabelDependent(max_expansions=2),
        weight_fn_cacher_factory=lambda _: last_torch.weight_fns.NullCacher(),
        weight_fn_factory=lambda _: last_torch.weight_fns.TableWeightFn(
            torch.zeros(2, 1, 1, 2)))
    blank = [torch.tensor([a, b]).reshape(2, 1, 1)
             for a, b in [(0., 1.), (2., -1.), (3., 4.)]]
    lexical = [torch.tensor([a, b]).reshape(2, 1, 1, 1)
               for a, b in [(1., -2.), (2., 3.), (0., 0.)]]
    use_blank = mask_kind in ("blank", "both")
    use_lexical = mask_kind in ("lexical", "both")
    b = [x.flatten() if use_blank else torch.zeros(2) for x in blank]
    l = [x.flatten() if use_lexical else torch.zeros(2) for x in lexical]
    expected = torch.logsumexp(torch.stack([b[0], l[0]+b[1], l[0]+l[1]+b[2]]), 0)
    actual, _ = lattice._forward(
        cache=None, frames=torch.zeros(2, 1, 1), num_frames=torch.ones(2, dtype=torch.long),
        semiring=last_torch.semirings.Log,
        blank_mask=blank if use_blank else None,
        lexical_mask=lexical if use_lexical else None)
    torch.testing.assert_close(actual, expected)
