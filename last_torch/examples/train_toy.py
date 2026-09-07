"""Overfit a small synthetic batch to verify encoder/lattice training."""
import argparse
import json
from pathlib import Path

import torch
from last_torch.examples.spoken_digits import Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = args.device
    labels = torch.tensor([[1, 2], [2, 1], [1, 1], [2, 2]], device=device)
    # Each target is observed for three frames; this is a training smoke test.
    frames = torch.nn.functional.one_hot(labels - 1, 2).float().repeat_interleave(3, 1)
    frames = frames + 0.05 * torch.randn_like(frames)
    num_frames = torch.full((4,), 6, dtype=torch.int64, device=device)
    num_labels = torch.full((4,), 2, dtype=torch.int64, device=device)
    batch = (frames, num_frames, labels, num_labels)
    model = Model(hidden_size=8, vocab_size=2, context_size=1, device=device)
    with torch.no_grad():
        initial_loss = model(*batch).item()  # Materialize lazy parameters first.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(*batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
    with torch.no_grad():
        final_loss = model(*batch).item()
    print(json.dumps(dict(initial_loss=initial_loss, final_loss=final_loss,
                          steps=args.steps, device=device)), flush=True)
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        step=args.steps, seed=args.seed), args.checkpoint)


if __name__ == "__main__":
    main()
