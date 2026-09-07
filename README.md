# LAST Torch

LAST Torch is a PyTorch port of Google's
[LAttice-based Speech Transducer (LAST)](https://github.com/google-research/last)
library for research on lattice-based speech recognition, including
[globally normalized autoregressive transducers (GNAT)](https://arxiv.org/abs/2205.13674).
It provides context dependencies, alignment topologies, semiring operations,
neural arc weights, differentiable sequence losses, and Viterbi decoding.

## Installation

Python 3.10 or newer and PyTorch 2.8 or newer (before 3.0) are required.
The tested environment uses Python 3.12 and PyTorch 2.8.0 on Linux, with CPU
and CUDA execution. Select a PyTorch build appropriate for your hardware using
the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

From a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python -c "import last_torch; print(last_torch.__version__)"
```

The distribution and import name are both `last_torch`. This repository can be
installed with pip locally; these instructions do not assume a PyPI release.
For development and audio experiments:

```bash
python -m pip install -e '.[audio,test,dev]'
```

Core installation requires only Torch, einops, and optree. The `audio` extra
adds torchaudio and SoundFile; use a torchaudio build matching your Torch
version and CPU/CUDA build. The `test`, `benchmark`, and `dev` extras add
test dependencies, JAX comparison dependencies, and package build tools,
respectively. The `requirements.txt` file installs the checkout with its audio and test extras;
dependency declarations are maintained in `pyproject.toml`.

## Verify training without a dataset

The installed package includes a synthetic training example:

```bash
python -m last_torch.examples.train_toy --steps 30
# With a CUDA-enabled Torch installation:
python -m last_torch.examples.train_toy --device cuda --steps 30
```

It trains an LSTM encoder and a globally normalized lattice on four synthetic
sequences, printing initial and final loss. It is an overfitting smoke test,
not a recognition benchmark. Use `--checkpoint runs/toy.pt` to save model and
optimizer state. The equivalent console command is `last-torch-toy`.

## Spoken-digit experiment

Install the audio extra, obtain the
[Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset),
and supply the recordings directory explicitly:

```bash
python -m pip install -e '.[audio]'
git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git
python -m last_torch.examples.spoken_digits \
  --recordings free-spoken-digit-dataset/recordings \
  --device cpu --steps 100 --eval-every 25 --batch-size 8 \
  --hidden-size 32 --context-size 1 --checkpoint runs/digits.pt
```

Use `--device cuda` for GPU training. Both `last-torch-spoken-digits` and
`python spoken_digit_dataset.py` invoke the same implementation.
Run with `--help` for all options.

The loader averages audio channels, resamples to 8 kHz, and computes 20 MFCCs
with a 512-sample FFT and a 160-sample hop. Features are padded or clipped to
64 frames by default (`--max-frames`); valid lengths are clipped with them.
The encoder receives tensors in `[batch, time, features]` order. A short frame
budget can discard speech; choose it for your data. Examples with fewer valid
frames than target letters are rejected because the frame-dependent alignment
cannot represent their targets.

Filenames must follow `digit_speaker_utterance.wav`. Utterance indices 0–4
form the evaluation split; indices 5 and above form the training split.
Sorting makes the split independent of directory enumeration order.
Speakers occur in both splits, so this experiment does not measure generalization
to unseen speakers. Use a separate validation split when tuning parameters
against a held-out test set.

Targets are the spelled-out digit names, with letters a–z encoded as 1–26;
0 denotes blank or padding. Evaluation removes blank symbols and compares the
complete predicted character sequence with the target. Repeated letters are
preserved: “three” contains two consecutive e symbols. Reported loss and exact
sequence accuracy are weighted by the number of examples, including the final
partial batch.

`--steps` counts optimizer updates. Training shuffles examples, checks finite
losses and gradients, clips the gradient norm to 5, and evaluates at the first
step, every `--eval-every` updates, and the final step. Evaluation restores the
previous training mode. `--seed` controls initialization and shuffling; exact
cross-device reproducibility is not guaranteed. JSON lines record configuration,
split sizes, training loss, gradient norm, and evaluation metrics.
The default optimizer is AdamW with learning rate 0.001; adjust with `--lr`.
Use `--locally-normalize` to compare local and global normalization.

Checkpoints contain `model`, `optimizer`, `config`, `step`, and `metrics`.
To restore in an experiment, reconstruct the same model and feature configuration,
run one representative forward pass to initialize lazy layers, load
`model.load_state_dict(checkpoint["model"])`, create the optimizer, and load its
state. The CLI saves checkpoints but does not implement automatic resume.

## Build your own training loop

`RecognitionLattice` combines a context dependency, alignment topology,
weight function, and context-weight cache. The packaged
[spoken-digit model](last_torch/examples/spoken_digits.py) shows an LSTM encoder,
`FullNGram`, `FrameDependent`, `SharedRNNCacher`, and `JointWeightFn`.
The [synthetic example](last_torch/examples/train_toy.py) provides a small complete
training loop.

The lattice accepts:

| Argument | Shape | Convention |
| --- | --- | --- |
| `frames` | `[batch, T, feature_size]` | Floating-point encoder outputs |
| `num_frames` | `[batch]` | Integer valid lengths, at most T |
| `labels` | `[batch, L]` | Integer lexical labels 1 through vocabulary size; pad with 0 |
| `num_labels` | `[batch]` | Integer valid target lengths, at most L |

Place the lattice and inputs on the same device; construct the lattice with its
intended `device`. Its context structures and lazy modules also use this device,
so moving a partially initialized model with `.to()` alone is not sufficient.
The examples use float32. Reduced-precision training has not been validated.

The lattice returns a loss for each example. Reduce it with `.mean()`, call
`.backward()`, and update encoder and lattice parameters together.
**Run a representative forward pass before constructing the optimizer**:
neural weight modules and the example encoder initialize parameters lazily.
Otherwise the optimizer can silently omit parameters. Initialize before loading
a checkpoint for the same reason.

For a vocabulary of size V and context order C, `FullNGram` has
`1 + V + ... + V**C` states. Start with a small vocabulary/context order and batch
size; the state space and lexical arc storage grow quickly.
`shortest_path` returns alignment labels, valid alignment lengths, and path scores;
alignment labels include blanks and must be converted to lexical sequences
before measuring recognition accuracy.

## GPU execution and memory

The globally normalized loss **shares arc weights between the numerator and
denominator**. This implementation is retained for its memory benefit.
In the recorded RTX 3090 workload, sharing reduced peak allocated tensor memory
from 99.396 to 29.881 MiB at T=200 and from 514.271 to 86.878 MiB at T=500
(69.9% and 83.1%). Its latency change alone was negligible.

Log-semiring training compiles complete forward/backward steps; context-state
construction is vectorized. Initial calls can take substantially longer because
of compilation. For repeated CUDA calls with fixed padded shapes, optionally use:

```python
import last_torch
import torch

# lattice and inputs are on CUDA; frames are encoder outputs.
frames.requires_grad_(True)
captured = last_torch.make_graphed_lattice(
    lattice, (frames, num_frames, labels, num_labels))
optimizer = torch.optim.Adam(lattice.parameters())  # lazy layers initialized

optimizer.zero_grad(set_to_none=True)
loss = captured(frames, num_frames, labels, num_labels).mean()
loss.backward()
optimizer.step()
```

An upstream encoder needs its parameters included in the optimizer as well.
Capture is optional and is not enabled by the training CLIs. Input values,
valid lengths, and parameter values may change; shapes, strides, dtypes, devices,
and `requires_grad` flags must match capture. Keep a wrapper per shape bucket,
including a separate bucket for a smaller final batch, or drop that batch.
Each wrapper consumes graph memory.

Complete backward before replaying a wrapper. Clone outputs or returned gradients
that must survive another replay. Capture supports first-order gradients and
requires capture-compatible weight functions. Concurrent replay of one wrapper
and replacing parameter objects are unsupported; ordinary lattice calls remain
available.

Recorded loss-plus-gradient latency (milliseconds):

| Execution | T=200 | T=500 |
| --- | ---: | ---: |
| JAX, corrected benchmark | 21.342 | 62.590 |
| Torch, corrected baseline | 34.672 | 85.635 |
| Torch, final uncaptured | 20.043 | 45.066 |
| Torch, CUDA Graph replay | 2.857 | 18.476 |

These are lattice-only measurements: batch 16, feature size 80, vocabulary 4,
context order 1, float32 with TF32 disabled, and maximum reference length T/2.
Both frameworks compute gradients for frames and all parameters. Timings exclude
the encoder, optimizer, data loading, and initial compilation/capture. They are
not end-to-end speech training speedups. Graph capture increases retained memory;
allocated tensor memory is distinct from reserved or total GPU memory.
See [measurement details](benchmarks/results/README.md) and [paper.tex](paper.tex)
for the sequential ablations and limitations.

## Tests and benchmarks

From the checkout:

```bash
python -m pip install -e '.[test,audio]'
python -m pytest tests -q
```

Audio tests require the audio extra; CUDA tests require a visible GPU.
To compare against JAX, also install the benchmark extra and clone upstream LAST
beside this checkout:

```bash
python -m pip install -e '.[benchmark]'
git clone https://github.com/google-research/last ../last
python -m benchmarks.bench_gpu --device cpu --frames 10 \
  --output benchmarks/results/cpu_current.csv
python -m benchmarks.bench_gpu --device cuda --frames 200 500 \
  --output benchmarks/results/gpu_current.csv
python -m benchmarks.bench_gpu --framework torch --cuda-graph \
  --output benchmarks/results/gpu_graph_current.csv
```

For JAX GPU comparisons, install a GPU-enabled JAX build following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
The benchmark currently imports JAX and upstream LAST even for Torch-only runs.
Use `--label-ratio` to vary reference length. CSV sidecars record the environment
and configuration. Historical `gpu.csv` results use a different protocol and
should not be mixed with the corrected measurements. Preserve recorded ablations
by choosing new output paths. Regenerate paper tables with
`python -m benchmarks.render_gpu_tables`.

Build distribution artifacts with `python -m build`. The wheel includes the
library and runnable examples; benchmarks and tests are run from the checkout.

## Attribution

This project derives from [Google Research LAST](https://github.com/google-research/last)
and is licensed under Apache 2.0. See the upstream project and the GNAT paper
for the underlying algorithms. The spoken-digit dataset has its own attribution
and license in its upstream repository.
