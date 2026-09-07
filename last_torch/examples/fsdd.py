"""Audio dataset adapted from https://github.com/eonu/torch-fsdd."""
from pathlib import Path

import torch


class TorchFSDD(torch.utils.data.Dataset):
    """Read FSDD WAV files as mono float32 audio, optionally resampling.

    Preloading is local to each dataset instance. Transforms receive [samples].
    A label_transform returning (labels, length) produces three-item examples.
    """
    def __init__(self, files, transforms=None, load_all=False,
                 label_transform=None, sample_rate=8000):
        self.files = [Path(f) for f in files]
        self.transforms = transforms
        self.label_transform = label_transform
        self.sample_rate = sample_rate
        self._recordings = [self._read(f) for f in self.files] if load_all else None

    def _read(self, file):
        import soundfile as sf
        data, rate = sf.read(file, dtype="float32", always_2d=True)
        audio = torch.from_numpy(data).mean(dim=1)
        if audio.numel() == 0:
            raise ValueError(f"Empty recording: {file}")
        if self.sample_rate is not None and rate != self.sample_rate:
            from torchaudio.functional import resample
            audio = resample(audio, rate, self.sample_rate)
        label = int(file.stem.split("_")[0])
        if not 0 <= label <= 9:
            raise ValueError(f"Invalid digit in {file}")
        return audio, label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        audio, label = (self._read(self.files[index]) if self._recordings is None
                        else self._recordings[index])
        audio = audio.clone()
        if self.transforms is not None:
            audio = self.transforms(audio)
        if self.label_transform is not None:
            labels, length = self.label_transform(label)
            return audio, labels, length
        return audio, label
