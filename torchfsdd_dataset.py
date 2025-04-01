'''

Slightly repurposed TorchFSDD from
    https://github.com/eonu/torch-fsdd/blob/master/lib/torchfsdd/dataset.py
to make transforming the label easier for ASR tasks as opposed to classification.

'''

import os, torch, torchaudio

class TorchFSDD(torch.utils.data.Dataset):
    """A :class:`torch:torch.utils.data.Dataset` wrapper for specified
    WAV audio recordings of the Free Spoken Digit Dataset.

    .. tip::

        There should rarely be a situation where you have to initialize this class manually,
        unless you are experimenting with specific subsets of the FSDD. You should use :class:`TorchFSDDGenerator`
        to either load the full data set or generate splits for training/validation/testing.

    Parameters
    ----------
    files: list of str
        List of file paths to the WAV audio recordings for the dataset.

    transforms: callable, optional
        A callable transformation to apply to a 1D :class:`torch:torch.Tensor` of audio samples.

        This can be a single transformation, such as the :class:`TrimSilence` transformation included in this package.

        .. code-block:: python

            from torchfsdd import TorchFSDDGenerator, TrimSilence

            fsdd = TorchFSDDGenerator(transforms=TrimSilence(threshold=150))

        It could also be a series of transformations composed together with :class:`torchvision:torchvision.transforms.Compose`.

        .. code-block:: python

            from torchfsdd import TorchFSDDGenerator, TrimSilence
            from torchaudio.transforms import MFCC
            from torchvision.transforms import Compose

            fsdd = TorchFSDDGenerator(transforms=Compose([
                TrimSilence(threshold=100),
                MFCC(sample_rate=8e3, n_mfcc=13)
            ]))

        There are many useful audio transformations in :py:mod:`torchaudio:torchaudio.transforms` such as :class:`torchaudio:torchaudio.transforms.MFCC`.

    load_all: bool
        Whether or not to load the entire dataset into memory.

        This essentially defeats the point of batching, but the dataset is relatively small
        enough that it can comfortably fit into memory and possibly provide some speed-up.

        If this is set to `True`, then the complete set of raw audio recordings and labels
        (for the specified files) can be accessed with ``self.recordings`` and ``self.labels``.

    label_transform: callable, optional
        Callable function to convert the word to numeric labels for ASR.

    **args: optional
        Arbitrary keyword arguments passed on to :py:func:`torchaudio:torchaudio.load`.
    """
    def __init__(self, files, transforms=None, load_all=False, label_transform=None, **args):
        super().__init__()
        self.files = files
        self.transforms = transforms
        self.label_transform = label_transform
        self.args = args

        get_audio = lambda file: torchaudio.load(file, **self.args)[0]
        get_label = lambda file: int(os.path.basename(file)[0])

        if load_all:
            self.recordings, self.labels = [], []
            for file in self.files:
                self.recordings.append(get_audio(file))
                self.labels.append(get_label(file))

            def _load(self, index):
                return self.recordings[index], self.labels[index]
        else:
            def _load(self, index):
                file = self.files[index]
                return get_audio(file), get_label(file)

        setattr(self.__class__, '_load', _load)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Fetch the audio and corresponding label
        x, y = self._load(index)
        x = x.flatten()

        # Transform data if a transformation is given
        if self.transforms is not None:
            x = self.transforms(x)
        
        if self.label_transform is not None:
            labels, num_labels = self.label_transform(y)
            return x, labels, num_labels

        else:
            return x, y