import torch
import numpy as np
import torchaudio
import torchfsdd  


def make_text_labels(int_label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts class labels into text.

    Args:
        int_label: [] int64 class label.

    Returns:
        (labels, num_labels) tuple:
        -   labels: [num_labels] int32 output labels, in the range of [1, 26],
        corresponding to letters "a" to "z".
        -   num_labels: [] int32 length of the output labels.
    """
    vocab = np.array([
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
])
    labels = torch.from_numpy(
            np.char.decode(vocab[int_label], 'ascii') - ord('a') + 1
        ).to(torch.int32)
    num_labels = labels.shape[0].to(torch.int32)
    return labels, num_labels
