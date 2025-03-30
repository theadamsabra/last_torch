import torch
import numpy as np
import torchfsdd  

from torch import nn
from torch.utils.data import Dataset
from torchaudio.transforms import MFCC

'''
HELPER FUNCTIONS
'''
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


def slice_and_process_dataset(
        files:list[str], 
        sample_rate:int=8000, 
        n_mfcc:int=80) -> tuple[Dataset, Dataset]:
    """Load and process FSDD using torch-fsdd. We will slice the 
    first 1000 files for evaluation, and use the rest for training.

    Args:
        files (list[str]): list of filepaths to dataset.
            Note: '.wav' in files[i] == True
        
        sample_rate (int): sampling rate of files. default to 8000 Hz
        as that is the sampling rate for the recordings.

        n_mfcc (int): number of MFCCs to calculate for feature extraction.
        default set to 80 to align with quick start notebook in JAX.
    
    Returns:
        train_dataset (Dataset): training slice of dataset (files 1001-N.)

        test_dataset (Dataset): test slice of dataset (first 1000 files.)
    """
    # Construct MFCC transformation and apply to dataset
    transform = MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc
    ) 
    fsdd_dataset = torchfsdd.TorchFSDD(files=files, transforms=transform)
    return fsdd_dataset[1000:], fsdd_dataset[:1000]


'''
MODEL DEFINITIONS
'''
class Encoder(nn.Module):
    """A stack of unidirectional LSTMs."""
    def __init__(self, hidden_size:int, num_layers:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, xs:torch.Tensor):
        """Encode the inputs.
        
        Args:
            xs: [batch_size, max_num_frames, feature_size] input sequences

        Returns:
            [batch_size, max_num_frames, hidden_size] output sequences.
        """
        input_size = xs.shape[-1]

        # Make it an attribute so we can access it for later analysis
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers # We can do stacked LSTM with this param
        )
        
        # Return final encoded output
        return self.lstm(xs)[0]
        