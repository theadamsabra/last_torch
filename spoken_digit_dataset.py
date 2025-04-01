import torch
import os
import numpy as np
from torchfsdd_dataset import TorchFSDD
import last_torch

from torchvision.transforms import Compose
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchaudio.transforms import MFCC

'''
Translation from the LAST quick start notebook over to
last_torch.
'''

class PadOrTrim:
    """Make torch pad into a class for easy composition of
    transforms. Will also handle trimming for simplicities'
    sake.

    Args:
        max_num_frames (int): length to pad to.
    """
    def __init__(self, max_num_frames:int):
        self.max_num_frames = max_num_frames

    def __call__(self, x:torch.Tensor):
        """Applies padding transformation

        Args:
            x (torch.Tensor): tensor to be padded
        """
        num_frames = x.shape[-1]
        shape_diff = self.max_num_frames - num_frames
        
        if shape_diff > 0:
            #    right pad on second dim
            #             |
            #             V
            pad_tuple = (0,shape_diff, 0,0) # why pytorch, why?
            #                           ^
            #                           |
            #                       no pad first dim
            return torch.nn.functional.pad(x, pad_tuple), num_frames
        else:
            return x[:, :self.max_num_frames], num_frames
'''
HELPER FUNCTIONS
'''
def make_text_labels(int_label: int, max_num_labels:int=5) -> tuple[torch.Tensor, torch.Tensor]:
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
    word = vocab[int_label]
    labels = torch.Tensor([ord(letter) - ord('a') + 1 for letter in word]).to(torch.int32)
    num_labels = labels.shape[0]
    label_diff = max_num_labels - num_labels
    if label_diff > 0:
        labels = torch.nn.functional.pad(labels, (0,label_diff))

    return labels, num_labels


def slice_and_process_dataset(
        files:list[str], 
        sample_rate:int=8000, 
        n_mfcc:int=80,
        n_fft:int=512,
        hop_length:int=160,
        max_num_frames:int=30
        ) -> tuple[Dataset, Dataset]:
    """Load and process FSDD using torch-fsdd. We will slice the 
    first 1000 files for evaluation, and use the rest for training.

    Args:
        files (list[str]): list of filepaths to dataset.
            Note: '.wav' in files[i] == True
        
        sample_rate (int): sampling rate of files. default to 8000 Hz
        as that is the sampling rate for the recordings.

        n_mfcc (int): number of MFCCs to calculate for feature extraction.
        default set to 80 to align with quick start notebook in JAX.
    
        n_fft (int): FFT size for mel calculation. 
        default set to 512 to align with quick start notebook in JAX.

        hop_length (int): hop size for FFT/mel calculation
        default set to 160 to align with quick start notebook in JAX.

    Returns:
        train_subset (Subset): train slice of dataset (file index 1000-end)
        test_dataset (Subset): test slice of dataset (first 1000 files.)
    """
    # Construct MFCC transformation and apply to dataset
    transform = Compose([
        MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length
            }
        ),
        PadOrTrim(max_num_frames)
    ])
    # Load in files with the necessary transformations
    fsdd_dataset = TorchFSDD(files=files, transforms=transform, 
                                       label_transform=make_text_labels)

    # Construct subset for simple experimentation
    test_subset = Subset(fsdd_dataset, [i for i in range(0,1000)])
    train_subset = Subset(fsdd_dataset, [i for i in range(1000,len(fsdd_dataset))])
    return train_subset, test_subset

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
            xs: [batch_size, num_frames, feature_size] input sequences

        Returns:
            [batch_size, num_frames, hidden_size] output sequences.
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

class Model(nn.Module):
    def __init__(self, 
                 locally_normalize:bool=False,
                 hidden_size:int=256,
                 num_encoder_layers:int=1,
                 # As convention in LAST, we do not count the blank (0) label in the vocab.
                 vocab_size:int=26,
                 context_size:int=2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.locally_normalize = locally_normalize
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.vocab_size = vocab_size
        self.context_size = context_size

        self.encoder = Encoder(
            self.hidden_size, self.num_encoder_layers
        )
    
        def weight_fn_cacher_factory(context):
            assert isinstance(context, last_torch.contexts.FullNGram)
            return last_torch.weight_fns.SharedRNNCacher(
                vocab_size=self.vocab_size,
                context_size=self.context_size,
                rnn_size=self.hidden_size,
                rnn_embedding_size=self.hidden_size
            )
        
        def weight_fn_factory(context):
            _, vocab_size = context.shape()
            weight_fn = last_torch.weight_fns.JointWeightFn(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size
            )
            if self.locally_normalize:
                weight_fn = last_torch.weight_fns.LocallyNormalizedWeightFn(
                    weight_fn
                )
            return weight_fn
        
        # Plug in all the pieces for the lattice
        self.lattice = last_torch.RecognitionLattice(
            context=last_torch.contexts.FullNGram(
                vocab_size=self.vocab_size,
                context_size=self.context_size
            ),
            alignment=last_torch.alignments.FrameDependent(),
            weight_fn_cacher_factory=weight_fn_cacher_factory,
            weight_fn_factory=weight_fn_factory
        )
    
    def forward(self,
                input_data:torch.Tensor,
                num_frames:torch.Tensor,
                labels:torch.Tensor,
                num_labels:torch.Tensor
                ) -> torch.Tensor:
        features = self.encoder(input_data)
        return self.lattice(
            frames=features,
            num_frames=num_frames,
            labels=labels,
            num_labels=num_labels
        ).mean()

    def decode(self, 
               input_tensor:torch.Tensor,
               num_frames:torch.Tensor
               ) -> torch.Tensor:
        """Decodes batch into [batch_size, max_num_frames] alignment labels"""
        features = self.encoder(input_tensor)
        return self.lattice.shortest_path(
            frames=features, num_frames=num_frames
        )[0]

'''
TRAIN AND EVAL LOOP
'''

def train_step(input_data, num_frames, labels, num_labels, device):
    # Permute input_data and get prediction
    input_data = input_data.to(device)
    input_data = torch.permute(input_data, (0,2,1))
    return model(input_data, num_frames, labels, num_labels)

def eval_step():
    pass

def training_loop(
    test_batch, 
    train_batches, 
    model, 
    optim,
    batch_size=128,
    num_steps=1000, 
    num_steps_per_eval=100,
    device='cpu'
    ):
    '''Core training loop comprised of training and eval steps.
    Very simple training loop for basic experimentation.

    Args:
        test_batch 
        train_batches 
        model 
        optim
        batch_size
        num_steps 
        num_steps_per_eval
        device
    '''
    # Basic setup
    model.to(device)

    train_set = DataLoader(train_batches, batch_size=batch_size)
    test_set = DataLoader(test_batch, batch_size=len(test_batch)) # Evaluate on entire test set at once

    # Core training loop:
    for i, ((input_data, num_frames), labels, num_labels) in enumerate(train_set):

        output = train_step(input_data, num_frames, labels, num_labels, device)

        # Evaluate every num_steps_per_eval
        if i+1 % num_steps_per_eval == 0:
            eval_step()


'''
CORE CODE HERE
'''

PATH_TO_RECORDINGS = 'free-spoken-digit-dataset/recordings'
files = [os.path.join(PATH_TO_RECORDINGS, file) for file in os.listdir(PATH_TO_RECORDINGS)]

TRAIN_BATCHES, TEST_BATCH = slice_and_process_dataset(
    files = files 
)

for locally_normalize in [True, False]:
    model = Model(locally_normalize=locally_normalize)
    optim = torch.optim.AdamW(model.parameters())
    training_loop(TEST_BATCH, TRAIN_BATCHES, model, optim)