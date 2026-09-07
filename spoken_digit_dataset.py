"""Compatibility entry point for the packaged spoken-digit experiment."""
from last_torch.examples.spoken_digits import (
    Encoder, Model, PadOrTrim, eval_step, main, make_text_labels,
    remove_blank_labels, sequence_accuracy, slice_and_process_dataset,
    training_loop,
)

if __name__ == "__main__":
    main()
