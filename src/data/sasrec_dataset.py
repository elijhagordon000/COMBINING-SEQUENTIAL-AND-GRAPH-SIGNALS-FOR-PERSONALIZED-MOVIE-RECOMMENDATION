from src.data.build_sequences import (
    load_ratings,
    build_user_sequences,
    split_user_sequences,
)

import torch
from torch.utils.data import Dataset, DataLoader


def pad_sequence(seq, max_len, pad_token=0):
    """
    Pad a sequence on the left with pad_token so that its length becomes max_len.
    If the sequence is too long, keep only the last max_len items.
    """
    if len(seq) > max_len:
        seq = seq[-max_len:]

    if len(seq) < max_len:
        seq = [pad_token] * (max_len - len(seq)) + seq

    return seq


def build_sasrec_examples(user_train, max_len=5):
    """
    Convert user training sequences into SASRec-style training examples.

    Example:
        [10, 20, 30, 40]
    becomes:
        [10] -> 20
        [10, 20] -> 30
        [10, 20, 30] -> 40

    Returns:
        examples: list of tuples
            (user_id, padded_input_seq, target_item)
    """
    examples = []

    for user_id, seq in user_train.items():
        for i in range(1, len(seq)):
            prefix_seq = seq[:i]
            target_item = seq[i]
            padded_prefix = pad_sequence(prefix_seq, max_len)

            examples.append((user_id, padded_prefix, target_item))

    return examples


class SASRecTorchDataset(Dataset):
    """
    Each example looks like:
        (user_id, padded_input_seq, target_item)
    """

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        user_id, input_seq, target_item = self.examples[idx]

        input_seq_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_item_tensor = torch.tensor(target_item, dtype=torch.long)

        return input_seq_tensor, target_item_tensor


def make_train_dataloader(user_train, max_len=5, batch_size=32, shuffle=True):
    """
    Build SASRec training examples, wrap them in a Dataset,
    and return a PyTorch DataLoader.
    """
    examples = build_sasrec_examples(user_train, max_len=max_len)
    dataset = SASRecTorchDataset(examples)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader


def main():
    ratings_path = "data/raw/ratings.csv"

    # Load and preprocess
    df = load_ratings(ratings_path)
    user_sequences = build_user_sequences(df)
    user_train, user_val, user_test = split_user_sequences(user_sequences)

    # Create DataLoader
    train_loader = make_train_dataloader(
        user_train,
        max_len=5,
        batch_size=32,
        shuffle=True
    )

    # Inspect one batch
    for input_batch, target_batch in train_loader:
        print("Input batch shape:", input_batch.shape)
        print("Target batch shape:", target_batch.shape)
        print()
        print("Input batch:")
        print(input_batch)
        print()
        print("Target batch:")
        print(target_batch)
        break


if __name__ == "__main__":
    main()