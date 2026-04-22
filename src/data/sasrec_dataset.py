from src.data.build_sequences import load_ratings, build_user_sequences, split_user_sequences
import torch
from torch.utils.data import Dataset


def pad_sequence(seq, max_len, pad_token=0):
    """
    Pad a sequence on the left with pad_token so that its length becomes max_len.
    """
    if len(seq) > max_len:
        seq = seq[-max_len:]

    if len(seq) < max_len:
        seq = [pad_token] * (max_len - len(seq)) + seq

    return seq


def build_sasrec_examples(user_train, max_len=5):
    """
    Convert user training sequences into SASRec-style training examples.
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
    Simple PyTorch dataset for SASRec training examples.

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


def main():
    ratings_path = "data/raw/ratings.csv"

    df = load_ratings(ratings_path)
    user_sequences = build_user_sequences(df)
    user_train, user_val, user_test = split_user_sequences(user_sequences)

    examples = build_sasrec_examples(user_train, max_len=5)

    dataset = SASRecTorchDataset(examples)

    print("Dataset length:", len(dataset))
    print("First item:", dataset[0])
    print("Second item:", dataset[1])
    print("Third item:", dataset[2])


if __name__ == "__main__":
    main()