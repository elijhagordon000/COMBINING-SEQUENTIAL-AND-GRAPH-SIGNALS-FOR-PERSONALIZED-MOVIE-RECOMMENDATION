import pandas as pd

def load_ratings(path):
    """
    Load MovieLens ratings file.
    """
    df = pd.read_csv(path)
    return df


def build_user_sequences(df, min_user_interactions=3):
    """
    Build chronological movie sequences for each user.

    Returns:
        user_sequences: dict
            key = userId
            value = list of movieIds in time order
    """
    # Keep only columns that are needed
    df = df[["userId", "movieId", "timestamp"]].copy()

    # Sort interactions by user, then time
    df = df.sort_values(["userId", "timestamp"])

    # Group movie interactions by user
    user_sequences = df.groupby("userId")["movieId"].apply(list).to_dict()

    # Filter out users with too few interactions
    filtered_sequences = {}
    for user_id, seq in user_sequences.items():
        if len(seq) >= min_user_interactions:
            filtered_sequences[user_id] = seq

    return filtered_sequences


def split_user_sequences(user_sequences):
    """
    Split each user's sequence into train / validation / test.
    """
    user_train = {}
    user_val = {}
    user_test = {}

    for user_id, seq in user_sequences.items():
        user_train[user_id] = seq[:-2]
        user_val[user_id] = seq[-2]
        user_test[user_id] = seq[-1]

    return user_train, user_val, user_test


def inspect_splits(user_train, user_val, user_test, num_users=5):
    """
    Print a few example users so you can sanity-check the split.
    """
    count = 0

    for user_id in user_train:
        print("User ID:", user_id)
        print("Train sequence:", user_train[user_id])
        print("Val item:", user_val[user_id])
        print("Test item:", user_test[user_id])
        print()

        count += 1
        if count == num_users:
            break


def main():
    ratings_path = "data/raw/ratings.csv"

    df = load_ratings(ratings_path)
    user_sequences = build_user_sequences(df)

    user_train, user_val, user_test = split_user_sequences(user_sequences)

    inspect_splits(user_train, user_val, user_test)


if __name__ == "__main__":
    main()