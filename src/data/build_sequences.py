import pandas as pd
from collections import defaultdict

def load_ratings(path):
    """
    Load MovieLens ratings file.
    """
    df = pd.read_csv(path)
    return df

def build_user_sequences(df,min_user_interactions=3):
    """
    Build chronological movie sequnces for each user.

    Returns:
        user_sequences: dict
            key = userId
            value = list of movieIds in time order
    """
    # Keep only columns that are needed 
    df = df[["userId", "movieId","timestamp"]].copy()

    # Sort interactions by user, then time
    df = df.sort_values(["userId", "timestamp"])

    # Group movie interactions by user
    user_sequences = df.groupby("userId")["movieId"].apply(list).to_dict()

    # Filter out users with too few interactions (Do not keep columns that have less than 3 ratings)
    filtered_sequences = {}
    for user_id, seq in user_sequences.items():
        if len(seq) >=  min_user_interactions:
            filtered_sequences[user_id] = seq
    return filtered_sequences

def split_user_sequences(user_sequences):
    """
    Split each user's sequence into train / validation / test

    For a sequence [i1, i2, i3, i4, i5]:
        train = [i1,i2,i3]
        val = i4
        test = i5

    Returns:
        user_train: dict
            key = userId, value = training sequence (list)
        user_val: dict
            key = userId, value = validation item (single movieId)
        user_test: dict 
            key  = userId, value = test item (single movieId)

    """
    #left off here

def main():
    ratings_path = "data/raw/ratings.csv"

    df = load_ratings(ratings_path)
    user_sequences = build_user_sequences(df)

    print("Number of users kept:", len(user_sequences))

    # Print a few examples
    count = 0
    for user_id, seq in user_sequences.items():
        print("User:", user_id, "Sequence length:", len(seq), "First few items:", seq[:10])
        count += 1
        if count == 5:
            break

if __name__ == "__main__":
    main()

