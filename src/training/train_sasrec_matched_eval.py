import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from src.models.sasrec import SASRec
from src.data.sasrec_dataset import make_train_dataloader, pad_sequence


def load_ml1m_ratings():
    """
    Load ratings from the existing CSV file in this repo.
    """
    ratings_path = "data/raw/ratings.csv"

    df = pd.read_csv(ratings_path)

   
    df = df[["userId", "movieId", "timestamp"]].drop_duplicates()

   
    df = df.rename(columns={
        "userId": "user",
        "movieId": "movie"
    })

    return df


def reindex_ids(df):
    """
    Re-index users and movies so they are contiguous integers.
    Reserve 0 for padding.
    """
    df = df.copy()
    df["user"] = pd.Categorical(df["user"]).codes + 1
    df["movie"] = pd.Categorical(df["movie"]).codes + 1

    n_users = df["user"].nunique()
    n_movies = df["movie"].nunique()

    return df, n_users, n_movies


def build_train_histories(train_df, min_len=2):
    """
    Build chronological train histories from the train split only.
    """
    train_df = train_df.sort_values(["user", "timestamp"])
    user_train = train_df.groupby("user")["movie"].apply(list).to_dict()

    # Keep only users with enough train items to make SASRec examples
    user_train = {u: seq for u, seq in user_train.items() if len(seq) >= min_len}
    return user_train


def build_test_sets(test_df):
    """
    Build multiple held-out test items per user.
    """
    user_test = test_df.groupby("user")["movie"].apply(lambda x: set(x.tolist())).to_dict()
    return user_test


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for input_batch, target_batch in dataloader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()
        logits = model(input_batch)
        loss = criterion(logits, target_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_at_k_multi_item(model, user_train, user_test, max_len, k, device):
    """
    LightGCN-style evaluation:
    - multiple relevant test items per user
    - Recall@K
    - NDCG@K
    """
    model.eval()

    recalls = []
    ndcgs = []

    with torch.no_grad():
        for user_id in user_train:
            if user_id not in user_test:
                continue

            train_seq = user_train[user_id]
            true_items = user_test[user_id]

            if len(train_seq) == 0 or len(true_items) == 0:
                continue

            input_seq = pad_sequence(train_seq, max_len=max_len)
            input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

            logits = model(input_tensor)
            scores = logits[0].clone()

            # never recommend padding
            scores[0] = -1e9

            # mask already-seen train items
            for item in set(train_seq):
                scores[item] = -1e9

            top_k = torch.topk(scores, k=k).indices.tolist()

            # Recall@K
            hits = len(true_items.intersection(set(top_k)))
            recalls.append(hits / len(true_items))

            # NDCG@K
            dcg = 0.0
            for rank, item in enumerate(top_k):
                if item in true_items:
                    dcg += 1.0 / math.log2(rank + 2)

            idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(true_items), k)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(recalls)), float(np.mean(ndcgs))


def evaluate_over_k_range(model, user_train, user_test, max_len, k_values, device):
    recall_scores = []
    ndcg_scores = []

    for k in k_values:
        recall_k, ndcg_k = evaluate_at_k_multi_item(
            model=model,
            user_train=user_train,
            user_test=user_test,
            max_len=max_len,
            k=k,
            device=device,
        )
        recall_scores.append(recall_k)
        ndcg_scores.append(ndcg_k)

    return recall_scores, ndcg_scores


def plot_metric_curves(k_values, recall_scores, ndcg_scores):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, recall_scores, marker="o")
    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title("Recall@K for SASRec")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, ndcg_scores, marker="s")
    plt.xlabel("K")
    plt.ylabel("NDCG@K")
    plt.title("NDCG@K for SASRec")
    plt.grid(True)
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load MovieLens1M
    df = load_ml1m_ratings()
    df, n_users, n_movies = reindex_ids(df)

    print(f"Users: {n_users} | Movies: {n_movies} | Interactions: {len(df):,}")

    # Match LightGCN's random split idea
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Build train histories and multi-item test sets
    user_train = build_train_histories(train_df, min_len=2)
    user_test = build_test_sets(test_df)

    # Keep only users present in both
    common_users = set(user_train.keys()).intersection(set(user_test.keys()))
    user_train = {u: user_train[u] for u in common_users}
    user_test = {u: user_test[u] for u in common_users}

    print(f"Users used for SASRec train/eval: {len(common_users)}")

    max_len = 50
    batch_size = 128
    hidden_dim = 64
    num_heads = 2
    num_layers = 2
    dropout = 0.2
    learning_rate = 1e-3
    num_epochs = 5
    k = 20

    train_loader = make_train_dataloader(
        user_train=user_train,
        max_len=max_len,
        batch_size=batch_size,
        shuffle=True,
    )

    model = SASRec(
        num_items=n_movies,
        max_len=max_len,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        recall_at_k, ndcg_at_k = evaluate_at_k_multi_item(
            model=model,
            user_train=user_train,
            user_test=user_test,
            max_len=max_len,
            k=k,
            device=device,
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f} - "
            f"Recall@{k}: {recall_at_k:.4f} - "
            f"NDCG@{k}: {ndcg_at_k:.4f}"
        )

    k_values = list(range(1, 31))

    recall_scores, ndcg_scores = evaluate_over_k_range(
        model=model,
        user_train=user_train,
        user_test=user_test,
        max_len=max_len,
        k_values=k_values,
        device=device,
    )

    print("K values:", k_values)
    print("Recall scores:", recall_scores)
    print("NDCG scores:", ndcg_scores)

    plot_metric_curves(k_values, recall_scores, ndcg_scores)


if __name__ == "__main__":
    main()