import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.data.build_sequences import (
    load_ratings,
    build_user_sequences,
    split_user_sequences,
)
from src.data.sasrec_dataset import make_train_dataloader, pad_sequence
from src.models.sasrec import SASRec


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

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate_validation_at_k(model, user_train, user_val, max_len, k, device):
    model.eval()

    total_recall = 0.0
    total_ndcg = 0.0
    num_users = 0

    with torch.no_grad():
        for user_id in user_train:
            train_seq = user_train[user_id]
            val_item = user_val[user_id]

            input_seq = pad_sequence(train_seq, max_len=max_len)
            input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

            logits = model(input_tensor)
            scores = logits[0].clone()

            # never recommend padding
            scores[0] = -1e9

            # mask already seen training items
            for item in set(train_seq):
                scores[item] = -1e9

            topk_items = torch.topk(scores, k=k).indices.tolist()

            if val_item in topk_items:
                total_recall += 1.0
                rank = topk_items.index(val_item) + 1   # 1-based rank
                total_ndcg += 1.0 / math.log2(rank + 1)

            num_users += 1

    recall_at_k = total_recall / num_users
    ndcg_at_k = total_ndcg / num_users

    return recall_at_k, ndcg_at_k

def evaluate_over_k_range(model, user_train, user_val, max_len, k_values, device):
    recall_scores = []
    ndcg_scores = []

    for k in k_values:
        recall_k, ndcg_k = evaluate_validation_at_k(
            model=model,
            user_train=user_train,
            user_val=user_val,
            max_len=max_len,
            k=k,
            device=device,
        )
        recall_scores.append(recall_k)
        ndcg_scores.append(ndcg_k)

    return recall_scores, ndcg_scores

def plot_metric_curves(k_values, recall_scores, ndcg_scores):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, recall_scores, marker='o')
    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title("Recall@K for SASRec")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, ndcg_scores, marker='s')
    plt.xlabel("K")
    plt.ylabel("NDCG@K")
    plt.title("NDCG@K for SASRec")
    plt.grid(True)
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ratings_path = "data/raw/ratings.csv"

    # Load data
    df = load_ratings(ratings_path)
    user_sequences = build_user_sequences(df)
    user_train, user_val, user_test = split_user_sequences(user_sequences)

    # Quick baseline: use raw max movieId
    num_items = int(df["movieId"].max())

    max_len = 5
    batch_size = 32
    hidden_dim = 64
    num_heads = 2
    num_layers = 2
    dropout = 0.2
    learning_rate = 1e-3
    num_epochs = 3
    k = 10

    train_loader = make_train_dataloader(
        user_train=user_train,
        max_len=max_len,
        batch_size=batch_size,
        shuffle=True,
    )

    model = SASRec(
        num_items=num_items,
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

        recall_at_k, ndcg_at_k = evaluate_validation_at_k(
            model=model,
            user_train=user_train,
            user_val=user_val,
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
        user_val=user_val,
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