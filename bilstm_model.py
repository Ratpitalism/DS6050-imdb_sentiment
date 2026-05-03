import time
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from dataset_utils import build_vocab, encode_text_for_bilstm


class BiLSTMDataset(Dataset):
    # PyTorch dataset for the BiLSTM model.

    def __init__(self, df, vocab, max_length, truncation_strategy):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.vocab = vocab
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = encode_text_for_bilstm(
            text=self.texts[idx],
            vocab=self.vocab,
            max_length=self.max_length,
            truncation_strategy=self.truncation_strategy
        )

        label = self.labels[idx]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )


class BiLSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_index=0, dropout=0.2):
        super(BiLSTMClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index
        )


        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )


        # Dropout and final classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)

        # output shape: (batch_size, sequence_length, hidden_dim * 2)
        # hidden_state shape: (2, batch_size, hidden_dim)
        output, (hidden_state, cell_state) = self.bilstm(embedded)

        forward_hidden = hidden_state[0]
        backward_hidden = hidden_state[1]

        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        final_hidden = self.dropout(final_hidden)

        logits = self.classifier(final_hidden).squeeze(1)
        return logits


class BiLSTMSentimentRunner:
    # End-to-end BiLSTM runner that matches the main runner.py interface.

    def __init__(
        self,
        max_vocab_size=20000,
        min_freq=2,
        embedding_dim=128,
        hidden_dim=128,
        batch_size=64,
        num_epochs=10,
        learning_rate=1e-3
    ):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.vocab = None
        self.model = None
        self.training_time_seconds = None
        self.inference_time_seconds = None

    def _build_loaders(self, train_df, val_df, max_length, truncation_strategy):
        # Build train and validation DataLoaders

        self.vocab = build_vocab(
            texts=train_df["text"].tolist(),
            max_vocab_size=self.max_vocab_size,
            min_freq=self.min_freq
        )

        train_dataset = BiLSTMDataset(train_df, self.vocab, max_length, truncation_strategy)
        val_dataset = BiLSTMDataset(val_df, self.vocab, max_length, truncation_strategy)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def _build_eval_loader(self, df, max_length, truncation_strategy):
        dataset = BiLSTMDataset(df, self.vocab, max_length, truncation_strategy)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def fit(self, train_df, val_df, max_length, truncation_strategy, device=None):
        # Train the BiLSTM model for one configuration
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader, val_loader = self._build_loaders(
            train_df=train_df,
            val_df=val_df,
            max_length=max_length,
            truncation_strategy=truncation_strategy
        )

        self.model = BiLSTMClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            pad_index=0
            ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        start_time = time.time()

        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_inputs, batch_labels in train_loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                logits = self.model(batch_inputs)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {avg_loss:.4f}")

        end_time = time.time()
        self.training_time_seconds = end_time - start_time

    def predict_proba(self, df, max_length, truncation_strategy, device=None):
        # Predict positive-class probabilities for a DataFrame
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model is None:
            raise ValueError("Call fit(...) before predict_proba(...).")

        eval_loader = self._build_eval_loader(df, max_length, truncation_strategy)

        self.model.eval()
        probabilities = []

        start_time = time.time()

        with torch.no_grad():
            for batch_inputs, _ in eval_loader:
                batch_inputs = batch_inputs.to(device)
                logits = self.model(batch_inputs)
                probs = torch.sigmoid(logits).cpu().numpy()
                probabilities.extend(probs.tolist())

        end_time = time.time()
        self.inference_time_seconds = end_time - start_time

        return np.array(probabilities)

    def predict(self, df, max_length, truncation_strategy, device=None):
        # Predict hard labels for a DataFrame
        probabilities = self.predict_proba(
            df=df,
            max_length=max_length,
            truncation_strategy=truncation_strategy,
            device=device
        )

        predictions = (probabilities >= 0.5).astype(int)
        return predictions
