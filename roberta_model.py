import time
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW


class RoBERTaDataset(Dataset):
    # Dataset for RoBERTa sentiment classification with configurable truncation strategy.

    def __init__(self, df, tokenizer, max_length, truncation_strategy):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy

    def _tokenize_head_only(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            verbose=False
        )

    def _tokenize_head_tail(self, text):
        # Tokenize using a head+tail truncation strategy with max length
        # Tokenize without adding special tokens
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False
        )

        tokens = encoded["input_ids"]

        max_content_length = self.max_length - 2

        if len(tokens) > max_content_length:
            head_len = max_content_length // 2
            tail_len = max_content_length - head_len
            tokens = tokens[:head_len] + tokens[-tail_len:]

        # special tokens manually
        tokens = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(tokens)

        if len(tokens) < self.max_length:
            pad_len = self.max_length - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.truncation_strategy == "head-only":
            encoded = self._tokenize_head_only(text)
        elif self.truncation_strategy == "head+tail":
            encoded = self._tokenize_head_tail(text)
        else:
            raise ValueError(f"Unknown truncation strategy: {self.truncation_strategy}")

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

        return item


class RoBERTaSentimentRunner:
    # End-to-end RoBERTa fine-tuning runner
    def __init__(
        self,
        model_name="roberta-base",
        batch_size=8,
        num_epochs=1,
        learning_rate=2e-5
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        self.training_time_seconds = None
        self.inference_time_seconds = None

    def _build_loader(self, df, max_length, truncation_strategy, shuffle=False):
        dataset = RoBERTaDataset(
            df=df,
            tokenizer=self.tokenizer,
            max_length=max_length,
            truncation_strategy=truncation_strategy
        )

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, train_df, val_df, max_length, truncation_strategy, device=None):
        # Fine-tune RoBERTa for one configuration
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)

        train_loader = self._build_loader(
            df=train_df,
            max_length=max_length,
            truncation_strategy=truncation_strategy,
            shuffle=True
        )

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        start_time = time.time()

        self.model.train()
        for epoch in range(self.num_epochs):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                loss.backward()
                optimizer.step()

        end_time = time.time()
        self.training_time_seconds = end_time - start_time

    def predict_proba(self, df, max_length, truncation_strategy, device=None):
        # Predict positive-class probabilities for AUC computation
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)

        eval_loader = self._build_loader(
            df=df,
            max_length=max_length,
            truncation_strategy=truncation_strategy,
            shuffle=False
        )

        probabilities = []

        start_time = time.time()

        self.model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
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
