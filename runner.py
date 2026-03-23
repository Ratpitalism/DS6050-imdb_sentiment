"""
Project runner for:
- Logistic Regression with TF-IDF
- BiLSTM
- RoBERTa

1. Use the predefined IMDB train/test split
2. Create a 20,000 / 5,000 train/validation split
3. Stratify by BOTH sentiment label and review-length bin
4. Report accuracy, F1-score, and AUC
5. Report performance by short / medium / long review groups
6. Measure training and inference time
7. Save graphs and qualitative error-analysis examples
8. Run ablations for:
   - max input length T in {128, 256, 512}
   - truncation strategy in {head-only, head+tail}
"""

import os
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pprint import pprint
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dataset_utils import set_seed, load_imdb_data
from logistic_regression_model import LogisticRegressionSentimentModel
from bilstm_model import BiLSTMSentimentRunner
from roberta_model import RoBERTaSentimentRunner


# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Output folders
OUTPUT_DIR = "project_outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
ERRORS_DIR = os.path.join(OUTPUT_DIR, "errors")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(ERRORS_DIR, exist_ok=True)


# Metric helpers
def compute_metrics(y_true, y_pred, y_score=None):
    """
    Compute overall metrics for one set of predictions.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred))
    }

    if y_score is None:
        metrics["auc"] = None
    else:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["auc"] = None

    return metrics


def compute_metrics_by_length(df, y_pred, y_score=None):
    # Compute metrics separately for short / medium / long reviews.
    results = {}

    for length_name in ["short", "medium", "long"]:
        group_df = df[df["length_bin"] == length_name].copy()

        group_true = group_df["label"].values
        group_pred = np.array(y_pred)[group_df.index]

        if y_score is None:
            group_score = None
        else:
            group_score = np.array(y_score)[group_df.index]

        results[length_name] = compute_metrics(group_true, group_pred, group_score)

    return results


def package_results(model_name, split_name, overall_metrics, by_length_metrics,
                    training_time_seconds, inference_time_seconds,
                    max_length=None, truncation_strategy=None):
    # Flatten results into rows for CSV export.
    rows = []

    rows.append({
        "model": model_name,
        "split": split_name,
        "metric_scope": "overall",
        "length_bin": "all",
        "max_length": max_length,
        "truncation_strategy": truncation_strategy,
        "accuracy": overall_metrics["accuracy"],
        "f1": overall_metrics["f1"],
        "auc": overall_metrics["auc"],
        "training_time_seconds": training_time_seconds,
        "inference_time_seconds": inference_time_seconds
    })

    for length_name, metrics in by_length_metrics.items():
        rows.append({
            "model": model_name,
            "split": split_name,
            "metric_scope": "by_length",
            "length_bin": length_name,
            "max_length": max_length,
            "truncation_strategy": truncation_strategy,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
            "training_time_seconds": training_time_seconds,
            "inference_time_seconds": inference_time_seconds
        })

    return rows


# Error analysis
def save_misclassified_examples(df, y_pred, model_name, split_name, max_examples_per_bin=5):
    """
    Save a small set of misclassified examples by length group.
    """
    temp_df = df.copy()
    temp_df["pred"] = y_pred
    temp_df["correct"] = (temp_df["label"] == temp_df["pred"]).astype(int)

    output = {}

    for length_name in ["short", "medium", "long"]:
        group_df = temp_df[
            (temp_df["length_bin"] == length_name) &
            (temp_df["correct"] == 0)
        ].head(max_examples_per_bin)

        output[length_name] = []
        for _, row in group_df.iterrows():
            output[length_name].append({
                "true_label": int(row["label"]),
                "predicted_label": int(row["pred"]),
                "review_word_count": int(row["review_word_count"]),
                "review_token_count": int(row["review_token_count"]),
                "text": row["text"]
            })

    output_path = os.path.join(ERRORS_DIR, f"{model_name}_{split_name}_misclassified.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved qualitative error examples: {output_path}")


# Plot helpers
def make_primary_length_plot(results_df):
    # line plot with review-length group on the x-axis and F1-score on the y-axis.
    subset = results_df[
        (results_df["split"] == "test") &
        (results_df["metric_scope"] == "by_length")
    ].copy()

    # one final representative configuration per model family
    # BiLSTM / RoBERTa, use the highest test F1 among overall rows
    chosen_models = []

    overall_test = results_df[
        (results_df["split"] == "test") &
        (results_df["metric_scope"] == "overall")
    ].copy()

    # Logistic Regression has only one configuration
    if "Logistic Regression" in overall_test["model"].values:
        chosen_models.append("Logistic Regression")

    for family in ["BiLSTM", "RoBERTa"]:
        family_rows = overall_test[overall_test["model"].str.startswith(family)].copy()
        if len(family_rows) > 0:
            best_row = family_rows.sort_values("f1", ascending=False).iloc[0]
            chosen_models.append(best_row["model"])

    plot_df = subset[subset["model"].isin(chosen_models)].copy()
    plot_df["length_bin"] = pd.Categorical(
        plot_df["length_bin"],
        categories=["short", "medium", "long"],
        ordered=True
    )
    plot_df = plot_df.sort_values(["model", "length_bin"])

    plt.figure(figsize=(8, 5))
    for model_name in plot_df["model"].unique():
        model_df = plot_df[plot_df["model"] == model_name]
        plt.plot(model_df["length_bin"], model_df["f1"], marker="o", label=model_name)

    plt.title("F1-score by Review Length Group")
    plt.xlabel("Review Length Group")
    plt.ylabel("F1-score")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "primary_f1_by_length.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved primary figure: {output_path}")


def make_efficiency_plot(results_df):
    # Plot training time across the final chosen model configurations.
    overall_test = results_df[
        (results_df["split"] == "test") &
        (results_df["metric_scope"] == "overall")
    ].copy()

    chosen_rows = []

    logreg_rows = overall_test[overall_test["model"] == "Logistic Regression"]
    if len(logreg_rows) > 0:
        chosen_rows.append(logreg_rows.iloc[0])

    for family in ["BiLSTM", "RoBERTa"]:
        family_rows = overall_test[overall_test["model"].str.startswith(family)].copy()
        if len(family_rows) > 0:
            chosen_rows.append(family_rows.sort_values("f1", ascending=False).iloc[0])

    chosen_df = pd.DataFrame(chosen_rows)

    plt.figure(figsize=(8, 5))
    plt.bar(chosen_df["model"], chosen_df["training_time_seconds"])
    plt.title("Training Time by Model")
    plt.ylabel("Seconds")
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "training_time_by_model.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved efficiency plot: {output_path}")


def make_ablation_plot(ablation_df, model_family, metric="f1"):
    # Plot ablation performance vs max input length for one model family
    family_df = ablation_df[ablation_df["model_family"] == model_family].copy()

    plt.figure(figsize=(8, 5))

    for truncation_strategy in family_df["truncation_strategy"].unique():
        subset = family_df[family_df["truncation_strategy"] == truncation_strategy].copy()
        subset = subset.sort_values("max_length")
        plt.plot(subset["max_length"], subset[metric], marker="o", label=truncation_strategy)

    plt.title(f"{model_family}: {metric.upper()} vs Max Input Length")
    plt.xlabel("Max Input Length")
    plt.ylabel(metric.upper())
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, f"{model_family.lower()}_{metric}_ablation.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved ablation plot: {output_path}")



# Model runners
def run_logistic_regression(train_df, val_df, test_df):
    # Logistic Regression baseline
    print("\n" + "=" * 70)
    print("RUNNING LOGISTIC REGRESSION")
    print("=" * 70)

    model = LogisticRegressionSentimentModel(
        max_features=20000,
        max_iter=1000,
        random_state=42
    )

    model.fit(train_df)

    val_pred = model.predict(val_df)
    val_score = model.predict_proba(val_df)

    val_overall = compute_metrics(val_df["label"].values, val_pred, val_score)
    val_by_length = compute_metrics_by_length(val_df, val_pred, val_score)

    test_pred = model.predict(test_df)
    test_score = model.predict_proba(test_df)

    test_overall = compute_metrics(test_df["label"].values, test_pred, test_score)
    test_by_length = compute_metrics_by_length(test_df, test_pred, test_score)

    print("\nValidation Results")
    pprint(val_overall)
    pprint(val_by_length)

    print("\nTest Results")
    pprint(test_overall)
    pprint(test_by_length)

    save_misclassified_examples(test_df, test_pred, "logreg", "test")

    rows = []
    rows.extend(package_results(
        model_name="Logistic Regression",
        split_name="validation",
        overall_metrics=val_overall,
        by_length_metrics=val_by_length,
        training_time_seconds=model.training_time_seconds,
        inference_time_seconds=model.inference_time_seconds
    ))
    rows.extend(package_results(
        model_name="Logistic Regression",
        split_name="test",
        overall_metrics=test_overall,
        by_length_metrics=test_by_length,
        training_time_seconds=model.training_time_seconds,
        inference_time_seconds=model.inference_time_seconds
    ))

    return rows


def run_bilstm(train_df, val_df, test_df, max_length, truncation_strategy):
    # Run one BiLSTM ablation configuration
    print("\n" + "=" * 70)
    print(f"RUNNING BILSTM | max_length={max_length} | truncation={truncation_strategy}")
    print("=" * 70)

    model = BiLSTMSentimentRunner(
        max_vocab_size=20000,
        min_freq=2,
        embedding_dim=128,
        hidden_dim=128,
        batch_size=64,
        num_epochs=2,
        learning_rate=1e-3
    )

    model.fit(
        train_df=train_df,
        val_df=val_df,
        max_length=max_length,
        truncation_strategy=truncation_strategy,
        device=device
    )

    val_pred = model.predict(
        df=val_df,
        max_length=max_length,
        truncation_strategy=truncation_strategy,
        device=device
    )
    val_score = model.predict_proba(
        df=val_df,
        max_length=max_length,
        truncation_strategy=truncation_strategy,
        device=device
    )

    val_overall = compute_metrics(val_df["label"].values, val_pred, val_score)
    val_by_length = compute_metrics_by_length(val_df, val_pred, val_score)

    test_pred = model.predict(
        df=test_df,
        max_length=max_length,
        truncation_strategy=truncation_strategy,
        device=device
    )
    test_score = model.predict_proba(
        df=test_df,
        max_length=max_length,
        truncation_strategy=truncation_strategy,
        device=device
    )

    test_overall = compute_metrics(test_df["label"].values, test_pred, test_score)
    test_by_length = compute_metrics_by_length(test_df, test_pred, test_score)

    print("\nValidation Results")
    pprint(val_overall)
    pprint(val_by_length)

    print("\nTest Results")
    pprint(test_overall)
    pprint(test_by_length)

    save_misclassified_examples(
        df=test_df,
        y_pred=test_pred,
        model_name=f"bilstm_T{max_length}_{truncation_strategy}",
        split_name="test"
    )

    rows = []
    rows.extend(package_results(
        model_name=f"BiLSTM_T{max_length}_{truncation_strategy}",
        split_name="validation",
        overall_metrics=val_overall,
        by_length_metrics=val_by_length,
        training_time_seconds=model.training_time_seconds,
        inference_time_seconds=model.inference_time_seconds,
        max_length=max_length,
        truncation_strategy=truncation_strategy
    ))
    rows.extend(package_results(
        model_name=f"BiLSTM_T{max_length}_{truncation_strategy}",
        split_name="test",
        overall_metrics=test_overall,
        by_length_metrics=test_by_length,
        training_time_seconds=model.training_time_seconds,
        inference_time_seconds=model.inference_time_seconds,
        max_length=max_length,
        truncation_strategy=truncation_strategy
    ))

    ablation_row = {
        "model_family": "BiLSTM",
        "model_name": f"BiLSTM_T{max_length}_{truncation_strategy}",
        "max_length": max_length,
        "truncation_strategy": truncation_strategy,
        "accuracy": test_overall["accuracy"],
        "f1": test_overall["f1"],
        "auc": test_overall["auc"],
        "training_time_seconds": model.training_time_seconds,
        "inference_time_seconds": model.inference_time_seconds
    }

    return rows, ablation_row


def run_roberta(train_df, val_df, test_df, max_length, truncation_strategy):
    # Run one RoBERTa ablation configuration
    print("\n" + "=" * 70)
    print(f"RUNNING ROBERTA | max_length={max_length} | truncation={truncation_strategy}")
    print("=" * 70)

    model = RoBERTaSentimentRunner(
        model_name="roberta-base",
        batch_size=8,
        num_epochs=1,
        learning_rate=2e-5
    )

    model.fit(
        train_df=train_df,
        val_df=val_df,
        max_length=max_length,
        truncation_strategy=truncation_strategy,
        device=device
    )

    val_pred = model.predict(
        df=val_df,
        max_length=max_length,
        truncation_strategy=truncation_strategy,
        device=device
    )
    val_score = model.predict_proba(
        df=val_df,
        max_length=max_length,
        truncation_strategy=truncation_strategy,
        device=device
    )

    val_overall = compute_metrics(val_df["label"].values, val_pred, val_score)
    val_by_length = compute_metrics_by_length(val_df, val_pred, val_score)

    test_pred = model.predict(
        df=test_df,
        max_length=max_length,
        truncation_strategy=truncation_strategy,
        device=device
    )
    test_score = model.predict_proba(
        df=test_df,
        max_length=max_length,
        truncation_strategy=truncation_strategy,
        device=device
    )

    test_overall = compute_metrics(test_df["label"].values, test_pred, test_score)
    test_by_length = compute_metrics_by_length(test_df, test_pred, test_score)

    print("\nValidation Results")
    pprint(val_overall)
    pprint(val_by_length)

    print("\nTest Results")
    pprint(test_overall)
    pprint(test_by_length)

    save_misclassified_examples(
        df=test_df,
        y_pred=test_pred,
        model_name=f"roberta_T{max_length}_{truncation_strategy}",
        split_name="test"
    )

    rows = []
    rows.extend(package_results(
        model_name=f"RoBERTa_T{max_length}_{truncation_strategy}",
        split_name="validation",
        overall_metrics=val_overall,
        by_length_metrics=val_by_length,
        training_time_seconds=model.training_time_seconds,
        inference_time_seconds=model.inference_time_seconds,
        max_length=max_length,
        truncation_strategy=truncation_strategy
    ))
    rows.extend(package_results(
        model_name=f"RoBERTa_T{max_length}_{truncation_strategy}",
        split_name="test",
        overall_metrics=test_overall,
        by_length_metrics=test_by_length,
        training_time_seconds=model.training_time_seconds,
        inference_time_seconds=model.inference_time_seconds,
        max_length=max_length,
        truncation_strategy=truncation_strategy
    ))

    ablation_row = {
        "model_family": "RoBERTa",
        "model_name": f"RoBERTa_T{max_length}_{truncation_strategy}",
        "max_length": max_length,
        "truncation_strategy": truncation_strategy,
        "accuracy": test_overall["accuracy"],
        "f1": test_overall["f1"],
        "auc": test_overall["auc"],
        "training_time_seconds": model.training_time_seconds,
        "inference_time_seconds": model.inference_time_seconds
    }

    return rows, ablation_row


# Main
def main():
    set_seed(42)

    # device info
    print("=" * 70)
    print(f"USING DEVICE: {device}")
    if device.type == "cuda":
        print(f"GPU NAME: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # Hugging Face IMDB dataset b/c it is a standard benchmark for sentiment analysis and includes predefined train/test splits
    # make a stratified validation split from training

    train_df, val_df, test_df = load_imdb_data(
        validation_size=5000,
        random_state=42
    )

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    all_rows = []
    ablation_rows = []


    # Logistic Regression baseline
    all_rows.extend(run_logistic_regression(train_df, val_df, test_df))


    # BiLSTM ablations
    for max_length in [128, 256, 512]:
        for truncation_strategy in ["head-only", "head+tail"]:
            rows, ablation_row = run_bilstm(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                max_length=max_length,
                truncation_strategy=truncation_strategy
            )
            all_rows.extend(rows)
            ablation_rows.append(ablation_row)


    # RoBERTa ablations
    for max_length in [128, 256, 512]:
        for truncation_strategy in ["head-only", "head+tail"]:
            rows, ablation_row = run_roberta(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                max_length=max_length,
                truncation_strategy=truncation_strategy
            )
            all_rows.extend(rows)
            ablation_rows.append(ablation_row)


    # Save result tables
    results_df = pd.DataFrame(all_rows)
    ablation_df = pd.DataFrame(ablation_rows)

    results_path = os.path.join(TABLES_DIR, "all_model_results.csv")
    ablation_path = os.path.join(TABLES_DIR, "ablation_results.csv")

    results_df.to_csv(results_path, index=False)
    ablation_df.to_csv(ablation_path, index=False)

    print(f"Saved results table: {results_path}")
    print(f"Saved ablation table: {ablation_path}")


    # Save plots
    make_primary_length_plot(results_df)
    make_efficiency_plot(results_df)
    make_ablation_plot(ablation_df, model_family="BiLSTM", metric="f1")
    make_ablation_plot(ablation_df, model_family="RoBERTa", metric="f1")
    make_ablation_plot(ablation_df, model_family="BiLSTM", metric="auc")
    make_ablation_plot(ablation_df, model_family="RoBERTa", metric="auc")

    print("\n" + "=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)
    print("Saved:")
    print("- all model results")
    print("- ablation results")
    print("- primary F1-by-length plot")
    print("- efficiency plot")
    print("- ablation plots")
    print("- qualitative error examples")


if __name__ == "__main__":
    main()
