import torch
from torch import nn
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    Trainer,
    TrainingArguments,
    RobertaForSequenceClassification,
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from models import classify_target_type
import time


# Definindo os labels
# Suas 5 classes
labels = ["target_price", "pct_change", "range", "ranking", "none"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")


def generate_labels():
    data = pd.read_json("../data/dataset.json")
    text = data["post_text"].tolist()
    # create item/label dict pair
    item_label = {}
    for idx, item in enumerate(text):
        if idx % 30 == 0 and idx != 0:
            print("Waiting 60 seconds...")
            time.sleep(60)
        label = classify_target_type(item)
        item_label[idx] = {"text": item, "label": label}

    # save item_label as json file
    pd.Series(item_label).to_json("../data/labels.json")


def load_dataset():
    df = pd.read_json("../data/labels.json")
    df = df.T.reset_index(drop=True)
    df = Dataset.from_pandas(df)
    return df


def load_dataset_mock():
    # Mock dataset
    # read dataset from json file in data/dataset.json
    data = {
        "text": [
            "Solana vai a $300 até o fim do ano",  # target_price
            "BTC vai cair 20% esse mês",  # pct_change
            "ETH vai ficar entre $1500 e $2000",  # range
            "SOL melhor que ADA nesse ciclo",  # ranking
            "Bom dia família cripto 🚀",  # none
        ],
        "label": ["target_price", "pct_change", "range", "ranking", "none"],
    }
    dataset = Dataset.from_dict(data)
    return dataset


def preprocess(example):
    enc = tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=64
    )
    enc["label"] = label2id[example["label"]]
    return enc


# def preprocess(example):
#     breakpoint()
#     enc = tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)
#     enc["label"] = label2id[example["label"]]
#     return enc


def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels_true = pred.label_ids
    report = classification_report(
        labels_true,
        preds,
        target_names=labels,
        labels=list(range(len(labels))),
        zero_division=0,
        output_dict=True,
    )
    return {"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]}


def custom_pipeline_classifier():

    dataset = load_dataset()
    # enc = preprocess(dataset[0])

    dataset = dataset.map(preprocess)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # Criação do modelo com as 5 saidas:
    model = RobertaForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # breakpoint()

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    predictions = trainer.predict(dataset["test"])
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)

    resultado = compute_metrics(predictions)
    print("Resultado:", resultado)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)


    trainer.save_model("./modelo_finetuned")  # salva modelo + tokenizer
    tokenizer.save_pretrained("./modelo_finetuned")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")  # Salva como PNG

    # plt.title("Confusion Matrix")
    # plt.show()


if __name__ == "__main__":
    custom_pipeline_classifier()
    # generate_labels()
    # load_dataset()
