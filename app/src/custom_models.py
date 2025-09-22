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
from src.models import classify_target_type
import time
from pathlib import Path
from transformers import pipeline
from src.schema import Tweet

# Definindo os labels
# Suas 5 classes
labels = ["target_price", "pct_change", "range", "ranking", "none"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
BEAR_BULL_THRESHOLD = 0.5


tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")


def model_init():
    return RobertaForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )


def hp_space(trial):
    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [2e-5, 3e-5, 5e-5]),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [2, 4, 8]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [3, 4, 5]),
    }


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
        # model=model,
        model_init=model_init,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    best_run = trainer.hyperparameter_search(
        direction="maximize",      # queremos maximizar f1/accuracy
        hp_space=hp_space,
        n_trials=5,                # quantos conjuntos de hiperparâmetros testar
    )

    print("Best hyperparameters:", best_run.hyperparameters)

    # Atualiza args
    for key, value in best_run.hyperparameters.items():
        setattr(trainer.args, key, value)

    trainer.train()

    predictions = trainer.predict(dataset["test"])
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)

    resultado = compute_metrics(predictions)
    print("Resultado:", resultado)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)


    trainer.save_model("./finetuned_models")  # salva modelo + tokenizer
    tokenizer.save_pretrained("./finetuned_models")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")  # Salva como PNG

    # plt.title("Confusion Matrix")
    # plt.show()



def custom_target_classify(tweet: Tweet) -> str:
    diretorio = Path("./src/finetuned_models")
    MODEL_PATH = "./src/finetuned_models"
    if diretorio.exists() and diretorio.is_dir():
        custom_model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
        new_tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        pipeline_model = pipeline(task="text-classification", model=custom_model, tokenizer=new_tokenizer)
        result = pipeline_model(tweet.post_text, top_k=5)
        # find highest value
        score = result[0]
        for item in result:
            if item['score'] > score['score']:
                score = item['score']
        return score['label']
    else:
        return None
        print('Modelo customizado não encontrado, definir um modelo padrão')

def sentiment_bear_bull(tweet: Tweet) -> float | int:
    """
    Analisa o sentimento do tweet e retorna um valor entre -100 (muito bearish) e +100 (muito bullish).
    """

    # diretorio = Path("./src/finetuned_models")
    # MODEL_PATH = "./src/finetuned_models"
    # if diretorio.exists() and diretorio.is_dir():
    #     custom_model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    #     new_tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    #     pipeline_model = pipeline(task="text-classification", model=custom_model, tokenizer=new_tokenizer)
    #     result = pipeline_model(tweet.post_text, top_k=3)
    # else:
    pipeline_model = pipeline(model="ProsusAI/finbert", task="text-classification")
    result = pipeline_model(tweet.post_text, top_k=3)
    # get percent from all classes
    # Positivo = 0, negativo = 1, neutro = 2
    # If neutral is higher than 50%, then bear_bull = 0
    # find neutral score
    neutral = 0
    for idx, item in enumerate(result):
        if item['label'] == 'neutral':
            if item['score'] > BEAR_BULL_THRESHOLD:
                neutral = idx
                break
    
    print(f'resultado bear bull: {result}')
    if result[neutral]['score'] > BEAR_BULL_THRESHOLD:
        bear_bull = 0
    else:
        bear_bull = result[0]['score'] * 100 - result[1]['score'] * 100 # Estou ignorando o caso neutro a principio
    return bear_bull


if __name__ == "__main__":
    custom_pipeline_classifier()
    # generate_labels()
    # load_dataset()
