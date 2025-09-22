import pandas as pd
import requests
from src.schema import Tweet

API_URL = "http://localhost:8000/parse_prediction"


def test_classify():
    import time

    df = pd.read_json("data/dataset.json")
    # for each item in df call classify_target_type
    texts = df["post_text"].tolist()
    datas = df["post_created_at"].tolist()
    result_correto = []
    result_errado = []
    for idx, _ in enumerate(texts):
        # Necessário se for usar o modulo do GROQ
        # For each 30 files, wait 1 minute
        # if idx % 30 == 0 and idx != 0:
        #     print("Waiting 60 seconds...")
        #     time.sleep(60)

        text = texts[idx]
        date = datas[idx]
        tweet = {"post_text": text, "post_created_at": str(date)}
        response = requests.post(API_URL, json=tweet)
        if response.status_code == 200:
            result_correto.append((text, response.status_code, response.text))
        else:
            result_errado.append((text, response.status_code, response.text))
    # escrever em arquivo de saida os resultados
    # separar em arquivos diferentes
    with open("data/result_correto.txt", "w") as f:
        for item in result_correto:
            f.write(f"Text: {item[0]}\nStatus Code: {item[1]}\nResponse: {item[2]}\n\n")
    with open("data/result_errado.txt", "w") as f:
        for item in result_errado:
            f.write(f"Text: {item[0]}\nStatus Code: {item[1]}\nResponse: {item[2]}\n\n")


if __name__ == "__main__":
    test_classify()
