import pandas as pd
from src.schema import Output, Tweet
from transformers import pipeline
from datetime import datetime
from src.models import classify_target_type, extract_model_notes
from src.custom_models import sentiment_bear_bull, custom_target_classify
from src.parsers import (
    get_asset_list,
    extract_asset,
    extract_info_based_on_type,
    extract_and_parse_time,
)
from fastapi import FastAPI
import os

BEAR_BULL_THRESHOLD = 0.3

app = FastAPI()

# Entrega:
# Serviço mínimo com endpoint POST /parse_prediction e README (setup/env/exemplos).
# Script de avaliação com métricas + matriz de confusão.
# Relatório de custo (p50, p95, batch=1 e batch=16, falhas).
# Arquivo tricky_cases.md com exemplos difíceis e explicação (é tudo bem ter tricky cases).


def processa_tweet(tweet: Tweet) -> Output:
    extra_notes = []

    # Primeiro identificar o tipo de alvo {target_price, pct_change, range, ranking, none}
    # target_type = classify_target_type(tweet.post_text)
    target_type = custom_target_classify(tweet)

    # Segundo identificar o tipo de asset {BTC, ETH, SOL, DOGE, etc.}
    asset_dict = get_asset_list()

    # Verifica se o ativo está no formato reduzido ou nome completo
    asset = extract_asset(tweet.post_text, list(asset_dict.keys()))
    if asset == None:
        asset = extract_asset(tweet.post_text, list(asset_dict.values()))
        if asset == None:
            asset = "the market asset was not identified"
            asset_note = "the market asset was not identified"

    # Terceiro Identificar os valores associados {price, currency, percentage, min, max, ranking}

    asset_value_info = extract_info_based_on_type(target_type, tweet.post_text, asset)
    # Quarto identificar o timeframe {explicit, start, end}
    dt = datetime.fromisoformat(tweet.post_created_at)

    # Normalização e extração do tempo está aqui
    timeframe = extract_and_parse_time(tweet.post_text, dt)

    # Quinto identificar o bear_bull (-100 a +100)
    bear_bull = sentiment_bear_bull(tweet)

    # TODO: Adicionar a parte do retweet
    # Sexto, identificar notas relevantes (ex.:
    # "Quarter detectado no contexto",
    # "Retweet atribuído ao autor original"
    # "Qual moeda foi assumida caso não esteja explícita no post"
    # "Conversão de prazos vagos para datas")

    # Sétimo, montar a saida no formato adequado.
    output = Output(
        post_text=tweet.post_text,
        target_type=target_type,
        extracted_value=asset_value_info,
        timeframe=timeframe,
        bear_bull=bear_bull,
        notes=None,
    )

    GROQ_KEY = os.getenv("GROQ_KEY")
    if GROQ_KEY:
        notes = extract_model_notes(tweet.post_text, contexto=str(output))
    else:
        extra_notes.append()

    output.notes = notes.message.content

    # print(output)
    return output
    # Oitavo criar uma API para receber o tweet e devolver a resposta.


@app.post("/parse_prediction", response_model=Output)
async def parse_prediction(tweet: Tweet):
    resultado = processa_tweet(tweet)
    return resultado


def main():
    df = pd.read_json("data/dataset.json")
    texts = df["post_text"].tolist()
    datas = df["post_created_at"].tolist()

    tweet = Tweet(post_text=texts[5], post_created_at=str(datas[5]))
    resultado = processa_tweet(tweet)
    print(resultado)


if __name__ == "__main__":
    main()
