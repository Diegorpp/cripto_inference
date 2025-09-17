import pandas as pd
from src.schema import Output, Tweet
from transformers import pipeline
from datetime import datetime
from src.models import classify_target_type, extract_model_notes
from src.parsers import ( get_asset_list, extract_asset,
                         extract_info_based_on_type, extract_and_parse_time
                         )

# Entrega:
# Serviço mínimo com endpoint POST /parse_prediction e README (setup/env/exemplos).
# Script de avaliação com métricas + matriz de confusão.
# Relatório de custo (p50, p95, batch=1 e batch=16, falhas).
# Arquivo tricky_cases.md com exemplos difíceis e explicação (é tudo bem ter tricky cases).

def processa_tweet(tweet: Tweet) -> Output:
    # Primeiro identificar o tipo de alvo {target_price, pct_change, range, ranking, none}
    target_type = classify_target_type(tweet.text)

    # Segundo identificar o tipo de asset {BTC, ETH, SOL, DOGE, etc.}
    asset_dict = get_asset_list()
    
    # Verifica se o ativo está no formato reduzido ou nome completo
    asset = extract_asset(tweet.text, list(asset_dict.keys()))
    if asset == None:
        asset = extract_asset(tweet.text, list(asset_dict.values()))
        if asset == None:
            asset = "the market asset was not identified"

    # Terceiro Identificar os valores associados {price, currency, percentage, min, max, ranking}

    asset_value_info = extract_info_based_on_type(target_type, tweet.text, asset)
    # Quarto identificar o timeframe {explicit, start, end}
    dt = datetime.fromisoformat(tweet.date)

    timeframe = extract_and_parse_time(tweet.text, dt)

    # Quinto identificar o bear_bull (-100 a +100)
    pipeline_model = pipeline(model="ProsusAI/finbert", task="text-classification")
    result = pipeline_model(tweet.text, top_k=3)
    # get percent from all classes
    # Positivo = 0, negativo = 1, neutro = 2
    # If neutral is higher than 50%, then bear_bull = 0
    # find neutral score
    for idx, item in enumerate(result):
        if item['label'] == 'neutral':
            if item['score'] > 0.5:
                neutral = idx
                break
    if result[neutral]['score'] > 0.5:
        bear_bull = 0
    else:
        bear_bull = result[0]['score'] * 100 - result[1]['score'] * 100 # Estou ignorando o caso neutro a principio
    print(result)

    # Sexto, identificar notas relevantes (ex.: 
    # "Quarter detectado no contexto", 
    # "Retweet atribuído ao autor original"
    # "Qual moeda foi assumida caso não esteja explícita no post"
    # "Conversão de prazos vagos para datas")


    # Sétimo, montar a saida no formato adequado.
    output =  Output(
        post_text = tweet.text,
        target_type = target_type, 
        extracted_value = asset_value_info, 
        timeframe = timeframe, 
        bear_bull = bear_bull, 
        notes=None,
    )

    notes = extract_model_notes(tweet.text, contexto=str(output))
    
    output.notes = notes.message.content

    print(output)
    return output
    # Oitavo criar uma API para receber o tweet e devolver a resposta.

def main():
    df = pd.read_json("data/dataset.json")
    texts = df["post_text"].tolist()
    datas = df["post_created_at"].tolist()

    tweet = Tweet(
        text=texts[5],
        date=str(datas[5])
    )
    resultado = processa_tweet(tweet)
    print(resultado)


if __name__ == "__main__":
    main()