import re
from typing import List, Dict
from decimal import Decimal
from groq import Groq
import os

GROQ_KEY = os.getenv("GROQ_KEY")

# Primeiro identificar o tipo de alvo {target_price, pct_change, range, ranking, none}
def classify_target_type(text: str) -> None:
    """
    Classifica o tipo de alvo financeiro mencionado no texto.
    Retorna um dos valores: "target_price", "pct_change", "range", "ranking", "none"
    """
    client = Groq(api_key=GROQ_KEY)
    model = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um assistente que classifica previsões financeiras em tweets. "
                    "Classifique o tipo de previsão em uma das seguintes categorias: "
                    "'target_price' (preço alvo), 'pct_change' (variação percentual), "
                    "'range' (faixa de preços), 'ranking' (posição no ranking), ou 'none' (nenhum). "
                    "Responda apenas com a categoria, sem explicações."
                ),
            },
            {
                "role": "user",
                "content": f"Classifique o seguinte tweet: '{text}'",
            },
        ],
        max_tokens=10,
        temperature=0.0,
    )

    response = model.choices[0].message.content.lower().strip()
    valid_types = {"target_price", "pct_change", "range", "ranking"}

    return response if response in valid_types else 'none'

def extract_model_notes(tweet, contexto) -> List[str]:
    """
    Extrai notas relevantes do texto.
    Retorna uma lista de strings com as notas.
    """
    client = Groq(api_key=GROQ_KEY)
    model = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um especialista em mercado de cripto moedas e faz analise de mercado."
                    "Identifique caracteristicas do tweet em contraste com as informações enviadas pelo usuário:"
                    "Responda apenas com frases curtas"
                ),
            },
            {
                "role": "user",
                "content": f"Analise o seguinte tweet: '{tweet}' com o seguinte contexto: '{contexto}'",
            },
        ],
        # max_tokens=10,
        temperature=0.0,
    )
    response = model.choices[0]
    # response = model.choices[0].message.content.lower().strip()
    return response


# classify_target_type("BTC breaking $100k before Christmas! Mark my words")