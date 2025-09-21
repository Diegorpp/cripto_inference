# Funcionalidades de tratamento dos textos e extração de informações
import re
import requests
from decimal import Decimal
from bs4 import BeautifulSoup
from typing import List, Dict
from itertools import takewhile
import dateparser
from datetime import datetime
from typing import Literal
from random import randint
from src.schema import (
    TargetPrice, PercentageChange, Range, Ranking,
                        Timeframe, Output
                        )

TargetType = Literal["target_price", "pct_change", "range", "ranking", "none"]


# Obtem a lista de ativos do CoinMarketCap. 
def get_asset_list() -> list[str]:
    # Obtem a lista de ativos do CoinMarketCap
    link = 'https://coinmarketcap.com/pt-br'
    criptos = {}

    response = requests.get(link)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    tbody = soup.find('tbody')
    get_all_tr = tbody.find_all('tr')
    for idx, tr in enumerate(get_all_tr):
        get_all_td = tr.find_all('td')
        asset = get_all_td[2].text
        # Trata os casos basicos de ativos
        coin_asset = ''.join(takewhile(lambda x: x.isupper(), asset[::-1]))
        coin_name = asset[:-len(coin_asset)]
        
        if coin_name == '': # Pegar os casos pares.
            if len(asset) % 2 == 0:
                coin_asset = asset[len(asset)//2:]
                coin_name = asset[:len(asset)//2]
            else: # Pegar os casos impares
                coin_asset = asset[len(asset)//2 + 1:]
                coin_name = asset[:len(asset)//2 + 1]
        else:
            # Quando o ativo esta todo maiusculo
            coin_asset = coin_asset[::-1]

        # TODO: Algumas coins ainda não ficaram com o nome certo.
        # TODO: Fiz apenas para as primeiras 100 moedas das 9450.
        # breakpoint()
        criptos[coin_asset] = coin_name
    return criptos
        # print(f'{coin_name}: {coin_asset}')


def extract_asset(text: str, asset_list: list[str]) -> str | None:
    """
    Extrai o ativo (ticker) do texto baseado na lista fornecida.
    Retorna o ticker ou None se não encontrar.
    """

    for asset in asset_list:
        if asset in text:
            return asset
    return None


def extract_numbers(text: str) -> List[Dict]:
    """
    Extrai preços, ranges e percentuais de um texto.
    Retorna lista de dicts com tipo e valor normalizado.
    """
    results = []

    # Preços ($50,000, 50000, 50k)
    price_pattern = re.compile(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?k?", re.IGNORECASE)
    for match in price_pattern.findall(text):
        value = match.lower().replace(",", "")
        if "k" in value:
            value = value.replace("k", "000")
        value = value.replace("$", "")
        results.append({"type": "price", "value": str(Decimal(value))})
    # Definir a regra para qual valor deve ser considerado.
    return results

    # # Percentuais (10%, +5.5%, -12%)
    # pct_pattern = re.compile(r"[+-]?\d+(?:\.\d+)?%")
    # for match in pct_pattern.findall(text):
    #     results.append({"type": "percent", "value": match})

    # # Ranges ($100 - $200, 10% - 15%)
    # range_pattern = re.compile(r"(\$?\d+(?:,\d+)?(?:\.\d+)?)[\s-]+(\$?\d+(?:,\d+)?(?:\.\d+)?)")
    # for low, high in range_pattern.findall(text):
    #     results.append({
    #         "type": "range",
    #         "low": str(Decimal(low.replace("$", "").replace(",", ""))),
    #         "high": str(Decimal(high.replace("$", "").replace(",", "")))
    #     })


# TODO: Função incompleta
def extract_currency(text: str) -> str | None:
    """
    Extrai a moeda de referência do texto.
    Retorna o código da moeda (USD, BRL, EUR) ou None se não encontrar.
    """
    currency_map = {
        "usd": ["$", "usd", "dollar", "dólar"],
        "brl": ["brl", "real", "reais", "r$"],
        "eur": ["eur", "euro", "€"]
    }

    text_lower = text.lower()
    for code, keywords in currency_map.items():
        for keyword in keywords:
            if keyword in text_lower:
                return code.upper()
    return None


def extract_percent(text: str) -> float | None:
    """
    Extrai a variação percentual do texto.
    Retorna o valor percentual como float ou None se não encontrar.
    """
    pct_pattern = re.compile(r"([+-]?\d+(?:\.\d+)?)%")
    match = pct_pattern.search(text)
    if match:
        return float(match.group(1))
    return None


def extract_min_max(text: str) -> (float | None, float | None):
    """
    Extrai valores mínimo e máximo de um range no texto.
    Retorna uma tupla (min, max) ou (None, None) se não encontrar.
    """
    range_pattern = re.compile(r"(\$?\d+(?:,\d+)?(?:\.\d+)?)[\s-]+(\$?\d+(?:,\d+)?(?:\.\d+)?)")
    match = range_pattern.search(text)
    if match:
        low = float(match.group(1).replace("$", "").replace(",", ""))
        high = float(match.group(2).replace("$", "").replace(",", ""))
        return low, high
    return None, None


def extract_ranking(text: str) -> int | None:
    """
    Extrai a posição de ranking do texto.
    Retorna o ranking como int ou None se não encontrar.
    """
    # existing top words like references
    top_patterns = re.compile(r"top\s*(?:to|at|as)?\s*\d{1,3}", re.IGNORECASE)

    match = top_patterns.search(text)
    if match:
        top = match.group(0).split()[-1]
        return top
    return None


# TODO: Implementar as funções:
# Extract_currency
# Extract_numbers
# Extract_percent
def extract_info_based_on_type(target_type: str, text: str, asset: str) -> Dict:
    """
    Extrai informações específicas com base no tipo de alvo.
    Retorna um dict com os valores extraídos.
    """
    notes = []
    currency = extract_currency(text)
    if currency is None:
        currency = "USD"  # Default currency if none found
        notas_currency = "USD assumed as default currency" # TODO: Adicionar isso na lista de notas depois
        notes.append(notas_currency)

    if target_type == "target_price":
        numbers = extract_numbers(text)
        if len(numbers) >= 1: # Da pra criar uma função de logica para escolher qual valor utilizar
            numbers = numbers[0]['value']
        if numbers == []:
            numbers = None
            notas_price = "No price found"
            notes.append(notas_price)
        target_price = TargetPrice(
            asset=asset,
            price=numbers,
            currency=currency
        )
        return target_price

    elif target_type == "pct_change":
        percent = extract_percent(text)
        # breakpoint()
        percent_change = PercentageChange(
            asset=asset,
            percentage=percent,
            currency=currency  # Default currency, can be improved to extract from text
        )
        return percent_change

    elif target_type == "range":
        min, max = extract_min_max(text)
        # breakpoint()
        range = Range(
            asset=asset,
            min=min,
            max=max,
            currency=currency  # Default currency, can be improved to extract from text
        )
        return range

    elif target_type == "ranking":
        rank = extract_ranking(text)
        ranking = Ranking(
            asset=asset,
            ranking=rank,
            currency=currency  # Default currency, can be improved to extract from text
        )
        return ranking
    print('Nenhum tipo foi identificado')
    return None


def parse_datetime(text: str, base_time: datetime | None = None) -> datetime | None:
    """
    Faz parsing de qualquer string de data/tempo usando dateparser.
    - Aceita ISO8601 (com ou sem microssegundos, com 'Z' ou '+00:00')
    - Aceita expressões relativas ("mês que vem", "fim do Q3")
    - Normaliza sempre para UTC
    - Usa base_time como âncora para datas relativas
    """
    settings = {
        "RETURN_AS_TIMEZONE_AWARE": True,
        "TIMEZONE": "UTC",
        "TO_TIMEZONE": "UTC"
    }
    if base_time:
        settings["RELATIVE_BASE"] = base_time

    dt = dateparser.parse(text, settings=settings)

    if dt is None:
        return Timeframe(
            explicit=False,
            start=None,
            end=None
        )
    return Timeframe(
        explicit=True,
        start=base_time.isoformat(),
        end=dt.isoformat()
    )
    # return dt


def extract_time_expression(text: str) -> str | None:
    TIME_PATTERNS = [
        r"(fim do\s+Q[1-4])",
        r"(Q[1-4])",
        r"(next month)",
        r"(next week)",
        r"(next quarter)",
        r"(next year)",
        r"(in \d+ days?)",
        ]

    for pat in TIME_PATTERNS:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            # TODO: Adicionar inicio e fim quando identificado
            return match.group(1)
    return None


def extract_and_parse_time(text: str, base_time: datetime) -> Timeframe | None:
    trecho = extract_time_expression(text)
    if trecho:
        return parse_datetime(trecho, base_time)
    return Timeframe(
        explicit=False,
        start=None,
        end=None
    )


# dict_assets = get_asset_list()
# print("Assets:", dict_assets)

# print("Target Type:", classify_target_type(tweet["text"]))
# print("Numbers:", extract_numbers(tweet["text"]))
# print("Time:", normalize_time("mês que vem", tweet["created_at"]))
# print("Quarter:", normalize_quarter(tweet["text"], tweet["created_at"].year))
# print("Sentiment:", sentiment_score(tweet["text"]))