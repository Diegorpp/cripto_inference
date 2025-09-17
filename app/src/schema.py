from pydantic import BaseModel


class TargetPrice(BaseModel):
    asset: str # Ticker: BTC, ETH, SOL, DOGE, etc.
    price: float # Valor do preço alvo: 20000, 30000, 100000
    currency: str # Moeda de referência: USD, BRL, EUR, etc.


class PercentageChange(BaseModel):
    asset: str # Ticker: BTC, ETH, SOL, DOGE, etc.
    percentage: float # Variação esperada: 50 (alta de 50%), -30 (queda de 30%)
    currency: str # Moeda de referência: USD, BRL, EUR, etc.


class Range(BaseModel):
    asset: str # Ticker: BTC, ETH, SOL, DOGE, etc.
    min: float # Limite inferior: 40000
    max: float # Limite superior: 60000
    currency: str # Moeda de referência: USD, BRL, EUR, etc.


class Ranking(BaseModel):
    asset: str # Ticker: BTC, ETH, SOL, DOGE, etc.
    ranking: int # Posição alvo: 1 (primeiro), 3 (terceiro), 10 (décimo)
    currency: str # Moeda de referência do ranking: USD, BRL, EUR


class Timeframe(BaseModel):
    explicit: bool # true se o post declarou prazo explícito
    start: str | None # timestamp ISO ex.: "2025-07-02T15:20:00Z"
    end: str | None # timestamp ISO ex.: "2025-09-30T23:59:59Z"


class Output(BaseModel):
    target_type: str # Tipo de target do post
    post_text: str # Texto do post
    extracted_value: TargetPrice | PercentageChange | Range | Ranking | None # Valores extraídos
    bear_bull: float | int # Escala: -100 (muito bearish) a +100 (muito bullish)
    timeframe: Timeframe # Prazo do post
    notes: list[str] | None # ex.: ["Quarter detectado no contexto", "Retweet atribuído ao autor original"]

class Tweet(BaseModel):
    text: str
    date: str

# Exemplo de saida
# {
#   "target_type": "ranking",
#   "extracted_value": {
#     "asset": "PEPE",
#     "ranking": 10,
#     "currency": "USD"
#   },
#   "timeframe": {
#     "explicit": false,
#     "start": null,
#     "end": null
#   },
#   "bear_bull": 65,
#   "notes": [
#     "Market cap ranking assumed",
#     "This cycle is vague timeframe",
#     "Quote tweet disagreeing with @bearish_analyst's bearish prediction",
#     "Frog and diamond emojis indicate strong bullish sentiment",
#     "USD market cap ranking context"
#   ]
# }