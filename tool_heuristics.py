# tool_heuristics.py

import os
import re
import json
import logging
import requests
import yfinance as yf
from datetime import datetime
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

import time
from datetime import datetime, timedelta

# Env laden (wichtig für lokalen Test)
load_dotenv()

# ==========================================
#               CONFIGURATION
# ==========================================

# API Client Setup
api_key = os.getenv("DEVELOPER_OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Standardized Error Messages
MSG_STOCK_NOT_FOUND = "I recognized the stock intent, but couldn't find a matching ticker."
MSG_CRYPTO_NOT_FOUND = "I recognized the crypto intent, but couldn't find a matching coin."
MSG_NO_SYMBOL_FOUND = "I recognized the intent, but couldn't extract a valid symbol or name."

# WMO Weather Codes (English)
WMO_CODES = {
    0: "Clear ☀️", 1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅", 3: "Overcast ☁️",
    45: "Fog 🌫️", 48: "Fog 🌫️", 51: "Drizzle 🌧️", 53: "Drizzle 🌧️", 55: "Drizzle 🌧️",
    61: "Rain 🌧️", 63: "Rain 🌧️", 65: "Heavy Rain 🌧️", 71: "Snow ❄️", 73: "Snow ❄️",
    75: "Snow ❄️", 80: "Showers 🌦️", 81: "Showers 🌦️", 82: "Violent Showers ⛈️",
    95: "Thunderstorm ⚡", 96: "Thunderstorm/Hail ⛈️", 99: "Thunderstorm/Hail ⛈️"
}

# Suffixes to remove for better stock search
COMPANY_SUFFIXES = [
    " inc", " corp", " corporation", " plc", " ltd", " limited",
    " ag", " se", " gmbh", " nv", " sa", " s.a."
]

# ==========================================
#               ROUTER PROMPT
# ==========================================

ROUTER_SYSTEM_PROMPT = """
You are a precise Intent Router for a real-time data system.
Analyze the user's question and decide if one of the following tools is absolutely necessary to answer it.

TOOLS:
- "weather": Current weather, forecasts, temperature (City needed).
- "stock": Stock prices, ETFs, Indices, Market Cap (Ticker/Company needed).
- "crypto": Cryptocurrency prices, Coins (Coin name/Symbol needed).

INSTRUCTIONS:
1. Identify the Intent.
2. Extract the SEARCH TERM (City, Company Name or Coin Name).
3. If no tool is needed (e.g. general knowledge, greetings), return null.
4. If ambiguous (e.g. "Solana"), prioritize Crypto. If it looks like a standard company, prioritize Stock.

OUTPUT JSON FORMAT:
{
  "tool": "weather" | "stock" | "crypto" | null,
  "query": "extracted_search_term_or_null"
}
"""

# ==========================================
#               SIMPLE CACHE
# ==========================================

# Struktur: { key: (timestamp, value) }
ROUTER_CACHE = {}
WEATHER_CACHE = {}
STOCK_CACHE = {}
CRYPTO_CACHE = {}

def _cache_get(cache: dict, key: str, max_age_seconds: int):
    """Return cached value or None if missing/expired."""
    now = time.time()
    entry = cache.get(key)
    if not entry:
        return None
    ts, value = entry
    if now - ts > max_age_seconds:
        # abgelaufen -> löschen
        cache.pop(key, None)
        return None
    return value

def _cache_set(cache: dict, key: str, value):
    """Store value in cache with current timestamp."""
    cache[key] = (time.time(), value)


def get_intent_from_llm(question: str) -> dict:
    """
    Calls gpt-4.1-mini to decide tool usage, with simple caching.
    """
    normalized_q = (question or "").strip().lower()
    if not normalized_q:
        return {"tool": None, "query": None}

    # 1. Cache-Check (z.B. 10 Minuten gültig)
    cached = _cache_get(ROUTER_CACHE, normalized_q, max_age_seconds=600)
    if cached is not None:
        return cached

    # 2. LLM-Call
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=100
        )
        
        response_content = completion.choices[0].message.content
        parsed = json.loads(response_content)

        # 3. In Cache legen
        _cache_set(ROUTER_CACHE, normalized_q, parsed)
        return parsed

    except Exception as e:
        logging.error(f"[Router] LLM call failed: {e}")
        return {"tool": None, "query": None}


# ==========================================
#              WEATHER LOGIC
# ==========================================

def get_weather_condition(code: int) -> str:
    """Helper to get text from WMO code."""
    return WMO_CODES.get(code, "Unknown")

def get_weather_data(city_name: str) -> str:
    """
    Fetches 3-day forecast from Open-Meteo, with caching.
    """
    key = (city_name or "").strip().lower()
    if not key:
        return "⚠️ Weather: Invalid city name."

    # 1. Cache-Check (z.B. 10 Minuten)
    cached = _cache_get(WEATHER_CACHE, key, max_age_seconds=600)
    if cached is not None:
        return cached

    try:
        # 2. Geocoding
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_res = requests.get(
            geo_url,
            params={"name": city_name, "count": 1, "language": "en", "format": "json"},
            timeout=5
        )
        geo_data = geo_res.json()

        if not geo_data.get("results"):
            result = f"⚠️ Weather: Could not find location '{city_name}'."
            _cache_set(WEATHER_CACHE, key, result)  # negative Cache
            return result

        location = geo_data["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        resolved_name = location["name"]
        country = location.get("country_code", "")

        # 3. Fetch Weather (3 Days)
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "current": ["temperature_2m", "weather_code"],
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_probability_max", "weather_code"],
            "timezone": "auto", "forecast_days": 3
        }
        w_res = requests.get(weather_url, params=params, timeout=5)
        w_data = w_res.json()

        current = w_data.get("current", {})
        daily = w_data.get("daily", {})

        cur_temp = current.get('temperature_2m')
        cur_cond = get_weather_condition(current.get('weather_code'))
        
        output_lines = [
            f"**Weather {resolved_name}, {country}:** {cur_cond}, {cur_temp}°C",
            "Forecast:"
        ]

        times = daily.get("time", [])
        codes = daily.get("weather_code", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        rain_probs = daily.get("precipitation_probability_max", [])

        for i in range(len(times)):
            date_obj = datetime.strptime(times[i], "%Y-%m-%d")
            day_name = date_obj.strftime("%a")
            cond = get_weather_condition(codes[i])
            line = f"> {day_name}: {cond} {min_temps[i]}°/{max_temps[i]}° ({rain_probs[i]}% rain)"
            output_lines.append(line)

        result = "\n".join(output_lines)
        _cache_set(WEATHER_CACHE, key, result)
        return result

    except Exception as e:
        logging.error(f"Weather fetch failed: {e}")
        return "⚠️ Error fetching weather data."


# ==========================================
#               STOCKS LOGIC
# ==========================================

TICKER_SEARCH_CACHE = {}

def search_ticker_symbol(search_term: str) -> Optional[str]:
    """
    Searches Yahoo Finance for a ticker, with caching.
    """
    try:
        raw = (search_term or "").strip()
        if not raw:
            return None

        # Normalisierte Key für Cache
        cache_key = raw.lower()
        cached = _cache_get(TICKER_SEARCH_CACHE, cache_key, max_age_seconds=3600)  # 1 Stunde
        if cached is not None:
            return cached  # darf auch None sein (negativer Cache)

        clean_search = re.sub(
            r'\s+(inc\.?|corp\.?|ag|se|gmbh|nv|sa|s\.a\.|ltd\.?|plc)\b', 
            '', 
            raw, 
            flags=re.IGNORECASE
        ).strip()

        url = "https://query2.finance.yahoo.com/v1/finance/search"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ConsensBot/1.0)"}
        params = {"q": clean_search, "quotesCount": 3, "newsCount": 0, "enableFuzzyQuery": "true"}

        res = requests.get(url, params=params, headers=headers, timeout=4)
        data = res.json()
        quotes = data.get("quotes") or []
        
        if not quotes:
            _cache_set(TICKER_SEARCH_CACHE, cache_key, None)
            return None

        symbol = None
        for q in quotes:
            qt = (q.get("quoteType") or "").lower()
            s = q.get("symbol")
            if qt in {"equity", "etf", "mutualfund", "index"} and s:
                symbol = s
                break

        if symbol is None:
            symbol = quotes[0].get("symbol")

        _cache_set(TICKER_SEARCH_CACHE, cache_key, symbol)
        return symbol

    except Exception as e:
        logging.error(f"[Stocks] Ticker search failed for '{search_term}': {e}")
        return None


def get_stock_data(query: str) -> str:
    """
    Main Stock Function: Search -> Fetch -> Format, with caching.
    """
    # normalize key for cache (unabhängig vom Ticker selbst, sondern der User-Query)
    cache_key = (query or "").strip().lower()
    if not cache_key:
        return MSG_STOCK_NOT_FOUND

    # 1. Cache-Check (z.B. 30 Sekunden – Kurse sind relativ dynamisch)
    cached = _cache_get(STOCK_CACHE, cache_key, max_age_seconds=30)
    if cached is not None:
        return cached

    # 2. Resolve Query to Ticker
    symbol = search_ticker_symbol(query)
    
    if not symbol:
        if len(query) <= 6 and query.isupper():
            symbol = query
        else:
            result = f"{MSG_STOCK_NOT_FOUND} (Searched for: '{query}')"
            _cache_set(STOCK_CACHE, cache_key, result)
            return result

    try:
        ticker = yf.Ticker(symbol)
        
        current_price = None
        prev_close = None
        currency = "USD"
        short_name = symbol

        # Attempt 1: fast_info
        try:
            fast_info = ticker.fast_info
            if fast_info:
                current_price = getattr(fast_info, "last_price", None)
                prev_close = getattr(fast_info, "previous_close", None)
                currency = getattr(fast_info, "currency", "USD")
        except Exception:
            pass

        # Attempt 2: standard info
        if current_price is None:
            info = ticker.info or {}
            current_price = info.get("regularMarketPrice") or info.get("previousClose")
            prev_close = prev_close or info.get("previousClose")
            currency = info.get("currency", currency)
            short_name = info.get("shortName", short_name)

        # Attempt 3: history
        if current_price is None:
            hist = ticker.history(period="2d")
            if not hist.empty:
                current_price = float(hist["Close"].iloc[-1])
                if len(hist) > 1:
                    prev_close = float(hist["Close"].iloc[-2])

        if current_price is None:
            result = f"📉 Could not retrieve data for {symbol}."
            _cache_set(STOCK_CACHE, cache_key, result)
            return result

        change_str = ""
        if prev_close and prev_close != 0:
            change_p = ((current_price - prev_close) / prev_close) * 100
            sign = "+" if change_p >= 0 else ""
            change_str = f" ({sign}{change_p:.2f}%)"

        result = (
            f"📈 **{short_name} ({symbol})**\n"
            f"Current Price: {current_price:.2f} {currency}{change_str}"
        )
        _cache_set(STOCK_CACHE, cache_key, result)
        return result

    except Exception as e:
        logging.error(f"[Stocks] Error fetching data for {symbol}: {e}")
        result = f"An error occurred while retrieving data for {symbol}."
        _cache_set(STOCK_CACHE, cache_key, result)
        return result


# ==========================================
#               CRYPTO LOGIC
# ==========================================

CRYPTO_SEARCH_CACHE = {}

def search_crypto_symbol(search_term: str) -> Optional[str]:
    """
    Uses CoinGecko Search to find an ID (e.g., "bitcoin") from a query, with caching.
    """
    try:
        raw = (search_term or "").strip()
        if not raw:
            return None

        cache_key = raw.lower()
        cached = _cache_get(CRYPTO_SEARCH_CACHE, cache_key, max_age_seconds=3600)
        if cached is not None:
            return cached  # darf None sein

        url = "https://api.coingecko.com/api/v3/search"
        res = requests.get(url, params={"query": search_term}, timeout=4)
        data = res.json()

        coins = data.get("coins") or []
        if not coins:
            _cache_set(CRYPTO_SEARCH_CACHE, cache_key, None)
            return None
        
        coin_id = coins[0].get("id")
        _cache_set(CRYPTO_SEARCH_CACHE, cache_key, coin_id)
        return coin_id

    except Exception as e:
        logging.error(f"[Crypto] Search failed: {e}")
        return None

def get_crypto_data(query: str) -> str:
    """
    Main Crypto Function: Search -> Fetch -> Format, with caching.
    """
    cache_key = (query or "").strip().lower()
    if not cache_key:
        return MSG_CRYPTO_NOT_FOUND

    # Cache-Check (z.B. 60 Sekunden)
    cached = _cache_get(CRYPTO_CACHE, cache_key, max_age_seconds=60)
    if cached is not None:
        return cached

    # 1. Resolve Query to Coin ID
    coin_id = search_crypto_symbol(query)
    
    if not coin_id:
        result = f"{MSG_CRYPTO_NOT_FOUND} (Searched for: '{query}')"
        _cache_set(CRYPTO_CACHE, cache_key, result)
        return result

    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": "usd", "ids": coin_id}
        res = requests.get(url, params=params, timeout=4)
        
        if res.status_code == 429:
            result = "⚠️ CoinGecko API limit reached. Please try again later."
            _cache_set(CRYPTO_CACHE, cache_key, result)
            return result
            
        data = res.json()

        if not data:
            result = f"🪙 No data found for '{coin_id}'."
            _cache_set(CRYPTO_CACHE, cache_key, result)
            return result

        c = data[0]
        name = c.get("name", coin_id)
        symbol = (c.get("symbol") or "").upper()
        price = c.get("current_price")
        change = c.get("price_change_percentage_24h")
        mc = c.get("market_cap")

        price_str = f"${price:,.2f}" if price is not None else "n/a"
        change_str = f"{change:+.2f}%" if change is not None else "n/a"
        mc_str = f"${mc:,.0f}" if mc is not None else "n/a"

        result = (
            f"🪙 **{name} ({symbol})**\n"
            f"Price: {price_str}\n"
            f"24h Change: {change_str}\n"
            f"Market Cap: {mc_str}"
        )
        _cache_set(CRYPTO_CACHE, cache_key, result)
        return result

    except Exception as e:
        logging.error(f"[Crypto] Data fetch error: {e}")
        result = f"Error fetching crypto data for '{coin_id}'."
        _cache_set(CRYPTO_CACHE, cache_key, result)
        return result


# ==========================================
#           MAIN ENTRY POINT
# ==========================================

def get_realtime_context(question: str) -> Optional[str]:
    """
    This is the ONE function called by main.py.
    """
    # 1. Router Call
    decision = get_intent_from_llm(question)
    tool = decision.get("tool")
    query = decision.get("query")
    
    if not tool or not query:
        return None
        
    logging.info(f"[Router] Tool: {tool}, Query: {query}")

    # 2. Dispatch
    try:
        if tool == "weather":
            return get_weather_data(query)
        elif tool == "stock":
            # Hier übergeben wir jetzt den Namen/Query, die Funktion sucht selbst
            return get_stock_data(query) 
        elif tool == "crypto":
            # Hier übergeben wir jetzt den Namen/Query, die Funktion sucht selbst
            return get_crypto_data(query)
            
    except Exception as e:
        logging.error(f"Tool execution failed: {e}")
        return None
    
    return None