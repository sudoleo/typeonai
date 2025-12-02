# tool_heuristics.py

import re
import logging
from typing import Optional, Tuple

import requests
import yfinance as yf
from datetime import datetime

# ==========================================
#              CONFIGURATION
# ==========================================

# Standardized Error Messages (constants used for logic checks)
MSG_STOCK_NOT_FOUND = "I recognized the stock intent, but couldn't find a matching ticker."
MSG_CRYPTO_NOT_FOUND = "I recognized the crypto intent, but couldn't find a matching coin."
MSG_NO_SYMBOL_FOUND = "I recognized the intent, but couldn't extract a valid symbol or name."

# WMO Weather Codes (English)
WMO_CODES = {
    0: "Clear ☀️",
    1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅", 3: "Overcast ☁️",
    45: "Fog 🌫️", 48: "Fog 🌫️",
    51: "Drizzle 🌧️", 53: "Drizzle 🌧️", 55: "Drizzle 🌧️",
    61: "Rain 🌧️", 63: "Rain 🌧️", 65: "Heavy Rain 🌧️",
    71: "Snow ❄️", 73: "Snow ❄️", 75: "Snow ❄️",
    80: "Showers 🌦️", 81: "Showers 🌦️", 82: "Violent Showers ⛈️",
    95: "Thunderstorm ⚡", 96: "Thunderstorm/Hail ⛈️", 99: "Thunderstorm/Hail ⛈️"
}

# ==========================================
#              WEATHER LOGIC
# ==========================================

WEATHER_INTENT_REGEX = re.compile(
    r"(?i)\b(wetter|weather|temperature|rain|sun|forecast|grad|degrees|snow|cloudy|wind|humidity|regen|sonne)\b"
)

# Verbessertes Regex: Erlaubt Kleinbuchstaben und Leerzeichen (z.B. "New York")
# Stoppt erst bei Satzzeichen oder Zeilenende.
LOCATION_REGEX = re.compile(
    r"(?i)\b(?:in|at|for|near|bei|für)\s+"      # Präposition
    r"([a-zäöüß\s\.-]+?)"                       # Der Ort (Capture Group)
    r"(?=\s+(?:tomorrow|morgen|today|heute|next|übermorgen)|[\?\.!,]|$)", # Stoppt vor Zeitwörtern oder Satzzeichen
    re.UNICODE
)

def get_weather_condition(code: int) -> str:
    """Helper to get text from WMO code."""
    return WMO_CODES.get(code, "Unbekannt")


def get_weather_data(city_name: str) -> str:
    """
    Fetches 3-day forecast from Open-Meteo.
    Output is token-optimized (compact list).
    """
    try:
        # 1. Geocoding
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_res = requests.get(
            geo_url,
            # 'language': 'en' ensures we get "Munich" instead of "München" for consistency
            params={"name": city_name, "count": 1, "language": "en", "format": "json"},
            timeout=5
        )
        geo_data = geo_res.json()

        if not geo_data.get("results"):
            return f"⚠️ Weather: Could not find location '{city_name}'."

        location = geo_data["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        resolved_name = location["name"]
        country = location.get("country_code", "") # Uses short code like "US" or "DE" to save tokens

        # 2. Fetch Weather (3 Days)
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "weather_code"],
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_probability_max", "weather_code"],
            "timezone": "auto",
            "forecast_days": 3
        }
        w_res = requests.get(weather_url, params=params, timeout=5)
        w_data = w_res.json()

        current = w_data.get("current", {})
        daily = w_data.get("daily", {})

        # Header with Current Weather
        cur_temp = current.get('temperature_2m')
        cur_cond = get_weather_condition(current.get('weather_code'))
        
        # Compact Header: "Weather London, GB: Overcast ☁️ 12.5°C"
        output_lines = [
            f"**Weather {resolved_name}, {country}:** {cur_cond}, {cur_temp}°C",
            "Forecast:"
        ]

        # Loop 3 days
        times = daily.get("time", [])
        codes = daily.get("weather_code", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        rain_probs = daily.get("precipitation_probability_max", [])

        for i in range(len(times)):
            # Convert date to short Day Name (Mon, Tue, Wed)
            date_obj = datetime.strptime(times[i], "%Y-%m-%d")
            day_name = date_obj.strftime("%a") # %a = Mon, Tue... (Short saves tokens)
            
            cond = get_weather_condition(codes[i])
            
            # Compact Line: "> Tue: Snow ❄️ -11°/-2° (20%)"
            line = f"> {day_name}: {cond} {min_temps[i]}°/{max_temps[i]}° ({rain_probs[i]}% rain)"
            output_lines.append(line)

        return "\n".join(output_lines)

    except Exception as e:
        logging.error(f"Weather fetch failed: {e}")
        return "⚠️ Error fetching weather data."


def inject_weather_context(question: str) -> Optional[str]:
    if not WEATHER_INTENT_REGEX.search(question):
        return None

    location_match = LOCATION_REGEX.search(question)
    if not location_match:
        return None

    # Strip whitespace
    city = location_match.group(1).strip()
    
    # ZUSATZ: Manuelles Herausfiltern bekannter Zeitwörter (Falls Regex versagt)
    ignore_words = ["tomorrow", "morgen", "today", "heute", "next week"]
    for word in ignore_words:
        if city.lower().endswith(" " + word):
            city = city[:-(len(word)+1)].strip()

    return get_weather_data(city)


# ==========================================
#           STOCKS & CRYPTO SHARED
# ==========================================

STOCK_INTENT_REGEX = re.compile(
    r"(?i)\b("
    r"stock|stocks|share|shares|equity|equities|dividend|dividends|ticker|isin|wkn|market\s*cap|earnings|pe\s*ratio"
    r"|price|quote|chart|value|valuation"
    r"|etf|fund|index|nasdaq|dow|sp500|dax"
    r"|costs|trading|listed"
    r")\b"
)

CRYPTO_INTENT_REGEX = re.compile(
    r"(?i)\b("
    r"crypto|cryptocurrency|coin|token|blockchain"
    r"|btc|eth|xrp|sol|ada|dot|bnb"
    r"|bitcoin|ethereum|ripple|solana|cardano|binance"
    r"|altcoin|memecoin"
    r")\b"
)

# Strong hints to resolve ambiguity
STOCK_STRONG_HINTS_REGEX = re.compile(
    r"(?i)\b(stock|shares|dividend|etf|index|corp|inc|ag|plc)\b"
)

CRYPTO_STRONG_HINTS_REGEX = re.compile(
    r"(?i)\b(crypto|coin|token|chain|btc|eth|sol|xrp)\b"
)

STOPWORDS = {
    # English
    "the", "a", "an", "current", "currently", "today", "now", "please", "show", "me", "tell",
    "what", "is", "are", "how", "much", "does", "cost", "value", "of", "for", "at", "in", "on",
    "price", "quote", "symbol", "ticker", "about",
    # German (included for mixed input robustness)
    "der", "die", "das", "den", "dem", "ein", "eine", "einen", "aktuell", "heute", "jetzt",
    "bitte", "zeig", "zeige", "mir", "uns", "sag", "sage",
    "wie", "viel", "was", "kostet", "steht", "wert", "von", "für", "bei",
    "aktie", "kurs", "preis",
    # Fillers
    "just", "approx", "about"
    # NEW: Quantities & Numbers
    "one", "two", "three", "amount", "quantity", "anzahl", "1", "10", "100"
}

COMPANY_SUFFIXES = [
    " inc", " corp", " corporation", " plc", " ltd", " limited",
    " ag", " se", " gmbh", " nv", " sa", " s.a."
]

def clean_query(text: str) -> str:
    """
    Removes punctuation and stopwords to isolate the search term.
    """
    # Keep $ for tickers, remove other punctuation
    text = re.sub(r"[^\w\s\-\$]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    cleaned_words = [w for w in words if w.lower() not in STOPWORDS]

    if not cleaned_words:
        return ""

    return " ".join(cleaned_words).strip()


# ==========================================
#               STOCKS LOGIC
# ==========================================

def extract_potential_symbol(text: str) -> Optional[str]:
    """
    Tries to find specific ticker patterns ($TSLA, ISIN, WKN).
    """
    # 1. Explicit Ticker ($TSLA)
    match_explicit = re.search(r"\$([A-Za-z.\-]{1,6})\b", text)
    if match_explicit:
        return match_explicit.group(1).upper()

    # 2. ISIN
    match_isin = re.search(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b", text)
    if match_isin:
        return match_isin.group(1)

    # 3. WKN (6 chars, alphanum, usually uppercase) - heuristic
    match_wkn = re.search(r"\b([A-Z0-9]{6})\b", text, flags=re.IGNORECASE)
    if match_wkn:
        # Filter out common English words that are 6 letters (e.g. "PLEASE")
        candidate = match_wkn.group(1).upper()
        if candidate not in {"PLEASE", "THANKS", "STOCKS", "CRYPTO"}:
            return candidate

    return None

def search_ticker_symbol(search_term: str) -> Optional[str]:
    """
    Searches Yahoo Finance for a ticker.
    """
    try:
        clean_search = search_term.strip()
        lower = clean_search.lower()
        
        # Remove suffixes for better search results
        for suffix in COMPANY_SUFFIXES:
            if lower.endswith(suffix):
                clean_search = clean_search[:-(len(suffix))].strip()
                break

        url = "https://query2.finance.yahoo.com/v1/finance/search"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ConsensBot/1.0)"}
        params = {
            "q": clean_search,
            "quotesCount": 3,
            "newsCount": 0,
            "enableFuzzyQuery": "true"
        }

        res = requests.get(url, params=params, headers=headers, timeout=4)
        data = res.json()

        quotes = data.get("quotes") or []
        if not quotes:
            return None

        # Prioritize Equity/ETFs
        for q in quotes:
            qt = (q.get("quoteType") or "").lower()
            symbol = q.get("symbol")
            if qt in {"equity", "etf", "mutualfund", "index"} and symbol:
                return symbol

        # Fallback
        return quotes[0].get("symbol")

    except Exception as e:
        logging.error(f"[Stocks] Ticker search failed for '{search_term}': {e}")
        return None

def get_stock_data(symbol: str) -> str:
    """
    Fetches price data via yfinance.
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Initialize variables
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

        # Attempt 3: history (fallback)
        if current_price is None:
            hist = ticker.history(period="2d")
            if not hist.empty:
                current_price = float(hist["Close"].iloc[-1])
                if len(hist) > 1:
                    prev_close = float(hist["Close"].iloc[-2])

        if current_price is None:
            return f"📉 Could not retrieve data for {symbol} at the moment."

        # Calculate Change
        change_str = ""
        if prev_close and prev_close != 0:
            change_p = ((current_price - prev_close) / prev_close) * 100
            sign = "+" if change_p >= 0 else ""
            change_str = f" ({sign}{change_p:.2f}%)"

        return (
            f"📈 **{short_name} ({symbol})**\n"
            f"Current Price: {current_price:.2f} {currency}{change_str}"
        )

    except Exception as e:
        logging.error(f"[Stocks] Error fetching data for {symbol}: {e}")
        return f"An error occurred while retrieving data for {symbol}."

def inject_stock_context(question: str) -> str:
    """
    Handles stock queries. Returns a string message (data or error).
    """
    if not STOCK_INTENT_REGEX.search(question):
        return "" # Should not happen if called correctly

    ticker_candidate = extract_potential_symbol(question)
    search_term = None
    found_symbol = None

    if ticker_candidate:
        search_term = ticker_candidate
        # Check if it's already a valid symbol via search
        found_symbol = search_ticker_symbol(search_term)
        # If search returns nothing but we had a short explicit ticker, try strictly that
        if not found_symbol and len(ticker_candidate) <= 6:
            found_symbol = ticker_candidate
    else:
        search_term = clean_query(question)
        if len(search_term) > 1:
            found_symbol = search_ticker_symbol(search_term)
        else:
            return MSG_NO_SYMBOL_FOUND

    if found_symbol:
        return get_stock_data(found_symbol)
    
    # Return specific failure message for the router to detect
    if search_term:
        return f"{MSG_STOCK_NOT_FOUND} (Search term: '{search_term}')"
    
    return MSG_STOCK_NOT_FOUND


# ==========================================
#               CRYPTO LOGIC
# ==========================================

CRYPTO_NAME_MAP = {
    "bitcoin": "btc", "ethereum": "eth", "solana": "sol", "ripple": "xrp",
    "cardano": "ada", "binance": "bnb", "polkadot": "dot", "dogecoin": "doge"
}

def extract_crypto_symbol(text: str) -> Optional[str]:
    """
    Extracts crypto symbol candidates.
    Handles punctuation correctly (e.g. "btc?" -> "btc").
    """
    # 1. Explicit Ticker ($BTC)
    match_explicit = re.search(r"\$([A-Za-z0-9]{2,10})\b", text)
    if match_explicit:
        return match_explicit.group(1).lower()

    lower_text = text.lower()
    
    # FIX: Nutze Regex findall statt split().
    # \w+ findet nur Wörter (Buchstaben/Zahlen) und ignoriert ?, !, ., etc.
    tokens = set(re.findall(r"\w+", lower_text))

    # 2. Check Name AND Symbol in Map
    for name, symbol in CRYPTO_NAME_MAP.items():
        # Check if full name is in text (e.g. "bitcoin price")
        if name in lower_text:
            return symbol
        # Check if symbol is a standalone token (e.g. "one btc?")
        if symbol in tokens:
            return symbol

    # 3. Uppercase tokens (SOL) - Fallback
    if not text.islower():
        match_upper = re.search(r"\b([A-Z]{2,10})\b", text)
        if match_upper:
            candidate = match_upper.group(1).lower()
            if candidate not in {"USD", "EUR", "ETF", "NOW", "ONE"}:
                return candidate
    return None

def search_crypto_symbol(search_term: str) -> Optional[str]:
    """
    Uses CoinGecko Search to find an ID (e.g., "bitcoin") from a query.
    """
    try:
        url = "https://api.coingecko.com/api/v3/search"
        res = requests.get(url, params={"query": search_term}, timeout=4)
        data = res.json()

        coins = data.get("coins") or []
        if not coins:
            return None
        
        # Return the API ID of the top result
        return coins[0].get("id")

    except Exception as e:
        logging.error(f"[Crypto] Search failed: {e}")
        return None

def get_crypto_data(coin_id: str) -> str:
    """
    Fetches market data from CoinGecko.
    """
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": "usd", "ids": coin_id}
        res = requests.get(url, params=params, timeout=4)
        
        if res.status_code == 429:
            return "⚠️ CoinGecko API limit reached. Please try again later."
            
        data = res.json()

        if not data:
            return f"🪙 No data found for '{coin_id}'."

        c = data[0]
        name = c.get("name", coin_id)
        symbol = (c.get("symbol") or "").upper()
        price = c.get("current_price")
        change = c.get("price_change_percentage_24h")
        mc = c.get("market_cap")

        price_str = f"${price:,.2f}" if price is not None else "n/a"
        change_str = f"{change:+.2f}%" if change is not None else "n/a"
        mc_str = f"${mc:,.0f}" if mc is not None else "n/a"

        return (
            f"🪙 **{name} ({symbol})**\n"
            f"Price: {price_str}\n"
            f"24h Change: {change_str}\n"
            f"Market Cap: {mc_str}"
        )

    except Exception as e:
        logging.error(f"[Crypto] Data fetch error: {e}")
        return f"Error fetching crypto data for '{coin_id}'."

def inject_crypto_context(question: str) -> str:
    if not CRYPTO_INTENT_REGEX.search(question):
        return ""

    coin_candidate = extract_crypto_symbol(question)
    search_term = None
    coin_id = None

    if coin_candidate:
        search_term = coin_candidate
        coin_id = search_crypto_symbol(search_term)
    else:
        search_term = clean_query(question)
        if len(search_term) > 1:
            coin_id = search_crypto_symbol(search_term)
        else:
            return MSG_NO_SYMBOL_FOUND

    if coin_id:
        return get_crypto_data(coin_id)
    
    if search_term:
        return f"{MSG_CRYPTO_NOT_FOUND} (Search term: '{search_term}')"

    return MSG_CRYPTO_NOT_FOUND


# ==========================================
#             MASTER ROUTER
# ==========================================

def inject_market_context(question: str) -> Optional[str]:
    """
    Main entry point. Decides between Stocks and Crypto.
    """
    stock_intent = bool(STOCK_INTENT_REGEX.search(question))
    crypto_intent = bool(CRYPTO_INTENT_REGEX.search(question))

    if not (stock_intent or crypto_intent):
        return None

    logging.info(f"[Market] Intent: Stocks={stock_intent}, Crypto={crypto_intent}")

    # 1. Unambiguous Intents
    if stock_intent and not crypto_intent:
        return inject_stock_context(question)
    
    if crypto_intent and not stock_intent:
        return inject_crypto_context(question)

    # 2. Ambiguous - Check strong hints
    if CRYPTO_STRONG_HINTS_REGEX.search(question):
        return inject_crypto_context(question)
    
    if STOCK_STRONG_HINTS_REGEX.search(question):
        return inject_stock_context(question)

    # 3. Ambiguous - Try Stock first, then Crypto
    logging.info("[Market] Ambiguous query. Trying Stock first.")
    stock_response = inject_stock_context(question)

    # Check against constants if the stock lookup failed
    if stock_response.startswith(MSG_STOCK_NOT_FOUND) or stock_response == MSG_NO_SYMBOL_FOUND:
        logging.info("[Market] Stock lookup failed. Fallback to Crypto.")
        crypto_response = inject_crypto_context(question)
        
        # If crypto also fails, just return the Stock error message 
        # (or a generic one) to avoid confusing the user too much.
        if crypto_response.startswith(MSG_CRYPTO_NOT_FOUND) or crypto_response == MSG_NO_SYMBOL_FOUND:
             return stock_response 
        
        return crypto_response

    return stock_response