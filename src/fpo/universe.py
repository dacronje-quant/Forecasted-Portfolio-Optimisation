"""Investable universe and benchmark definitions (Yahoo Finance tickers).

The universe is a hand-picked list of liquid large caps. NOTE: using today's
list of large caps to backtest the past introduces *survivorship bias* (the
companies that failed or shrank are missing). The backtester mitigates this by
always comparing against an equal-weight portfolio of the *same* universe and
against random portfolios drawn from it, which isolates stock-selection skill
from the universe itself. See README for details.
"""

from __future__ import annotations

US_STOCKS: dict[str, str] = {
    # Technology
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "META": "Meta Platforms", "AVGO": "Broadcom", "ORCL": "Oracle",
    "CSCO": "Cisco", "ADBE": "Adobe", "CRM": "Salesforce", "INTC": "Intel",
    "AMD": "AMD", "QCOM": "Qualcomm", "TXN": "Texas Instruments", "IBM": "IBM",
    "NFLX": "Netflix", "TSLA": "Tesla",
    # Financials
    "BRK-B": "Berkshire Hathaway", "JPM": "JPMorgan Chase", "V": "Visa",
    "MA": "Mastercard", "BAC": "Bank of America", "WFC": "Wells Fargo",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley", "C": "Citigroup", "AXP": "American Express",
    # Health care
    "JNJ": "Johnson & Johnson", "UNH": "UnitedHealth", "LLY": "Eli Lilly",
    "MRK": "Merck", "ABBV": "AbbVie", "PFE": "Pfizer", "TMO": "Thermo Fisher",
    "ABT": "Abbott", "AMGN": "Amgen", "GILD": "Gilead", "BMY": "Bristol-Myers Squibb",
    "DHR": "Danaher",
    # Consumer
    "PG": "Procter & Gamble", "KO": "Coca-Cola", "PEP": "PepsiCo", "WMT": "Walmart",
    "COST": "Costco", "HD": "Home Depot", "LOW": "Lowe's", "MCD": "McDonald's",
    "NKE": "Nike", "SBUX": "Starbucks", "TGT": "Target", "DIS": "Disney",
    # Industrials / energy / materials / utilities / telecom
    "CAT": "Caterpillar", "DE": "Deere", "HON": "Honeywell", "GE": "GE Aerospace",
    "MMM": "3M", "BA": "Boeing", "LMT": "Lockheed Martin", "RTX": "RTX",
    "UPS": "UPS", "UNP": "Union Pacific", "XOM": "Exxon Mobil", "CVX": "Chevron",
    "COP": "ConocoPhillips", "LIN": "Linde", "NEE": "NextEra Energy",
    "DUK": "Duke Energy", "SO": "Southern Co", "T": "AT&T", "VZ": "Verizon",
}

SE_STOCKS: dict[str, str] = {
    "ABB.ST": "ABB", "ALFA.ST": "Alfa Laval", "ASSA-B.ST": "Assa Abloy",
    "ATCO-A.ST": "Atlas Copco", "AZN.ST": "AstraZeneca", "BOL.ST": "Boliden",
    "ELUX-B.ST": "Electrolux", "ERIC-B.ST": "Ericsson", "ESSITY-B.ST": "Essity",
    "EVO.ST": "Evolution", "GETI-B.ST": "Getinge", "HEXA-B.ST": "Hexagon",
    "HM-B.ST": "H&M", "INVE-B.ST": "Investor", "KINV-B.ST": "Kinnevik",
    "NDA-SE.ST": "Nordea", "NIBE-B.ST": "NIBE", "SAAB-B.ST": "Saab",
    "SAND.ST": "Sandvik", "SCA-B.ST": "SCA", "SEB-A.ST": "SEB",
    "SHB-A.ST": "Handelsbanken", "SKA-B.ST": "Skanska", "SKF-B.ST": "SKF",
    "SWED-A.ST": "Swedbank", "TEL2-B.ST": "Tele2", "TELIA.ST": "Telia",
    "VOLV-B.ST": "Volvo", "EPI-A.ST": "Epiroc", "LIFCO-B.ST": "Lifco",
    "INDU-C.ST": "Industrivärden", "TREL-B.ST": "Trelleborg", "HUSQ-B.ST": "Husqvarna",
    "AXFO.ST": "Axfood", "CAST.ST": "Castellum", "LATO-B.ST": "Latour",
    "HOLM-B.ST": "Holmen", "ADDT-B.ST": "Addtech", "BEIJ-B.ST": "Beijer Ref",
    "SECU-B.ST": "Securitas", "SAGA-B.ST": "Sagax", "SINCH.ST": "Sinch",
    "SBB-B.ST": "SBB", "INDT.ST": "Indutrade", "SWEC-B.ST": "Sweco",
}

# Benchmarks an index-fund investor could realistically hold.
BENCHMARKS: dict[str, str] = {
    "SPY": "S&P 500 (SPY)",
    "URTH": "MSCI World (URTH)",
    "^OMX": "OMX Stockholm 30 (price index)",
}

FX_TICKER = "USDSEK=X"  # SEK per 1 USD


def ticker_currency(ticker: str) -> str:
    """Trading currency of a Yahoo ticker in this universe."""
    if ticker.endswith(".ST") or ticker in ("^OMX", "^OMXSPI"):
        return "SEK"
    return "USD"


def ticker_country(ticker: str) -> str:
    return "SE" if ticker_currency(ticker) == "SEK" else "US"


def get_universe(markets: str = "both") -> dict[str, str]:
    """Return {ticker: name} for 'us', 'se' or 'both'."""
    markets = markets.lower()
    if markets == "us":
        return dict(US_STOCKS)
    if markets == "se":
        return dict(SE_STOCKS)
    if markets == "both":
        return {**US_STOCKS, **SE_STOCKS}
    raise ValueError(f"markets must be 'us', 'se' or 'both', got {markets!r}")


def all_names() -> dict[str, str]:
    return {**US_STOCKS, **SE_STOCKS, **BENCHMARKS}
