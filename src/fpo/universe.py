"""Investable universe, sector metadata, and benchmark definitions (Yahoo Finance tickers).

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

SECTORS: dict[str, str] = {
    # US Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Technology",
    "META": "Technology", "AVGO": "Technology", "ORCL": "Technology", "CSCO": "Technology",
    "ADBE": "Technology", "CRM": "Technology", "INTC": "Technology", "AMD": "Technology",
    "QCOM": "Technology", "TXN": "Technology", "IBM": "Technology", "NFLX": "Technology",
    # US Consumer
    "AMZN": "Consumer", "TSLA": "Consumer", "PG": "Consumer", "KO": "Consumer",
    "PEP": "Consumer", "WMT": "Consumer", "COST": "Consumer", "HD": "Consumer",
    "LOW": "Consumer", "MCD": "Consumer", "NKE": "Consumer", "SBUX": "Consumer",
    "TGT": "Consumer", "DIS": "Consumer",
    # US Financials
    "BRK-B": "Financials", "JPM": "Financials", "V": "Financials", "MA": "Financials",
    "BAC": "Financials", "WFC": "Financials", "GS": "Financials", "MS": "Financials",
    "C": "Financials", "AXP": "Financials",
    # US Health Care
    "JNJ": "Health Care", "UNH": "Health Care", "LLY": "Health Care", "MRK": "Health Care",
    "ABBV": "Health Care", "PFE": "Health Care", "TMO": "Health Care", "ABT": "Health Care",
    "AMGN": "Health Care", "GILD": "Health Care", "BMY": "Health Care", "DHR": "Health Care",
    # US Industrials
    "CAT": "Industrials", "DE": "Industrials", "HON": "Industrials", "GE": "Industrials",
    "MMM": "Industrials", "BA": "Industrials", "LMT": "Industrials", "RTX": "Industrials",
    "UPS": "Industrials", "UNP": "Industrials",
    # US Energy / Materials / Utilities / Telecom
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "LIN": "Materials",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "T": "Telecom", "VZ": "Telecom",
    # Swedish stocks
    "ABB.ST": "Industrials", "ALFA.ST": "Industrials", "ASSA-B.ST": "Industrials",
    "ATCO-A.ST": "Industrials", "EPI-A.ST": "Industrials", "LIFCO-B.ST": "Industrials",
    "NIBE-B.ST": "Industrials", "SAAB-B.ST": "Industrials", "SAND.ST": "Industrials",
    "SKA-B.ST": "Industrials", "SKF-B.ST": "Industrials", "VOLV-B.ST": "Industrials",
    "TREL-B.ST": "Industrials", "HUSQ-B.ST": "Industrials", "ADDT-B.ST": "Industrials",
    "BEIJ-B.ST": "Industrials", "SECU-B.ST": "Industrials", "INDT.ST": "Industrials",
    "SWEC-B.ST": "Industrials",
    "ERIC-B.ST": "Technology", "HEXA-B.ST": "Technology", "SINCH.ST": "Technology",
    "AZN.ST": "Health Care", "GETI-B.ST": "Health Care",
    "BOL.ST": "Materials", "SCA-B.ST": "Materials", "HOLM-B.ST": "Materials",
    "ELUX-B.ST": "Consumer", "ESSITY-B.ST": "Consumer", "EVO.ST": "Consumer",
    "HM-B.ST": "Consumer", "AXFO.ST": "Consumer",
    "INVE-B.ST": "Financials", "KINV-B.ST": "Financials", "NDA-SE.ST": "Financials",
    "SEB-A.ST": "Financials", "SHB-A.ST": "Financials", "SWED-A.ST": "Financials",
    "INDU-C.ST": "Financials", "LATO-B.ST": "Financials",
    "TEL2-B.ST": "Telecom", "TELIA.ST": "Telecom",
    "CAST.ST": "Real Estate", "SAGA-B.ST": "Real Estate", "SBB-B.ST": "Real Estate",
}

# Benchmarks an index-fund investor could realistically hold.
BENCHMARKS: dict[str, str] = {
    "SPY": "S&P 500 (SPY)",
    "URTH": "MSCI World (URTH)",
    "XACT-OMXS30.ST": "OMXS30 Total Return (XACT)",
    "^OMX": "OMX Stockholm 30 (price index)",
}

# Investable low-cost core ETFs used as a fallback sleeve when model signal is weak.
CORE_ETFS: dict[str, str] = {
    "SPY": "S&P 500 (SPY)",
    "URTH": "MSCI World (URTH)",
    "XACT-OMXS30.ST": "XACT OMXS30 (total return SE)",
}

# PRIIPs-compliant UCITS ETFs & 0%-courtage Swedish mutual fund proxies verified live on Yahoo Finance.
# Each entry maps an investable market-traded ETF proxy to its Swedish Nordnet/Avanza mutual fund
# equivalent (0% courtage & 0% FX fee when traded as a SEK mutual fund).
ETF_UNIVERSE: dict[str, dict] = {
    "EUNL.DE": {
        "name": "iShares Core MSCI World UCITS",
        "fund_proxy": "Nordnet Global Index / LF Global Index",
        "asset_class": "Global Equity",
        "role": "offensive",
        "canary": True,
        "ter_bps": 20.0,
        "fund_zero_fee": True,
    },
    "SXR8.DE": {
        "name": "iShares Core S&P 500 UCITS",
        "fund_proxy": "Nordnet USA Index",
        "asset_class": "US Equity",
        "role": "offensive",
        "canary": False,
        "ter_bps": 7.0,
        "fund_zero_fee": True,
    },
    "SXRV.DE": {
        "name": "iShares Nasdaq 100 UCITS",
        "fund_proxy": "Swedbank Robur Technology / Nordnet Teknologi",
        "asset_class": "US Tech",
        "role": "offensive",
        "canary": False,
        "ter_bps": 33.0,
        "fund_zero_fee": True,
    },
    "XACT-SVERIGE.ST": {
        "name": "XACT Sverige (Broad Sweden)",
        "fund_proxy": "Nordnet Sverige Index / Spiltan Aktiefond Investmentbolag",
        "asset_class": "Sweden Equity",
        "role": "offensive",
        "canary": False,
        "ter_bps": 15.0,
        "fund_zero_fee": True,
    },
    "XACT-OMXS30.ST": {
        "name": "XACT OMXS30 (Sweden Large Cap)",
        "fund_proxy": "Nordnet Sverige Index",
        "asset_class": "Sweden Large Cap",
        "role": "offensive",
        "canary": False,
        "ter_bps": 10.0,
        "fund_zero_fee": True,
    },
    "XACT-NORDEN.ST": {
        "name": "XACT Norden (Nordic Equities)",
        "fund_proxy": "Nordnet Norden Index",
        "asset_class": "Nordic Equity",
        "role": "offensive",
        "canary": False,
        "ter_bps": 15.0,
        "fund_zero_fee": True,
    },
    "IS3N.DE": {
        "name": "iShares Core MSCI EM IMI UCITS",
        "fund_proxy": "Nordnet Tillväxtmarknad Index / Avanza Emerging Markets",
        "asset_class": "Emerging Markets",
        "role": "offensive",
        "canary": True,
        "ter_bps": 18.0,
        "fund_zero_fee": True,
    },
    "IUSN.DE": {
        "name": "iShares MSCI World Small Cap UCITS",
        "fund_proxy": "Handelsbanken Global Småbolag Index",
        "asset_class": "Global Small Cap",
        "role": "offensive",
        "canary": False,
        "ter_bps": 35.0,
        "fund_zero_fee": True,
    },
    "4GLD.DE": {
        "name": "Xetra-Gold ETC (Physical Gold)",
        "fund_proxy": "Xetra-Gold ETC (4GLD, exchange-traded)",
        "asset_class": "Gold",
        "role": "both",          # can participate in offensive inflation regimes AND defensive crashes
        "canary": False,
        "ter_bps": 0.0,
        "fund_zero_fee": False,
    },
    "EUNA.DE": {
        "name": "iShares Core Global Aggregate Bond UCITS",
        "fund_proxy": "AMF Räntefond Mix / Storebrand Obligation",
        "asset_class": "Global Bonds",
        "role": "defensive",
        "canary": True,
        "ter_bps": 10.0,
        "fund_zero_fee": True,
    },
    "IB01.L": {
        "name": "iShares $ Treasury Bond 0-1yr UCITS (Cash Proxy)",
        "fund_proxy": "AMF Räntefond Kort / Spiltan Räntefond Sverige",
        "asset_class": "Short Cash / T-Bills",
        "role": "defensive",
        "canary": False,
        "ter_bps": 7.0,
        "fund_zero_fee": True,
    },
}

FX_TICKER = "USDSEK=X"      # SEK per 1 USD
EUR_FX_TICKER = "EURSEK=X"  # SEK per 1 EUR


def ticker_currency(ticker: str) -> str:
    """Trading currency of a Yahoo ticker in this universe."""
    if ticker.endswith(".ST") or ticker in ("^OMX", "^OMXSPI"):
        return "SEK"
    if ticker.endswith(".DE") or ticker.endswith(".PA") or ticker.endswith(".AS"):
        return "EUR"
    return "USD"


def ticker_country(ticker: str) -> str:
    cur = ticker_currency(ticker)
    if cur == "SEK":
        return "SE"
    if cur == "EUR":
        return "EU"
    return "US"


def ticker_sector(ticker: str) -> str:
    """Standardized sector label for risk concentration limits and sector-neutral ranking."""
    if ticker in ETF_UNIVERSE:
        return ETF_UNIVERSE[ticker]["asset_class"]
    return SECTORS.get(ticker, "Other")


def get_universe(markets: str = "both") -> dict[str, str]:
    """Return {ticker: name} for 'us', 'se', 'both', or 'etf'."""
    markets = markets.lower()
    if markets == "us":
        return dict(US_STOCKS)
    if markets == "se":
        return dict(SE_STOCKS)
    if markets == "both":
        return {**US_STOCKS, **SE_STOCKS}
    if markets == "etf":
        return {t: meta["name"] for t, meta in ETF_UNIVERSE.items()}
    raise ValueError(f"markets must be 'us', 'se', 'both', or 'etf', got {markets!r}")


def get_etf_universe() -> dict[str, dict]:
    """Return metadata dict for all UCITS ETFs & Swedish mutual fund proxies."""
    return {k: dict(v) for k, v in ETF_UNIVERSE.items()}


def etf_fund_proxy(ticker: str) -> str:
    """Return the 0%-courtage Swedish mutual fund equivalent for an ETF ticker, if any."""
    if ticker in ETF_UNIVERSE:
        return str(ETF_UNIVERSE[ticker]["fund_proxy"])
    return ticker


def all_names() -> dict[str, str]:
    etf_names = {t: meta["name"] for t, meta in ETF_UNIVERSE.items()}
    return {**US_STOCKS, **SE_STOCKS, **BENCHMARKS, **CORE_ETFS, **etf_names}
