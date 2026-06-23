import httpx
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

_rate_cache = {"rates": None, "fetched_at": None}
CACHE_DURATION = timedelta(hours=12)
BASE_CURRENCY = "EUR"


async def get_exchange_rates() -> dict:
    """
    Fetch exchange rates with EUR as base, cached for 12 hours.
    Uses frankfurter.app — free, no API key, ECB-published rates.
    """
    now = datetime.utcnow()
    if _rate_cache["rates"] and _rate_cache["fetched_at"] and \
       (now - _rate_cache["fetched_at"] < CACHE_DURATION):
        return _rate_cache["rates"]

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"https://api.frankfurter.dev/v1/latest?from={BASE_CURRENCY}")
            response.raise_for_status()
            data = response.json()
            rates = data.get("rates", {})
            rates[BASE_CURRENCY] = 1.0
            _rate_cache["rates"] = rates
            _rate_cache["fetched_at"] = now
            logger.info(f"Fetched exchange rates: {list(rates.keys())}")
            return rates
    except Exception as e:
        logger.error(f"Failed to fetch exchange rates: {e}")
        return _rate_cache["rates"] or {}


def convert_to_base(amount: float, currency: str, rates: dict):
    """Convert amount in `currency` to BASE_CURRENCY. Returns None if currency unknown."""
    if not currency or currency == BASE_CURRENCY:
        return amount
    rate = rates.get(currency)
    if not rate or rate == 0:
        return None
    return amount / rate