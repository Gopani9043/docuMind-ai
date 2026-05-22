import statistics
import logging

logger = logging.getLogger(__name__)


def detect_outliers(results: list[dict], amount_field: str = "amount") -> list[dict]:
    """
    Z-score outlier detection on invoice amounts.
    Flags anything above z-score 2.0.
    """
    if len(results) < 3:
        return []

    amounts = []
    for r in results:
        try:
            val = r.get(amount_field)
            if val is not None:
                amounts.append((float(str(val).replace(",", "")), r))
        except (ValueError, TypeError):
            continue

    if len(amounts) < 3:
        return []

    values = [a[0] for a in amounts]
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)

    if stdev == 0:
        return []

    outliers = []
    for val, row in amounts:
        z_score = abs(val - mean) / stdev
        if z_score > 2.0:
            deviation_pct = round(((val - mean) / mean) * 100, 1)
            outliers.append({
                **row,
                "z_score": round(z_score, 2),
                "deviation_pct": deviation_pct,
                "mean_amount": round(mean, 2),
                "flag": "HIGH" if val > mean else "LOW"
            })

    logger.info(f"Detected {len(outliers)} outliers")
    return outliers


def detect_duplicates(results: list[dict]) -> list[dict]:
    """
    Find invoices with same vendor and same amount.
    """
    seen = {}
    duplicates = []

    for row in results:
        vendor = str(row.get("vendor", "")).strip().lower()
        amount = str(row.get("amount", "")).strip()
        key = f"{vendor}|{amount}"

        if key in seen:
            if key not in [d.get("_dup_key") for d in duplicates]:
                duplicates.append({**seen[key], "_dup_key": key, "duplicate": True})
            duplicates.append({**row, "_dup_key": key, "duplicate": True})
        else:
            seen[key] = row

    logger.info(f"Detected {len(duplicates)} duplicate entries")
    return duplicates