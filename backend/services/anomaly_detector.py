import statistics
import logging

logger = logging.getLogger(__name__)


def detect_outliers(results: list, amount_field: str = "amount") -> list:
    """
    Z-score outlier detection on invoice amounts.
    Flags anything above z-score 2.0.
    Works for any dataset — amount field is configurable.
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


def detect_duplicates(results: list) -> list:
    """
    Find duplicate invoices by matching on filename or invoice_number.
    Universal — works for any dataset, any vendor.

    Previous version matched vendor+amount which failed when amount=0.
    This version matches on the actual invoice identifier.
    """
    seen = {}
    duplicates = []

    for row in results:
        # ── Universal key: use filename or invoice_number, not amount ──
        # Try multiple field names to be compatible with any SQL result shape
        identifier = (
            row.get("filename") or
            row.get("invoice_number") or
            row.get("invoice_id") or
            row.get("doc_id") or
            ""
        ).strip().lower()

        vendor = (
            row.get("vendor") or
            row.get("vendor_name") or
            ""
        ).strip().lower()

        # Key on identifier alone — vendor is secondary
        key = identifier if identifier else vendor

        if not key:
            continue

        if key in seen:
            # First time we see a duplicate — add the original too
            orig_key = f"_orig_{key}"
            if orig_key not in seen:
                duplicates.append({**seen[key], "duplicate": True})
                seen[orig_key] = True
            duplicates.append({**row, "duplicate": True})
        else:
            seen[key] = row

    logger.info(f"Detected {len(duplicates)} duplicate entries")
    return duplicates