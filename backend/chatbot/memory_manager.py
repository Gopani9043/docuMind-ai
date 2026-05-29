import json
import redis
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

_redis = redis.Redis(
    host=os.getenv("REDIS_HOST", "127.0.0.1"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

MAX_TURNS = 10


def _key(session_id: str) -> str:
    return f"docparse:chat:{session_id}"


def save_turn(session_id: str, role: str, content: str, metadata: dict = None):
    key = _key(session_id)
    turn = {
        "role": role,
        "content": content[:1000],
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": metadata or {}
    }
    _redis.rpush(key, json.dumps(turn))
    _redis.ltrim(key, -(MAX_TURNS * 2), -1)
    _redis.expire(key, 7200)


def get_history(session_id: str) -> list:
    raw = _redis.lrange(_key(session_id), 0, -1)
    return [json.loads(t) for t in raw]


def get_recent_context(session_id: str, turns: int = 6) -> str:
    history = get_history(session_id)
    if not history:
        return ""
    lines = []
    for turn in history[-turns:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content'][:400]}")
    return "\n".join(lines)


def get_last_metadata(session_id: str) -> dict:
    history = get_history(session_id)
    for turn in reversed(history):
        if turn["role"] == "assistant" and turn.get("metadata"):
            return turn["metadata"]
    return {}


def clear_session(session_id: str):
    _redis.delete(_key(session_id))
    _redis.delete(_entity_key(session_id))
    _redis.delete(_focus_key(session_id))
    _redis.delete(_results_key(session_id))
    _redis.delete(_state_key(session_id))


def get_session_stats(session_id: str) -> dict:
    history = get_history(session_id)
    return {
        "total_turns": len(history),
        "user_turns": sum(1 for t in history if t["role"] == "user"),
        "session_id": session_id
    }


# ── Entity tracking ───────────────────────────────────────────────────────────

ENTITY_KEYS = ["vendor", "amount", "doc_type", "filename", "currency", "date_range"]


def _entity_key(session_id: str) -> str:
    return f"docparse:entities:{session_id}"


def set_entity(session_id: str, entity_type: str, value: str):
    if entity_type not in ENTITY_KEYS:
        return
    _redis.hset(_entity_key(session_id), entity_type, value)
    _redis.expire(_entity_key(session_id), 7200)


def get_entity(session_id: str, entity_type: str) -> str | None:
    return _redis.hget(_entity_key(session_id), entity_type)


def get_all_entities(session_id: str) -> dict:
    return _redis.hgetall(_entity_key(session_id)) or {}


# ── Conversational focus tracking ─────────────────────────────────────────────

def _focus_key(session_id: str) -> str:
    return f"docparse:focus:{session_id}"


def set_conversation_focus(session_id: str, focus: dict):
    _redis.set(_focus_key(session_id), json.dumps(focus), ex=7200)


def get_conversation_focus(session_id: str) -> dict:
    raw = _redis.get(_focus_key(session_id))
    return json.loads(raw) if raw else {}


# ── Last results persistence ──────────────────────────────────────────────────

def _results_key(session_id: str) -> str:
    return f"docparse:results:{session_id}"


def set_last_results(session_id: str, results: list):
    _redis.set(
        _results_key(session_id),
        json.dumps(results[:20], default=str),
        ex=7200
    )


def get_last_results(session_id: str) -> list:
    raw = _redis.get(_results_key(session_id))
    return json.loads(raw) if raw else []


# ── NEW: Active resultset state ───────────────────────────────────────────────

def _state_key(session_id: str) -> str:
    return f"docparse:state:{session_id}"


def set_active_state(session_id: str, state: dict):
    """
    Store full conversational dataset state.
    This is the core of the resultset memory engine.
    Structure:
    {
        "intent": "list|aggregation|duplicate_detection|...",
        "active_dataset": "invoices|contracts|receipts|reports",
        "filters": {"currency": "EUR", "amount_gt": 10000, "vendor": "..."},
        "sort": {"field": "amount", "direction": "desc"},
        "aggregations": {"metric": "sum", "field": "amount"},
        "entities": {"vendor": "...", "currency": "...", "amount": "..."},
        "pagination": {"limit": 200, "offset": 0},
        "result_ids": [],
        "result_count": 0,
        "shown_count": 0,
        "query_scope": "all|filtered|vendor_specific",
        "last_sql": ""
    }
    """
    _redis.set(_state_key(session_id), json.dumps(state, default=str), ex=7200)


def get_active_state(session_id: str) -> dict:
    """Get current active resultset state."""
    raw = _redis.get(_state_key(session_id))
    if raw:
        return json.loads(raw)
    return {
        "intent": None,
        "active_dataset": None,
        "filters": {},
        "sort": {},
        "aggregations": {},
        "entities": {},
        "pagination": {"limit": 200, "offset": 0},
        "result_ids": [],
        "result_count": 0,
        "shown_count": 0,
        "query_scope": "all",
        "last_sql": ""
    }


def update_active_state(session_id: str, updates: dict):
    """
    Merge updates into existing state.
    Use this to apply incremental changes like adding a filter.
    """
    state = get_active_state(session_id)
    # Deep merge filters, entities, sort
    for key, val in updates.items():
        if key in ("filters", "entities", "sort", "aggregations", "pagination"):
            if isinstance(val, dict) and isinstance(state.get(key), dict):
                state[key].update(val)
            else:
                state[key] = val
        else:
            state[key] = val
    set_active_state(session_id, state)