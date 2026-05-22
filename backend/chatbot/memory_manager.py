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


def get_session_stats(session_id: str) -> dict:
    history = get_history(session_id)
    return {
        "total_turns": len(history),
        "user_turns": sum(1 for t in history if t["role"] == "user"),
        "session_id": session_id
    }

ENTITY_KEYS = ["vendor", "amount", "doc_type", "filename", "currency", "date_range"]


def _entity_key(session_id: str) -> str:
    return f"docparse:entities:{session_id}"


def set_entity(session_id: str, entity_type: str, value: str):
    """Store a named entity for the session."""
    if entity_type not in ENTITY_KEYS:
        return
    _redis.hset(_entity_key(session_id), entity_type, value)
    _redis.expire(_entity_key(session_id), 7200)


def get_entity(session_id: str, entity_type: str) -> str | None:
    """Get a specific entity for the session."""
    return _redis.hget(_entity_key(session_id), entity_type)


def get_all_entities(session_id: str) -> dict:
    """Get all tracked entities for the session."""
    return _redis.hgetall(_entity_key(session_id)) or {}
