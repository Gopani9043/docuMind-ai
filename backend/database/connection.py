import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv

load_dotenv()

# ── Database URL from .env ────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://docparse:docparse123@localhost:5432/docparse_db"
)

# ── Engine ────────────────────────────────────────
# The engine is the connection pool to PostgreSQL.
# echo=True prints every SQL query to the terminal (great for debugging)
engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    pool_size=5,
    max_overflow=10
)

# ── Session factory ───────────────────────────────
# Every API request gets its own session (like a transaction)
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# ── Base class for all models ─────────────────────
class Base(DeclarativeBase):
    pass

# ── Dependency for FastAPI endpoints ─────────────
async def get_db():
    """
    FastAPI dependency — injects a DB session into any endpoint.
    Automatically closes the session when the request is done.
    Usage in endpoint: db: AsyncSession = Depends(get_db)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# ── Create all tables ─────────────────────────────
async def init_db():
    """
    Called once when the app starts.
    Creates all tables if they don't exist yet.
    """
    async with engine.begin() as conn:
        # Import models so Base knows about them
        from database import models  # noqa
        await conn.run_sync(Base.metadata.create_all)