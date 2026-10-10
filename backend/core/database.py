from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncEngine
)
from backend.core.config import settings

# 1. Initialize the Async Engine with Connection Pooling
# pool_size=5 keeps idle RAM low (~20MB); max_overflow=10 handles bursts
# pool_pre_ping=True tests connections before checkout to prevent stale sockets
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False
)

# 2. Async Session Factory
# expire_on_commit=False is MANDATORY in async to prevent greenlet spawn errors
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# 3. Dependency Generator for FastAPI and pipeline scripts
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yields a thread-safe, isolated async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()