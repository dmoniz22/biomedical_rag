"""
Database connection and session management
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from app.core.config import settings

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+asyncpg://"),
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.DEBUG,
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Metadata for database operations
metadata = MetaData()


async def get_database_session() -> AsyncSession:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


class DatabaseManager:
    """Database connection and management utilities"""
    
    def __init__(self):
        self.engine = engine
        self.session_factory = AsyncSessionLocal
    
    async def create_tables(self):
        """Create all database tables"""
        from app.models.database import Base
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self):
        """Drop all database tables"""
        from app.models.database import Base
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        async with self.session_factory() as session:
            return session
    
    async def health_check(self) -> bool:
        """Check database connection health"""
        try:
            async with self.engine.connect() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
            return False


# Global database manager instance
db_manager = DatabaseManager()