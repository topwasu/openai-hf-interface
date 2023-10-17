# Ref: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/cache.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.sql.expression import func

from sqlalchemy import Column, Integer, String, create_engine, select, ARRAY, Float
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base


class BaseCache(ABC):
    """Base interface for cache."""

    @abstractmethod
    def lookup(self, prompt: str, llm_string: str):
        """Look up based on prompt and llm_string."""

    @abstractmethod
    def update(self, prompt: str, llm_string: str, return_val):
        """Update cache based on prompt and llm_string."""


class InMemoryCache(BaseCache):
    """Cache that stores things in memory."""

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache = {}

    def lookup(self, prompt: str, llm_string: str, temperature: float, max_tokens: int, stop):
        """Look up based on prompt and llm_string."""
        if not isinstance(stop, str): stop = "|".join(stop)
        return self._cache.get((prompt, llm_string, temperature, max_tokens, stop), None)

    def update(self, prompt: str, llm_string: str, return_val, temperature: float, max_tokens: int, stop) -> None:
        """Update cache based on prompt and llm_string."""
        if not isinstance(stop, str): stop = "|".join(stop)
        self._cache[(prompt, llm_string, temperature, max_tokens, stop)] = return_val


Base = declarative_base()
class FullLLMCache(Base):  # type: ignore
    """SQLite table for full LLM Cache (all generations)."""

    __tablename__ = "full_llm_cache"
    idx = Column(Integer, primary_key=True)
    prompt = Column(String, primary_key=True)
    llm = Column(String, primary_key=True)
    temperature = Column(Float, primary_key=True)
    max_tokens = Column(Integer, primary_key=True)
    stop = Column(String, primary_key=True)
    response = Column(String)
    

class SQLAlchemyCache(BaseCache):
    """Cache that uses SQAlchemy as a backend."""

    def __init__(self, engine: Engine, cache_schema: Any = FullLLMCache):
        """Initialize by creating all tables."""
        self.engine = engine
        self.cache_schema = cache_schema
        self.cache_schema.metadata.create_all(self.engine)

    def lookup(self, prompt: str, llm_string: str, temperature: float, max_tokens: int, stop):
        """Look up based on prompt and llm_string."""
        if not isinstance(stop, str): stop = "|".join(stop)
        stmt = (
            select(self.cache_schema.response,
                self.cache_schema.idx)
            .where(self.cache_schema.prompt == prompt)
            .where(self.cache_schema.llm == llm_string)
            .where(self.cache_schema.temperature == temperature)
            .where(self.cache_schema.max_tokens == max_tokens)
            .where(self.cache_schema.stop == stop)
            .order_by(self.cache_schema.idx)
        )
        with Session(self.engine) as session:
            generations = [row for row in session.execute(stmt)]
            print(generations)
            generations.sort(key=lambda x: x[1])
            generations = [row[0] for row in generations]
            
            if len(generations) > 0:
                return generations
        return None

    def n_entries(self, prompt: str, llm_string: str, temperature: float, max_tokens: int, stop, session=None):
        """Look up based on prompt and llm_string."""
        if not isinstance(stop, str): stop = "|".join(stop)
        stmt = (
            select(func.max(self.cache_schema.idx))
            .where(self.cache_schema.prompt == prompt)
            .where(self.cache_schema.llm == llm_string)
            .where(self.cache_schema.temperature == temperature)
            .where(self.cache_schema.max_tokens == max_tokens)
            .where(self.cache_schema.stop == stop)
        )
        if session is None:
            with Session(self.engine) as session:
                generations = list(session.execute(stmt))
                if len(generations) > 0:
                    assert len(generations) == 1
                    g = generations[0][0]
                    if g is None: return 0
                    return g+1
        else:
            generations = list(session.execute(stmt))
            if len(generations) > 0:
                assert len(generations) == 1
                g = generations[0][0]
                if g is None: return 0
                return g+1
                    
        return 0

    def update(self, prompt: str, llm_string: str, return_val, temperature: float, max_tokens: int, stop) -> None:
        if not isinstance(stop, str): stop = "|".join(stop)
        for i, generation in enumerate(return_val):
            item = self.cache_schema(
                prompt=prompt, llm=llm_string, response=generation, idx=i,
                temperature=temperature, max_tokens=max_tokens, stop=stop
            )
            with Session(self.engine) as session, session.begin():
                session.merge(item)
                session.commit()

    def extend(self, prompt: str, llm_string: str, return_val, temperature: float, max_tokens: int, stop, session=None) -> None:
        if not isinstance(stop, str): stop = "|".join(stop)
        n = self.n_entries(prompt, llm_string, temperature, max_tokens, stop, session=session)
        print(n, "+", len(return_val), "=>", n+len(return_val), "entries")

        new_items = []
        for i, generation in enumerate(return_val):
            item = self.cache_schema(
                prompt=prompt, llm=llm_string, response=generation, idx=i+n,
                temperature=temperature, max_tokens=max_tokens, stop=stop
            )
            new_items.append(item)

        if session is None:
            with Session(self.engine) as session, session.begin():
                session.add_all(new_items)
                session.commit()
        else:
            print("about to add new items")
            session.add_all(new_items)
            print("about to commit new items")
            session.commit()
            print("committed new items")


class SQLiteCache(SQLAlchemyCache):
    """Cache that uses SQLite as a backend."""

    def __init__(self, database_path: str = "completions.db"):
        """Initialize by creating the engine and all tables."""
        engine = create_engine(f"sqlite:///{database_path}")
        super().__init__(engine)
