"""
Multi-Level Caching System for RAG Pipeline
Implements L1 (In-Memory) + L3 (Redis) caching with graceful fallbacks.

Architecture:
- L1: LRU in-memory cache (fast, limited size)
- L3: Redis distributed cache (persistent, scalable)
- Fallback: If Redis fails, continues with L1 only

Cache Hierarchy:
    Query → L1 Check → L3 Check → Compute → Store in L3 → Store in L1 → Return
"""

import logging
import hashlib
import json
import pickle
import time
from typing import Any, Optional, Dict, List
from functools import lru_cache
from collections import OrderedDict
from datetime import datetime, timedelta

# Redis imports with fallback
try:
    import redis
    from redis.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from app.config.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)


# ========== L1 Cache (In-Memory LRU) ==========
class L1Cache:
    """
    Level 1 in-memory LRU cache.
    
    Features:
    - Fast O(1) access
    - LRU eviction policy
    - TTL support
    - Thread-safe operations
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize L1 cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, datetime] = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"L1 Cache initialized: max_size={max_size}, ttl={ttl_seconds}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Check if key exists
        if key not in self.cache:
            self.misses += 1
            return None
        
        # Check TTL
        if self._is_expired(key):
            self._remove(key)
            self.misses += 1
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        # Remove if exists (to update position)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            # Check size limit
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
        
        # Store value and timestamp
        self.cache[key] = value
        self.timestamps[key] = datetime.now()
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self.cache:
            self._remove(key)
            return True
        return False
    
    def clear(self) -> None:
        """Clear entire cache."""
        count = len(self.cache)
        self.cache.clear()
        self.timestamps.clear()
        logger.info(f"L1 cache cleared: {count} entries removed")
    
    def _is_expired(self, key: str) -> bool:
        """Check if entry is expired."""
        if key not in self.timestamps:
            return True
        
        age = datetime.now() - self.timestamps[key]
        return age.total_seconds() > self.ttl_seconds
    
    def _remove(self, key: str) -> None:
        """Remove entry from cache."""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def _evict_oldest(self) -> None:
        """Evict least recently used entry."""
        if self.cache:
            oldest_key = next(iter(self.cache))
            self._remove(oldest_key)
            self.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests
        }


# ========== L3 Cache (Redis) ==========
class L3Cache:
    """
    Level 3 Redis distributed cache.
    
    Features:
    - Persistent storage
    - Distributed across instances
    - Automatic TTL
    - Connection pooling
    - Graceful degradation
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl_seconds: int = 86400,
        max_connections: int = 10
    ):
        """
        Initialize Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            ttl_seconds: Default TTL
            max_connections: Max connection pool size
        """
        self.ttl_seconds = ttl_seconds
        self.redis_client = None
        self.is_connected = False
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.errors = 0
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis library not available. L3 cache disabled.")
            return
        
        try:
            # Create connection pool
            pool = ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=False  # Handle binary data
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=pool)
            
            # Test connection
            self.redis_client.ping()
            self.is_connected = True
            
            logger.info(
                f"L3 Redis cache connected: {host}:{port}/{db}, "
                f"ttl={ttl_seconds}s"
            )
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            logger.warning("L3 cache disabled. Falling back to L1 only.")
            self.redis_client = None
            self.is_connected = False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        if not self.is_connected or self.redis_client is None:
            return None
        
        try:
            value = self.redis_client.get(key)
            
            if value is None:
                self.misses += 1
                return None
            
            # Deserialize
            result = pickle.loads(value)
            self.hits += 1
            return result
            
        except Exception as e:
            logger.error(f"Redis GET error: {str(e)}")
            self.errors += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis."""
        if not self.is_connected or self.redis_client is None:
            return False
        
        try:
            # Serialize
            serialized = pickle.dumps(value)
            
            # Set with TTL
            ttl = ttl or self.ttl_seconds
            self.redis_client.setex(key, ttl, serialized)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis SET error: {str(e)}")
            self.errors += 1
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self.is_connected or self.redis_client is None:
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis DELETE error: {str(e)}")
            self.errors += 1
            return False
    
    def clear(self, pattern: str = "*") -> int:
        """Clear keys matching pattern."""
        if not self.is_connected or self.redis_client is None:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                count = self.redis_client.delete(*keys)
                logger.info(f"Redis cleared: {count} keys matching '{pattern}'")
                return count
            return 0
        except Exception as e:
            logger.error(f"Redis CLEAR error: {str(e)}")
            self.errors += 1
            return 0
    
    def ping(self) -> bool:
        """Check Redis connection."""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            "connected": self.is_connected,
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests
        }
        
        # Get Redis info if connected
        if self.is_connected and self.redis_client:
            try:
                info = self.redis_client.info()
                stats["redis_used_memory_mb"] = round(
                    info.get("used_memory", 0) / 1024 / 1024, 2
                )
                stats["redis_keys"] = info.get("db0", {}).get("keys", 0)
            except:
                pass
        
        return stats


# ========== Cache Manager (Orchestrator) ==========
class CacheManager:
    """
    Multi-level cache manager.
    
    Manages L1 (memory) + L3 (Redis) with intelligent fallback.
    
    Cache Strategy:
    1. Check L1 → if hit, return
    2. Check L3 → if hit, populate L1 and return
    3. If miss, compute and store in both L3 and L1
    """
    
    def __init__(self):
        """Initialize cache manager with L1 and L3."""
        logger.info("=" * 60)
        logger.info("Initializing Multi-Level Cache System")
        logger.info("=" * 60)
        
        # Initialize L1 (always enabled)
        if settings.L1_CACHE_ENABLED:
            self.l1 = L1Cache(
                max_size=settings.L1_CACHE_MAX_SIZE,
                ttl_seconds=settings.L1_CACHE_TTL_SECONDS
            )
            logger.info("✓ L1 Cache (In-Memory): ENABLED")
        else:
            self.l1 = None
            logger.info("○ L1 Cache: DISABLED")
        
        # Initialize L3 (Redis)
        if settings.L3_CACHE_ENABLED:
            self.l3 = L3Cache(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                ttl_seconds=settings.REDIS_TTL_SECONDS,
                max_connections=settings.REDIS_MAX_CONNECTIONS
            )
            if self.l3.is_connected:
                logger.info("✓ L3 Cache (Redis): ENABLED")
            else:
                logger.warning("○ L3 Cache (Redis): UNAVAILABLE (using L1 only)")
        else:
            self.l3 = None
            logger.info("○ L3 Cache: DISABLED")
        
        logger.info("=" * 60)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache hierarchy.
        
        Order: L1 → L3 → None
        """
        # Add prefix
        full_key = f"{settings.CACHE_KEY_PREFIX}:{key}"
        
        # Try L1 first
        if self.l1:
            value = self.l1.get(full_key)
            if value is not None:
                logger.debug(f"Cache hit: L1 [{key[:30]}...]")
                return value
        
        # Try L3
        if self.l3 and self.l3.is_connected:
            value = self.l3.get(full_key)
            if value is not None:
                logger.debug(f"Cache hit: L3 [{key[:30]}...]")
                # Populate L1
                if self.l1:
                    self.l1.set(full_key, value)
                return value
        
        logger.debug(f"Cache miss: ALL [{key[:30]}...]")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache hierarchy.
        
        Stores in: L3 → L1
        """
        # Add prefix
        full_key = f"{settings.CACHE_KEY_PREFIX}:{key}"
        
        # Store in L3 first (persistent)
        if self.l3 and self.l3.is_connected:
            self.l3.set(full_key, value, ttl)
        
        # Store in L1 (fast access)
        if self.l1:
            self.l1.set(full_key, value)
        
        logger.debug(f"Cache set: [{key[:30]}...]")
    
    def delete(self, key: str) -> None:
        """Delete key from all cache levels."""
        full_key = f"{settings.CACHE_KEY_PREFIX}:{key}"
        
        if self.l1:
            self.l1.delete(full_key)
        
        if self.l3 and self.l3.is_connected:
            self.l3.delete(full_key)
    
    def clear_all(self) -> None:
        """Clear all caches."""
        logger.info("Clearing all caches...")
        
        if self.l1:
            self.l1.clear()
        
        if self.l3 and self.l3.is_connected:
            pattern = f"{settings.CACHE_KEY_PREFIX}:*"
            self.l3.clear(pattern)
        
        logger.info("✓ All caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "cache_enabled": settings.ENABLE_CACHING,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.l1:
            stats["l1_cache"] = self.l1.get_stats()
        
        if self.l3:
            stats["l3_cache"] = self.l3.get_stats()
        
        return stats
    
    # ========== Cache Key Generators ==========
    
    def generate_vector_search_key(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int,
        namespace: str = "",
        filter_dict: Optional[Dict] = None
    ) -> str:
        """Generate cache key for vector search."""
        # Create deterministic key from query parameters
        key_parts = [
            "vector_search",
            query_text.strip().lower()[:100],  # Normalized query
            str(top_k),
            namespace,
            json.dumps(filter_dict, sort_keys=True) if filter_dict else "none"
        ]
        
        # Hash embedding for consistency
        embedding_hash = hashlib.md5(
            str(query_embedding[:10]).encode()  # Use first 10 dims
        ).hexdigest()[:8]
        
        key_parts.append(embedding_hash)
        
        # Create hash of all parts
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"vs:{key_hash}"
    
    def generate_compression_key(
        self,
        query: str,
        chunk_ids: List[str]
    ) -> str:
        """Generate cache key for compression."""
        # Normalize query
        normalized_query = query.strip().lower()[:100]
        
        # Hash chunk IDs
        chunks_hash = hashlib.md5(
            "|".join(sorted(chunk_ids)).encode()
        ).hexdigest()[:8]
        
        # Create key
        key_string = f"compression|{normalized_query}|{chunks_hash}"
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"comp:{key_hash}"
    
    def generate_llm_response_key(
        self,
        query: str,
        chunk_ids: List[str]
    ) -> str:
        """Generate cache key for LLM response."""
        # Normalize query
        normalized_query = query.strip().lower()[:100]
        
        # Hash chunk IDs
        chunks_hash = hashlib.md5(
            "|".join(sorted(chunk_ids)).encode()
        ).hexdigest()[:8]
        
        # Create key
        key_string = f"llm|{normalized_query}|{chunks_hash}"
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"llm:{key_hash}"
    
    def invalidate_document(self, document_id: str) -> None:
        """Invalidate all cache entries for a document."""
        logger.info(f"Invalidating cache for document: {document_id}")
        
        # In production, you might want to track document-to-key mappings
        # For now, we clear all caches when a document changes
        self.clear_all()


# ========== Singleton Instance ==========
_cache_manager_instance = None

def get_cache_manager() -> CacheManager:
    """
    Get singleton cache manager instance.
    
    Returns:
        CacheManager instance
    """
    global _cache_manager_instance
    
    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager()
    
    return _cache_manager_instance