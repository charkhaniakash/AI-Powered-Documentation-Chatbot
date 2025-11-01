"""
View and inspect Redis cache contents.
"""

import redis
import pickle
import json
from datetime import datetime
from app.config.settings import settings

# Connect to Redis
r = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    password=settings.REDIS_PASSWORD,
    decode_responses=False  # Keep binary for pickle
)

def list_all_keys():
    """List all cache keys."""
    pattern = f"{settings.CACHE_KEY_PREFIX}:*"
    keys = r.keys(pattern)
    
    print(f"\n{'=' * 80}")
    print(f"REDIS CACHE KEYS ({len(keys)} total)")
    print(f"{'=' * 80}\n")
    
    for i, key in enumerate(keys, 1):
        key_str = key.decode('utf-8')
        ttl = r.ttl(key)
        size = r.memory_usage(key)
        
        print(f"{i}. {key_str}")
        print(f"   TTL: {ttl}s | Size: {size} bytes")
        print()
    
    return keys


def view_key_details(key):
    """View detailed information about a key."""
    if isinstance(key, str):
        key = key.encode('utf-8')
    
    key_str = key.decode('utf-8')
    
    print(f"\n{'=' * 80}")
    print(f"KEY DETAILS: {key_str}")
    print(f"{'=' * 80}\n")
    
    # Check if exists
    if not r.exists(key):
        print("❌ Key does not exist!")
        return
    
    # Get metadata
    print("METADATA:")
    print(f"  Type: {r.type(key).decode('utf-8')}")
    print(f"  TTL: {r.ttl(key)} seconds")
    print(f"  Memory: {r.memory_usage(key)} bytes")
    print()
    
    # Get value
    try:
        raw_value = r.get(key)
        
        # Try to unpickle
        try:
            value = pickle.loads(raw_value)
            
            print("VALUE (Unpickled):")
            print("-" * 40)
            
            # Pretty print based on type
            if isinstance(value, dict):
                print(json.dumps(value, indent=2, default=str))
            elif isinstance(value, list):
                print(f"List with {len(value)} items:")
                for i, item in enumerate(value[:5], 1):  # Show first 5
                    print(f"  [{i}] {str(item)[:100]}...")
                if len(value) > 5:
                    print(f"  ... and {len(value) - 5} more items")
            else:
                print(f"{type(value).__name__}: {str(value)[:500]}...")
            
        except Exception as e:
            print(f"RAW VALUE (Could not unpickle: {e}):")
            print("-" * 40)
            print(raw_value[:500])
            if len(raw_value) > 500:
                print(f"... ({len(raw_value) - 500} more bytes)")
    
    except Exception as e:
        print(f"❌ Error reading value: {e}")


def search_keys(pattern):
    """Search for keys matching pattern."""
    full_pattern = f"{settings.CACHE_KEY_PREFIX}:{pattern}"
    keys = r.keys(full_pattern)
    
    print(f"\n{'=' * 80}")
    print(f"SEARCH RESULTS: {pattern}")
    print(f"Found {len(keys)} keys")
    print(f"{'=' * 80}\n")
    
    for key in keys:
        print(f"  - {key.decode('utf-8')}")
    
    return keys


def get_cache_stats():
    """Get overall Redis statistics."""
    info = r.info()
    
    print(f"\n{'=' * 80}")
    print("REDIS STATISTICS")
    print(f"{'=' * 80}\n")
    
    print("MEMORY:")
    print(f"  Used: {info['used_memory_human']}")
    print(f"  Peak: {info['used_memory_peak_human']}")
    print(f"  RSS: {info['used_memory_rss_human']}")
    print()
    
    print("KEYS:")
    total_keys = sum(info.get(f'db{i}', {}).get('keys', 0) for i in range(16))
    print(f"  Total: {total_keys}")
    print()
    
    print("PERFORMANCE:")
    print(f"  Uptime: {info['uptime_in_seconds']} seconds")
    print(f"  Connected clients: {info['connected_clients']}")
    print(f"  Commands processed: {info['total_commands_processed']:,}")
    print()


def delete_key(key):
    """Delete a specific key."""
    if isinstance(key, str):
        key = key.encode('utf-8')
    
    result = r.delete(key)
    if result:
        print(f"✓ Deleted: {key.decode('utf-8')}")
    else:
        print(f"❌ Key not found: {key.decode('utf-8')}")


def clear_all_cache():
    """Clear all cache keys."""
    pattern = f"{settings.CACHE_KEY_PREFIX}:*"
    keys = r.keys(pattern)
    
    if keys:
        confirm = input(f"⚠️  Delete {len(keys)} keys? (yes/no): ")
        if confirm.lower() == 'yes':
            deleted = r.delete(*keys)
            print(f"✓ Deleted {deleted} keys")
        else:
            print("Cancelled")
    else:
        print("No keys to delete")


def interactive_menu():
    """Interactive menu to explore Redis cache."""
    while True:
        print(f"\n{'=' * 80}")
        print("REDIS CACHE VIEWER")
        print(f"{'=' * 80}")
        print("1. List all keys")
        print("2. View key details")
        print("3. Search keys")
        print("4. Get cache statistics")
        print("5. Delete a key")
        print("6. Clear all cache")
        print("7. Exit")
        print(f"{'=' * 80}")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            list_all_keys()
        
        elif choice == '2':
            keys = list_all_keys()
            if keys:
                try:
                    idx = int(input("\nEnter key number to view: ")) - 1
                    if 0 <= idx < len(keys):
                        view_key_details(keys[idx])
                    else:
                        print("Invalid number")
                except ValueError:
                    print("Invalid input")
        
        elif choice == '3':
            pattern = input("Enter search pattern (e.g., vs:* for vector search): ")
            search_keys(pattern)
        
        elif choice == '4':
            get_cache_stats()
        
        elif choice == '5':
            keys = list_all_keys()
            if keys:
                try:
                    idx = int(input("\nEnter key number to delete: ")) - 1
                    if 0 <= idx < len(keys):
                        delete_key(keys[idx])
                    else:
                        print("Invalid number")
                except ValueError:
                    print("Invalid input")
        
        elif choice == '6':
            clear_all_cache()
        
        elif choice == '7':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid option")


if __name__ == "__main__":
    try:
        # Test connection
        r.ping()
        print("✓ Connected to Redis")
        
        # Start interactive menu
        interactive_menu()
        
    except redis.ConnectionError:
        print("❌ Could not connect to Redis")
        print(f"   Host: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        print("   Make sure Redis is running!")