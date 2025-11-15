"""
Firebase caching utilities for API responses.
Stores data with timestamps and only fetches from API if data is >24 hours old or missing.
"""
import os
import json
import time
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Firebase credentials file path from environment
FIREBASE_CREDENTIALS_FILE = os.environ.get(
    "FIREBASE_CREDENTIALS_FILE",
    "nba-props-app-57fec-firebase-adminsdk-fbsvc-047c134065.json"
)

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("[WARN] Firebase Admin SDK not installed. Install with: pip install firebase-admin")

# Initialize Firebase (singleton pattern)
_firebase_app = None
_db = None


def initialize_firebase():
    """Initialize Firebase Admin SDK with service account credentials.
    
    Supports both file path and JSON string from environment variable.
    On Streamlit Cloud, use FIREBASE_CREDENTIALS_JSON environment variable.
    """
    global _firebase_app, _db
    
    if not FIREBASE_AVAILABLE:
        return False
    
    if _firebase_app is not None:
        return True
    
    try:
        # First, try to get credentials from environment variable (Streamlit Cloud secret)
        firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS_JSON", None)
        
        if firebase_creds_json:
            # Parse JSON string from environment variable
            try:
                cred_dict = json.loads(firebase_creds_json)
                cred = credentials.Certificate(cred_dict)
                _firebase_app = firebase_admin.initialize_app(cred)
                _db = firestore.client()
                print("[INFO] Firebase initialized successfully from environment variable")
                return True
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse FIREBASE_CREDENTIALS_JSON: {e}")
                # Fall through to try file path
            except Exception as e:
                print(f"[WARN] Failed to initialize Firebase from JSON: {e}")
                # Fall through to try file path
        
        # Fallback: Try to load from file path
        cred_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            FIREBASE_CREDENTIALS_FILE
        )
        
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            _firebase_app = firebase_admin.initialize_app(cred)
            _db = firestore.client()
            print("[INFO] Firebase initialized successfully from file")
            return True
        else:
            print(f"[WARN] Firebase credentials file not found at: {cred_path}")
            print(f"[WARN] Firebase will not be available. Set FIREBASE_CREDENTIALS_JSON environment variable for Streamlit Cloud.")
            return False
            
    except Exception as e:
        print(f"[WARN] Failed to initialize Firebase: {e}")
        print(f"[WARN] App will continue without Firebase caching.")
        return False


def get_firestore_db():
    """Get Firestore database instance."""
    if _db is None:
        initialize_firebase()
    return _db


def get_cached_data(cache_key: str, max_age_hours: int = 24) -> Optional[Dict[str, Any]]:
    """
    Get cached data from Firebase if it exists and is not older than max_age_hours.
    
    Args:
        cache_key: Unique key for the cached data
        max_age_hours: Maximum age in hours (default 24)
    
    Returns:
        The cached data if found and fresh, None otherwise
    """
    db = get_firestore_db()
    if db is None:
        print(f"[CACHE] [ERROR] Firebase DB not available for {cache_key}")
        return None
    
    try:
        print(f"[CACHE] [SEARCH] Checking cache for: {cache_key}")
        start_time = time.time()
        doc_ref = db.collection('api_cache').document(cache_key)
        doc = doc_ref.get()
        fetch_time = time.time() - start_time
        
        if not doc.exists:
            print(f"[CACHE] [ERROR] Cache MISS - Document not found: {cache_key} (checked in {fetch_time:.3f}s)")
            return None
        
        data = doc.to_dict()
        timestamp = data.get('timestamp')
        
        if timestamp:
            # Check if timestamp is a Firestore timestamp
            if hasattr(timestamp, 'to_datetime'):
                # Firestore Timestamp object
                cached_time = timestamp.to_datetime()
            elif hasattr(timestamp, 'timestamp'):
                cached_time = datetime.fromtimestamp(timestamp.timestamp())
            elif isinstance(timestamp, (int, float)):
                cached_time = datetime.fromtimestamp(timestamp)
            else:
                # Try parsing as ISO string
                try:
                    cached_time = datetime.fromisoformat(str(timestamp))
                except:
                    # If all else fails, assume cache is invalid
                    print(f"[CACHE] [WARN] Could not parse timestamp for {cache_key}")
                    return None
            
            age = datetime.now() - cached_time
            max_age = timedelta(hours=max_age_hours)
            
            if age > max_age:
                print(f"[CACHE] [CLOCK] Cache EXPIRED for {cache_key} (age: {age}, max: {max_age})")
                return None
            
            cached_data = data.get('data')
            data_size = len(str(cached_data)) if cached_data else 0
            print(f"[CACHE] [OK] Cache HIT for {cache_key} (age: {age}, size: {data_size} chars, fetched in {fetch_time:.3f}s)")
            return cached_data
        
        print(f"[CACHE] [WARN] No timestamp found in cache for {cache_key}")
        return None
    except Exception as e:
        print(f"[CACHE] [ERROR] ERROR reading from Firebase cache for {cache_key}: {e}")
        import traceback
        traceback.print_exc()
        return None


def set_cached_data(cache_key: str, data: Any):
    """
    Store data in Firebase cache with current timestamp.
    
    Args:
        cache_key: Unique key for the cached data
        data: Data to cache (must be JSON serializable)
    """
    db = get_firestore_db()
    if db is None:
        print(f"[CACHE] [ERROR] Cannot save to cache - DB not available: {cache_key}")
        return False
    
    try:
        # Ensure data is JSON serializable
        try:
            json.dumps(data)
        except (TypeError, ValueError) as e:
            print(f"[CACHE] [WARN] Data not JSON serializable, converting: {e}")
            # If data is not directly JSON serializable, convert to dict/list
            if hasattr(data, 'to_dict'):
                data = data.to_dict()
            elif hasattr(data, '__dict__'):
                data = data.__dict__
            else:
                data = str(data)
        
        start_time = time.time()
        doc_ref = db.collection('api_cache').document(cache_key)
        # Use SERVER_TIMESTAMP for accurate server-side timestamp
        from firebase_admin import firestore
        doc_ref.set({
            'data': data,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'cached_at': datetime.now().isoformat()
        })
        save_time = time.time() - start_time
        data_size = len(str(data))
        print(f"[CACHE] [SAVE] Saved data to cache: {cache_key} (size: {data_size} chars, saved in {save_time:.3f}s)")
        return True
    except Exception as e:
        print(f"[CACHE] [ERROR] ERROR writing to Firebase cache for {cache_key}: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_or_fetch(
    cache_key: str,
    fetch_func,
    max_age_hours: int = 24,
    *args,
    **kwargs
) -> Any:
    """
    Get data from cache or fetch from API if cache is stale/missing.
    
    Args:
        cache_key: Unique key for caching
        fetch_func: Function to call if cache miss/stale
        max_age_hours: Maximum cache age in hours
        *args, **kwargs: Arguments to pass to fetch_func
    
    Returns:
        Cached or freshly fetched data
    """
    # Try to get from cache
    cached_data = get_cached_data(cache_key, max_age_hours)
    if cached_data is not None:
        return cached_data
    
    # Cache miss or stale - fetch from API
    print(f"[INFO] Cache miss/stale for {cache_key}, fetching from API...")
    try:
        fresh_data = fetch_func(*args, **kwargs)
        
        # Store in cache
        if fresh_data is not None:
            set_cached_data(cache_key, fresh_data)
        
        return fresh_data
    except Exception as e:
        print(f"[ERROR] Error fetching data for {cache_key}: {e}")
        return None


def debug_list_cache_keys(prefix: str = None):
    """
    Debug function to list all cache keys in Firebase.
    
    Args:
        prefix: Optional prefix to filter keys (e.g., 'game_data_')
    
    Returns:
        List of cache keys
    """
    db = get_firestore_db()
    if db is None:
        print("[DEBUG] Firebase DB not available")
        return []
    
    try:
        docs = db.collection('api_cache').stream()
        keys = []
        for doc in docs:
            key = doc.id
            if prefix is None or key.startswith(prefix):
                data = doc.to_dict()
                timestamp = data.get('timestamp')
                if timestamp and hasattr(timestamp, 'to_datetime'):
                    cached_time = timestamp.to_datetime()
                    age = datetime.now() - cached_time
                    keys.append({
                        'key': key,
                        'age': str(age),
                        'cached_at': cached_time.isoformat()
                    })
                else:
                    keys.append({'key': key, 'age': 'unknown', 'cached_at': 'unknown'})
        
        print(f"[DEBUG] Found {len(keys)} cache keys" + (f" with prefix '{prefix}'" if prefix else ""))
        for item in keys:
            print(f"  - {item['key']} (age: {item['age']})")
        
        return keys
    except Exception as e:
        print(f"[DEBUG] Error listing cache keys: {e}")
        return []
