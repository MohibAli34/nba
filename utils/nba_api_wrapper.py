"""
NBA API wrapper with proper headers, retries, and timeout handling.
This module configures nba_api to work better in cloud environments.
"""
import time
from typing import Optional, Callable, Any
from functools import wraps

# Import requests with error handling
try:
    import requests
except ImportError:
    print("[WARN] requests library not found. Install with: pip install requests")
    requests = None

# NBA API headers that make requests look more browser-like
NBA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

# Configure requests session with NBA headers
_nba_session = None


def get_nba_session():
    """Get or create a requests session configured for NBA API calls."""
    global _nba_session
    if requests is None:
        raise ImportError("requests library is required but not installed")
    if _nba_session is None:
        _nba_session = requests.Session()
        _nba_session.headers.update(NBA_HEADERS)
    return _nba_session


def configure_nba_api():
    """
    Configure nba_api library to use our custom session with proper headers.
    This monkey-patches the nba_api's internal requests calls.
    """
    try:
        # Try to patch the HTTP layer
        import nba_api.library.http as nba_http
        
        # Store original request method
        if hasattr(nba_http, 'NBAHTTP'):
            original_request = nba_http.NBAHTTP.request
            
            def patched_request(self, url, headers=None, timeout=600, **kwargs):  # 10 minutes default timeout per request
                """Patched request method with proper headers and extended timeout."""
                # Merge our headers with any provided headers
                merged_headers = NBA_HEADERS.copy()
                if headers:
                    merged_headers.update(headers)
                
                # Use our session
                session = get_nba_session()
                
                # Make request with retries and extended timeout
                response = safe_request_with_retries(
                    lambda: session.get(url, headers=merged_headers, timeout=timeout, **kwargs),
                    url=url,
                    max_retries=5,  # Increased retries
                    timeout=timeout
                )
                if response is None:
                    raise requests.exceptions.Timeout(f"Request timed out after retries (timeout={timeout}s)")
                return response
            
            nba_http.NBAHTTP.request = patched_request
            print("[NBA_API] Configured nba_api with custom headers and retries")
            return True
    except Exception as e:
        # If we can't patch, that's okay - we'll use safe_nba_api_call wrapper instead
        print(f"[NBA_API] [INFO] Using wrapper approach (could not patch nba_api directly): {e}")
        return False


def safe_request_with_retries(
    request_func: Callable,
    url: str = "",
    max_retries: int = 5,  # Increased from 3 to 5
    base_delay: float = 2.0,  # Start with 2s instead of 1.5s
    timeout: float = 600.0  # 10 minutes timeout per API request for very slow API
) -> Optional[requests.Response]:
    """
    Execute a request function with retries and exponential backoff.
    
    Args:
        request_func: Function that returns a requests.Response
        url: URL being requested (for logging)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds between retries
        timeout: Request timeout in seconds
    
    Returns:
        Response object if successful, None if all retries failed
    """
    last_exception = None
    
    for attempt in range(1, max_retries + 1):
        try:
            response = request_func()
            response.raise_for_status()
            if attempt > 1:
                print(f"[NBA_API] [RETRY] Success on attempt {attempt} for {url}")
            return response
        except requests.exceptions.Timeout as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                print(
                    f"[NBA_API] [RETRY] Timeout on attempt {attempt}/{max_retries} for {url}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                print(f"[NBA_API] [ERROR] All {max_retries} attempts failed for {url}: {e}")
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                print(
                    f"[NBA_API] [RETRY] Request error on attempt {attempt}/{max_retries} for {url}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                print(f"[NBA_API] [ERROR] All {max_retries} attempts failed for {url}: {e}")
        except Exception as e:
            # For non-request exceptions, don't retry
            print(f"[NBA_API] [ERROR] Non-retryable error for {url}: {e}")
            return None
    
    return None


def safe_nba_api_call(
    api_call_class,
    *args,
    max_retries: int = 5,  # Increased from 3 to 5
    timeout: float = 180.0,  # 3 minutes timeout for slow API
    **kwargs
) -> Optional[Any]:
    """
    Safely execute an nba_api call with retries and error handling.
    
    Args:
        api_call_class: Class that makes an nba_api call (e.g., playergamelog.PlayerGameLog)
        *args: Positional arguments for the API call
        max_retries: Maximum number of retry attempts
        **kwargs: Keyword arguments for the API call
    
    Returns:
        Instance of the API call class if successful, None otherwise
    """
    last_exception = None
    
    for attempt in range(1, max_retries + 1):
        try:
            # Instantiate the API call class with extended timeout
            # Note: Some nba_api classes make the request in __init__, others in get_data_frames()
            # Try to pass timeout if the class supports it
            try:
                result = api_call_class(*args, timeout=timeout, **kwargs)
            except TypeError:
                # Class doesn't accept timeout parameter, use default
                result = api_call_class(*args, **kwargs)
            
            # Try to get data frames (this triggers the actual API call if not already done)
            # Some endpoints might fail here, so we catch exceptions
            try:
                _ = result.get_data_frames()  # Trigger the API call if not already done
            except Exception as inner_e:
                # If getting data frames fails, it might be a timeout
                error_str = str(inner_e).lower()
                if "timeout" in error_str or "read timed out" in error_str or "timed out" in error_str:
                    raise requests.exceptions.Timeout(str(inner_e))
                # Re-raise if it's not a timeout
                raise
            
            if attempt > 1:
                print(f"[NBA_API] [SUCCESS] Got response on attempt {attempt}")
            return result
        except requests.exceptions.Timeout as e:
            last_exception = e
            if attempt < max_retries:
                delay = 2.0 * (2 ** (attempt - 1))  # Start with 2s, increase exponentially
                print(
                    f"[NBA_API] [RETRY] Timeout on attempt {attempt}/{max_retries} (timeout={timeout}s). "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                print(f"[NBA_API] [ERROR] All {max_retries} attempts failed (timeout={timeout}s): {e}")
        except Exception as e:
            # Check if it's a timeout-related error
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["timeout", "read timed out", "connection", "pool", "timed out", "ssl", "certificate"]):
                last_exception = e
                if attempt < max_retries:
                    delay = 2.0 * (2 ** (attempt - 1))
                    print(
                        f"[NBA_API] [RETRY] Network/timeout error on attempt {attempt}/{max_retries}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    print(f"[NBA_API] [ERROR] All {max_retries} attempts failed: {e}")
            else:
                # Non-timeout errors - don't retry
                print(f"[NBA_API] [ERROR] Non-retryable error: {e}")
                return None
    
    return None


# Initialize on import
try:
    configure_nba_api()
except Exception as e:
    print(f"[NBA_API] [WARN] Could not initialize NBA API wrapper: {e}")

