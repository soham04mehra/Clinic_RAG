import threading
from logging import raiseExceptions
from supabase import create_client, Client
import config

_supabase_client = None
_client_lock = threading.Lock()

def get_supabase() -> Client:
    """
    Returns a thread-safe singleton synchronous Supabase client.
    """
    global _supabase_client
    if _supabase_client is None:
        with _client_lock:
            if _supabase_client is None:
                _supabase_client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    return _supabase_client



