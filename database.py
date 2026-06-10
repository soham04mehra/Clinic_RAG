from logging import raiseExceptions
from supabase import create_client, Client
import config

_supabase_client = None

def get_supabase() -> Client:
    """
    Returns a singleton synchronous Supabase client.
    """
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    return _supabase_client

