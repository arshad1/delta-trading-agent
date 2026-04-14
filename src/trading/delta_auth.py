"""Delta Exchange HMAC-SHA256 request signing helpers.

Delta Exchange REST API v2 requires every authenticated request to carry
three headers built from the API key pair:

    api-key   : the api_key string
    timestamp : Unix epoch **seconds** (str)
    signature : HMAC-SHA256 hex digest of:
                    METHOD + timestamp + /path + query_string + body

The signature window is 5 seconds - stale signatures are rejected.
"""

import hashlib
import hmac
import time


def _generate_signature(api_secret: str, message: str) -> str:
    """Return the hex-encoded HMAC-SHA256 signature for *message*."""
    return hmac.new(
        api_secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def build_auth_headers(
    api_key: str,
    api_secret: str,
    method: str,
    path: str,
    query_string: str = "",
    body: str = "",
) -> dict:
    """Build the three authentication headers required by every private Delta call.

    Parameters
    ----------
    api_key:
        Delta Exchange API key string.
    api_secret:
        Delta Exchange API secret string.
    method:
        HTTP verb in uppercase (``"GET"``, ``"POST"``, ``"DELETE"`` …).
    path:
        URL path including leading slash, e.g. ``"/v2/orders"``.
    query_string:
        URL-encoded query string **with** leading ``?`` if non-empty,
        e.g. ``"?product_id=27&state=open"``.  Pass ``""`` when unused.
    body:
        Raw JSON request body string.  Pass ``""`` for GET/DELETE.

    Returns
    -------
    dict
        Headers ready to merge into an ``aiohttp`` or ``requests`` call.
    """
    timestamp = str(int(time.time()))
    # Signature covers: METHOD + timestamp + path + query_string + body
    signature_data = method + timestamp + path + query_string + body
    signature = _generate_signature(api_secret, signature_data)
    return {
        "api-key": api_key,
        "timestamp": timestamp,
        "signature": signature,
        "User-Agent": "python-delta-trading-agent",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
