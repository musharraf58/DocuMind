"""Which IP address a request comes from, for per-IP rate limits.

X-Forwarded-For is a header any client can set to anything. Behind a proxy,
each proxy APPENDS the address it received the request from:

    client sends    X-Forwarded-For: 6.6.6.6            (a lie, or nothing)
    our proxy adds                   6.6.6.6, 203.0.113.7   (the real client)
    we receive it from the proxy's address

Only the entries added by proxies we run are trustworthy, and those are the
rightmost ones. With TRUSTED_PROXY_COUNT = N, the client is the Nth entry
from the right; everything left of it was written by the client. With N = 0
(the default: no proxy in front), the header is ignored completely and the
connection's own address is used.
"""

import ipaddress

from starlette.requests import Request

from app.core.config import get_settings

UNKNOWN = "unknown"


def client_ip(request: Request, trusted_proxies: int | None = None) -> str:
    if trusted_proxies is None:
        trusted_proxies = get_settings().TRUSTED_PROXY_COUNT
    connection = request.client.host if request.client else UNKNOWN
    if trusted_proxies <= 0:
        return _normalize(connection) or UNKNOWN
    # Several X-Forwarded-For headers count as one comma-separated list.
    values = request.headers.getlist("x-forwarded-for")
    entries = [part.strip() for value in values for part in value.split(",") if part.strip()]
    if len(entries) < trusted_proxies:
        # Fewer entries than proxies: the request didn't come through all of
        # them (or a proxy is misconfigured). Don't guess from the header.
        return _normalize(connection) or UNKNOWN
    return _normalize(entries[-trusted_proxies]) or _normalize(connection) or UNKNOWN


def _normalize(value: str) -> str | None:
    """The canonical form of an IP address, or None if it isn't one. Canonical,
    so "::FFFF:1.2.3.4"-style variants of one address share one limit."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return str(address)
