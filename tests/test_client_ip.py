from starlette.requests import Request

from app.core.client_ip import client_ip


def make_request(peer: str | None = "10.0.0.5", forwarded: list[str] = ()) -> Request:
    headers = [(b"x-forwarded-for", value.encode()) for value in forwarded]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
        "client": (peer, 1234) if peer else None,
    }
    return Request(scope)


def test_no_proxy_uses_the_connection_address():
    assert client_ip(make_request("203.0.113.7"), trusted_proxies=0) == "203.0.113.7"


def test_no_proxy_ignores_forwarded_header():
    request = make_request("203.0.113.7", ["1.2.3.4"])
    assert client_ip(request, trusted_proxies=0) == "203.0.113.7"


def test_one_proxy_takes_the_rightmost_entry():
    # The client claimed 6.6.6.6; our proxy appended the real address.
    request = make_request("10.0.0.2", ["6.6.6.6, 203.0.113.7"])
    assert client_ip(request, trusted_proxies=1) == "203.0.113.7"


def test_two_proxies_take_the_second_from_the_right():
    request = make_request("10.0.0.3", ["6.6.6.6, 203.0.113.7, 10.0.0.2"])
    assert client_ip(request, trusted_proxies=2) == "203.0.113.7"


def test_multiple_headers_are_one_list():
    request = make_request("10.0.0.2", ["6.6.6.6", "203.0.113.7"])
    assert client_ip(request, trusted_proxies=1) == "203.0.113.7"


def test_too_few_entries_fall_back_to_the_connection():
    request = make_request("10.0.0.3", ["203.0.113.7"])
    assert client_ip(request, trusted_proxies=2) == "10.0.0.3"


def test_garbage_entry_falls_back_to_the_connection():
    request = make_request("10.0.0.2", ["not-an-ip"])
    assert client_ip(request, trusted_proxies=1) == "10.0.0.2"


def test_ipv4_mapped_ipv6_is_the_same_address():
    request = make_request("::ffff:203.0.113.7")
    assert client_ip(request, trusted_proxies=0) == "203.0.113.7"


def test_no_client_at_all():
    assert client_ip(make_request(None), trusted_proxies=0) == "unknown"
