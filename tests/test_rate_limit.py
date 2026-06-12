import unittest

from starlette.requests import Request

from app.core.rate_limit import client_ip_key


def make_request(headers=None, client=("10.0.0.1", 80)):
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "scheme": "http",
        "server": ("testserver", 80),
        "client": client,
        "headers": [
            (key.lower().encode("latin-1"), value.encode("latin-1"))
            for key, value in (headers or {}).items()
        ],
    }
    return Request(scope)


class ClientIpKeyTests(unittest.TestCase):
    def test_without_proxy_header_uses_socket_address(self):
        request = make_request(client=("203.0.113.7", 1234))
        self.assertEqual(client_ip_key(request), "203.0.113.7")

    def test_render_proxy_header_yields_client_ip(self):
        # request.client wäre hier die Proxy-IP; der Header gewinnt.
        request = make_request(headers={"X-Forwarded-For": "198.51.100.23"})
        self.assertEqual(client_ip_key(request), "198.51.100.23")

    def test_spoofed_forwarded_entries_are_ignored(self):
        # Der Client schickt selbst ein X-Forwarded-For mit; Render hängt die
        # echte Verbindungs-IP hinten an – nur die zählt.
        request = make_request(
            headers={"X-Forwarded-For": "1.2.3.4, 5.6.7.8, 198.51.100.23"}
        )
        self.assertEqual(client_ip_key(request), "198.51.100.23")

    def test_empty_header_falls_back(self):
        request = make_request(headers={"X-Forwarded-For": "  "},
                               client=("203.0.113.7", 1234))
        self.assertEqual(client_ip_key(request), "203.0.113.7")


if __name__ == "__main__":
    unittest.main()
