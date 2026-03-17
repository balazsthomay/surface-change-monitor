"""CDSE OAuth2 authentication module.

Provides :class:`TokenManager`, which fetches and caches short-lived Bearer
tokens from the Copernicus Data Space Ecosystem (CDSE) identity provider.
Tokens are refreshed automatically when they approach expiry, so callers
can simply call :meth:`TokenManager.get_token` on every request without
worrying about caching or expiry logic.
"""

import time

import requests

from surface_change_monitor.config import CDSE_TOKEN_URL

# How many seconds before the reported expiry we treat the token as stale.
# This gives a safety margin to avoid using a token that expires mid-request.
_EXPIRY_BUFFER_SECONDS = 60


class AuthenticationError(Exception):
    """Raised when CDSE token endpoint rejects the provided credentials."""


class TokenManager:
    """Manages a single CDSE OAuth2 Bearer token with transparent caching.

    The first call to :meth:`get_token` fetches a token from the CDSE
    identity provider.  Subsequent calls within the token's lifetime return
    the cached value without making any HTTP request.  Once the token has
    less than ``_EXPIRY_BUFFER_SECONDS`` remaining, the next call will
    transparently fetch a fresh one.

    Args:
        username: CDSE account username (usually an e-mail address).
        password: CDSE account password.
    """

    def __init__(self, username: str, password: str) -> None:
        self._username = username
        self._password = password
        self._token: str | None = None
        self._token_expiry: float = 0.0  # monotonic timestamp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_token(self) -> str:
        """Return a valid access token, refreshing it when it is about to expire.

        Returns:
            The raw Bearer token string.

        Raises:
            AuthenticationError: If the identity provider rejects the credentials.
        """
        if self._token is None or time.monotonic() >= self._token_expiry:
            self._token, self._token_expiry = self._fetch_token()
        return self._token

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_token(self) -> tuple[str, float]:
        """POST credentials to the CDSE token endpoint.

        Returns:
            A ``(token, expiry_timestamp)`` tuple where *expiry_timestamp* is a
            :func:`time.monotonic` value after which the token should no longer
            be used.

        Raises:
            AuthenticationError: If the server responds with a 401 status.
            requests.HTTPError: For other non-2xx responses.
        """
        payload = {
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": self._username,
            "password": self._password,
        }

        response = requests.post(CDSE_TOKEN_URL, data=payload, timeout=30)

        if response.status_code == 401:
            raise AuthenticationError(
                "CDSE authentication failed: invalid username or password. "
                "Check CDSE_USERNAME and CDSE_PASSWORD in your .env file."
            )

        response.raise_for_status()

        body = response.json()
        token: str = body["access_token"]
        expires_in: int = body.get("expires_in", 300)

        expiry = time.monotonic() + expires_in - _EXPIRY_BUFFER_SECONDS
        return token, expiry
