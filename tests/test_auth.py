"""Tests for CDSE OAuth2 authentication module."""

import time

import pytest
import responses as responses_lib

from surface_change_monitor.auth import AuthenticationError, TokenManager
from surface_change_monitor.config import CDSE_TOKEN_URL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_TOKEN = "eyJhbGciOiJSUzI1NiJ9.fake.token"
_EXPIRES_IN = 600  # 10 minutes


def _token_response(token: str = _FAKE_TOKEN, expires_in: int = _EXPIRES_IN) -> dict:
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": expires_in,
    }


# ---------------------------------------------------------------------------
# 2.1.1  test_get_access_token_success
# ---------------------------------------------------------------------------


@responses_lib.activate
def test_get_access_token_success(mock_env_credentials):
    """A successful POST to the token endpoint returns the access token string."""
    responses_lib.add(
        responses_lib.POST,
        CDSE_TOKEN_URL,
        json=_token_response(),
        status=200,
    )

    manager = TokenManager(
        username=mock_env_credentials["username"],
        password=mock_env_credentials["password"],
    )
    token = manager.get_token()

    assert token == _FAKE_TOKEN


# ---------------------------------------------------------------------------
# 2.1.2  test_get_access_token_bad_credentials
# ---------------------------------------------------------------------------


@responses_lib.activate
def test_get_access_token_bad_credentials(mock_env_credentials):
    """A 401 response from the token endpoint raises AuthenticationError."""
    responses_lib.add(
        responses_lib.POST,
        CDSE_TOKEN_URL,
        json={"error": "invalid_grant", "error_description": "Invalid user credentials"},
        status=401,
    )

    manager = TokenManager(
        username=mock_env_credentials["username"],
        password="wrong_password",
    )

    with pytest.raises(AuthenticationError):
        manager.get_token()


# ---------------------------------------------------------------------------
# 2.1.3  test_token_manager_caches_token
# ---------------------------------------------------------------------------


@responses_lib.activate
def test_token_manager_caches_token(mock_env_credentials):
    """Two get_token() calls within the token lifetime produce only one HTTP request."""
    responses_lib.add(
        responses_lib.POST,
        CDSE_TOKEN_URL,
        json=_token_response(),
        status=200,
    )

    manager = TokenManager(
        username=mock_env_credentials["username"],
        password=mock_env_credentials["password"],
    )

    token1 = manager.get_token()
    token2 = manager.get_token()

    assert token1 == token2 == _FAKE_TOKEN
    # Only one HTTP call should have been made
    assert len(responses_lib.calls) == 1


# ---------------------------------------------------------------------------
# 2.1.4  test_token_manager_refreshes_expired_token
# ---------------------------------------------------------------------------


@responses_lib.activate
def test_token_manager_refreshes_expired_token(mock_env_credentials):
    """After manually expiring the cached token, the next get_token() fetches a new one."""
    second_token = "second.fake.token"

    responses_lib.add(
        responses_lib.POST,
        CDSE_TOKEN_URL,
        json=_token_response(token=_FAKE_TOKEN),
        status=200,
    )
    responses_lib.add(
        responses_lib.POST,
        CDSE_TOKEN_URL,
        json=_token_response(token=second_token),
        status=200,
    )

    manager = TokenManager(
        username=mock_env_credentials["username"],
        password=mock_env_credentials["password"],
    )

    # First call — caches token
    token1 = manager.get_token()
    assert token1 == _FAKE_TOKEN

    # Force expiry by setting the cached expiry to the past
    manager._token_expiry = time.monotonic() - 1.0

    # Second call — cache is stale, should fetch fresh token
    token2 = manager.get_token()
    assert token2 == second_token
    assert len(responses_lib.calls) == 2


# ---------------------------------------------------------------------------
# 2.1.5  test_sends_correct_payload
# ---------------------------------------------------------------------------


@responses_lib.activate
def test_sends_correct_payload(mock_env_credentials):
    """The token request body must include client_id=cdse-public and grant_type=password."""
    responses_lib.add(
        responses_lib.POST,
        CDSE_TOKEN_URL,
        json=_token_response(),
        status=200,
    )

    manager = TokenManager(
        username=mock_env_credentials["username"],
        password=mock_env_credentials["password"],
    )
    manager.get_token()

    assert len(responses_lib.calls) == 1
    request_body = responses_lib.calls[0].request.body

    # application/x-www-form-urlencoded body is a plain string like
    # "client_id=cdse-public&grant_type=password&username=...&password=..."
    # The '@' in the e-mail address is percent-encoded as '%40'.
    from urllib.parse import parse_qs, unquote_plus

    assert "client_id=cdse-public" in request_body
    assert "grant_type=password" in request_body

    parsed = parse_qs(request_body)
    assert parsed.get("username") == [mock_env_credentials["username"]]
    assert parsed.get("password") == [mock_env_credentials["password"]]
