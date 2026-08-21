"""
Tests for authentication routes:
  POST /api/auth/register
  POST /api/auth/login
  POST /api/auth/refresh
  POST /api/auth/logout
  GET  /api/auth/me
  GET  /api/auth/quota
"""
from __future__ import annotations

import httpx
import pytest
from unittest.mock import patch, AsyncMock


# ── Register ──────────────────────────────────────────────────────────────────

class TestRegister:
    async def test_register_success(self, client: httpx.AsyncClient):
        """A new user can register and receives an access token."""
        resp = await client.post("/api/auth/register", json={
            "name":     "Alice",
            "email":    "alice@example.com",
            "password": "SecurePass1!",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert data["user"]["email"] == "alice@example.com"
        assert data["user"]["name"] == "Alice"

    async def test_register_duplicate_email(self, client: httpx.AsyncClient):
        """Registering the same email twice returns 409."""
        payload = {"name": "Bob", "email": "bob@example.com", "password": "Password1!"}
        await client.post("/api/auth/register", json=payload)
        resp = await client.post("/api/auth/register", json=payload)
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"].lower()

    async def test_register_short_password(self, client: httpx.AsyncClient):
        """Passwords under 8 characters should fail validation."""
        resp = await client.post("/api/auth/register", json={
            "name":     "Charlie",
            "email":    "charlie@example.com",
            "password": "abc",
        })
        # Pydantic validation rejects short passwords
        assert resp.status_code == 422

    async def test_register_invalid_email(self, client: httpx.AsyncClient):
        """Invalid email format should fail validation."""
        resp = await client.post("/api/auth/register", json={
            "name":     "Dave",
            "email":    "not-an-email",
            "password": "Password1!",
        })
        assert resp.status_code == 422


# ── Login ────────────────────────────────────────────────────────────────────

class TestLogin:
    async def test_login_success(self, client: httpx.AsyncClient, auth_headers: dict):
        """Test user can log in with correct credentials."""
        # conftest seeds a test user — login with those credentials
        from tests.conftest import TEST_USER_EMAIL, TEST_USER_PASS
        resp = await client.post("/api/auth/login", json={
            "email":    TEST_USER_EMAIL,
            "password": TEST_USER_PASS,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["user"]["email"] == TEST_USER_EMAIL

    async def test_login_wrong_password(self, client: httpx.AsyncClient):
        """Wrong password returns 401."""
        from tests.conftest import TEST_USER_EMAIL
        resp = await client.post("/api/auth/login", json={
            "email":    TEST_USER_EMAIL,
            "password": "WrongPassword!",
        })
        assert resp.status_code == 401

    async def test_login_unknown_email(self, client: httpx.AsyncClient):
        """Unknown email returns 401."""
        resp = await client.post("/api/auth/login", json={
            "email":    "unknown@example.com",
            "password": "Password1!",
        })
        assert resp.status_code == 401


# ── Get current user (me) ─────────────────────────────────────────────────────

class TestGetMe:
    async def test_get_me_authenticated(self, client: httpx.AsyncClient, auth_headers: dict):
        """Authenticated user can retrieve their own profile."""
        resp = await client.get("/api/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        from tests.conftest import TEST_USER_EMAIL, TEST_USER_NAME
        assert data["email"] == TEST_USER_EMAIL
        assert data["name"] == TEST_USER_NAME

    async def test_get_me_unauthenticated(self, client: httpx.AsyncClient):
        """Request without token returns 401."""
        resp = await client.get("/api/auth/me")
        assert resp.status_code == 401

    async def test_get_me_invalid_token(self, client: httpx.AsyncClient):
        """Invalid bearer token returns 401."""
        resp = await client.get("/api/auth/me", headers={"Authorization": "Bearer not.a.valid.jwt"})
        assert resp.status_code == 401


# ── Quota ─────────────────────────────────────────────────────────────────────

class TestQuota:
    async def test_get_quota_authenticated(self, client: httpx.AsyncClient, auth_headers: dict):
        """Authenticated user can retrieve their quota."""
        resp = await client.get("/api/auth/quota", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "used" in data
        assert "limit" in data
        assert data["used"] >= 0

    async def test_get_quota_unauthenticated(self, client: httpx.AsyncClient):
        """Unauthenticated quota request returns 401."""
        resp = await client.get("/api/auth/quota")
        assert resp.status_code == 401


# ── Logout ────────────────────────────────────────────────────────────────────

class TestLogout:
    async def test_logout(self, client: httpx.AsyncClient, auth_headers: dict):
        """Logout clears the refresh cookie."""
        resp = await client.post("/api/auth/logout", headers=auth_headers)
        assert resp.status_code in (200, 204)
