"""HTTP client for NREL/NLR REopt v3.

The REopt v3 API is asynchronous: a job is submitted via POST, then results
are fetched by polling a results endpoint until the status reaches a terminal
value. This module factors the protocol into pure URL/payload builders plus
an orchestrator ``ReoptClient`` with injectable HTTP transport and clock,
so tests can run entirely offline.

Per the NLR domain migration (May 2026), the base URL targets
``developer.nlr.gov``; the legacy ``developer.nrel.gov`` domain is being
shut down May 29, 2026.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


REOPT_BASE_URL = "https://developer.nlr.gov/api/reopt/stable"

# REopt v3 returns these statuses; "Optimizing..." is the in-flight token.
TERMINAL_STATUSES = frozenset({"optimal", "Infeasible", "error", "Error"})


class ReoptError(RuntimeError):
    """Raised when REopt returns a terminal error status or transport fails."""


def build_job_submit_request(
    payload: Mapping[str, Any], *, api_key: str
) -> tuple[str, dict[str, Any]]:
    """Return ``(submit_url, body)`` for POSTing a REopt job."""

    url = f"{REOPT_BASE_URL}/job/?api_key={api_key}"
    return url, dict(payload)


def build_results_poll_request(run_uuid: str, *, api_key: str) -> str:
    """Return the GET URL for polling REopt job results."""

    return f"{REOPT_BASE_URL}/job/{run_uuid}/results/?api_key={api_key}"


def is_terminal_status(status: str) -> bool:
    """``True`` if the REopt status no longer requires polling."""

    return status in TERMINAL_STATUSES


def load_nlr_api_key(
    *,
    env_var: str = "NLR_API_KEY",
    fallback_path: Path | None = None,
) -> str:
    """Resolve the NLR developer API key from the environment or a local file.

    The fallback file is intentionally gitignored; production callers should
    set ``NLR_API_KEY`` instead.
    """

    value = os.environ.get(env_var)
    if value:
        return value.strip()
    path = fallback_path
    if path is not None and path.exists():
        return path.read_text(encoding="utf-8").strip()
    raise ReoptError(
        f"NLR API key not found: set {env_var} env var or provide "
        f"a fallback file at {fallback_path}"
    )


class ReoptClient:
    """Callable wrapping REopt v3 submit + poll + parse, with optional cache.

    The transport is injected via ``http_post`` and ``http_get``: each must
    accept a URL and (for POST) a JSON-serialisable body, and return the
    response decoded into a dict.
    """

    def __init__(
        self,
        http_post: Callable[[str, dict[str, Any]], dict[str, Any]],
        http_get: Callable[[str], dict[str, Any]],
        *,
        api_key: str,
        cache_dir: Path | None = None,
        poll_interval_s: float = 5.0,
        max_poll_attempts: int = 240,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._http_post = http_post
        self._http_get = http_get
        self._api_key = api_key
        self._cache_dir = cache_dir
        self._poll_interval_s = poll_interval_s
        self._max_poll_attempts = max_poll_attempts
        self._sleep = sleep if sleep is not None else _default_sleep

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        cached = self._cache_load(payload)
        if cached is not None:
            return cached

        submit_url, body = build_job_submit_request(payload, api_key=self._api_key)
        submit_response = self._http_post(submit_url, body)
        run_uuid = submit_response.get("run_uuid")
        if not run_uuid:
            raise ReoptError(
                f"REopt submit did not return a run_uuid: {submit_response!r}"
            )

        poll_url = build_results_poll_request(run_uuid, api_key=self._api_key)
        for _ in range(self._max_poll_attempts):
            results = self._http_get(poll_url)
            status = str(results.get("status", ""))
            if is_terminal_status(status):
                if status in {"error", "Error"}:
                    raise ReoptError(
                        f"REopt job {run_uuid} returned error: "
                        f"{results.get('messages') or results}"
                    )
                self._cache_store(payload, results)
                return results
            self._sleep(self._poll_interval_s)
        raise ReoptError(
            f"REopt job {run_uuid} did not reach a terminal status after "
            f"{self._max_poll_attempts} polls"
        )

    def _cache_path(self, payload: Mapping[str, Any]) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = _payload_digest(payload)
        return self._cache_dir / f"{digest}.json"

    def _cache_load(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self._cache_path(payload)
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _cache_store(self, payload: Mapping[str, Any], results: dict[str, Any]) -> None:
        path = self._cache_path(payload)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_redact_reopt_secrets(results), sort_keys=True),
            encoding="utf-8",
        )


def _payload_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _redact_reopt_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: "<redacted>" if key == "api_key" else _redact_reopt_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_reopt_secrets(item) for item in value]
    return value


def _default_sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


DEFAULT_NLR_API_KEY_FILE = Path("docs/nlr_api.txt")


def default_reopt_client(
    *,
    api_key: str | None = None,
    cache_dir: Path | None = None,
    poll_interval_s: float = 5.0,
    max_poll_attempts: int = 240,
    fallback_key_path: Path = DEFAULT_NLR_API_KEY_FILE,
) -> ReoptClient:
    """Wire a ``ReoptClient`` against stdlib HTTP transport.

    Resolves the NLR API key in this order: explicit ``api_key`` argument,
    then ``NLR_API_KEY`` environment variable, then the gitignored
    ``docs/nlr_api.txt`` local fallback. Reads from disk only at construction
    time so the key never appears in serialised state.
    """

    resolved_key = api_key or load_nlr_api_key(fallback_path=fallback_key_path)
    return ReoptClient(
        _urllib_post,
        _urllib_get,
        api_key=resolved_key,
        cache_dir=cache_dir,
        poll_interval_s=poll_interval_s,
        max_poll_attempts=max_poll_attempts,
    )


def _urllib_post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    from urllib.error import HTTPError
    from urllib import request

    encoded = json.dumps(body).encode("utf-8")
    req = request.Request(
        url,
        data=encoded,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req) as resp:  # noqa: S310 -- trusted REopt API endpoint
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        raise ReoptError(_http_error_message(exc)) from exc


def _urllib_get(url: str) -> dict[str, Any]:
    from urllib.error import HTTPError
    from urllib import request

    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req) as resp:  # noqa: S310 -- trusted REopt API endpoint
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        raise ReoptError(_http_error_message(exc)) from exc


def _http_error_message(exc: Any) -> str:
    body = ""
    if exc.fp is not None:
        try:
            body = exc.fp.read().decode("utf-8")
        except Exception:  # pragma: no cover - best-effort diagnostics only
            body = ""
    if body:
        try:
            decoded = json.loads(body)
            if isinstance(decoded, dict):
                decoded = _redact_reopt_secrets(decoded)
                diagnostic_keys = (
                    "status",
                    "messages",
                    "errors",
                    "error",
                    "run_uuid",
                    "api_version",
                )
                compact = {key: decoded[key] for key in diagnostic_keys if key in decoded}
                decoded = compact if compact else decoded
            body = json.dumps(decoded, sort_keys=True)
        except json.JSONDecodeError:
            body = body[:1000]
    return f"REopt HTTP {exc.code} {exc.reason}: {body}".strip()
