from __future__ import annotations

from typing import Any

import httpx

from . import models as m


class GameClient:
    """
    A typed client for the Recruitment Game Server API.

    Configure with base_url and a bearer token, then call the endpoints.
    """

    def __init__(self, base_url: str, token: str | None = None, *, timeout: float = 30.0):
        if not base_url:
            raise ValueError("base_url is required, e.g. 'https://example.com'")
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    @property
    def token(self) -> str | None:
        return self._token

    @token.setter
    def token(self, value: str | None) -> None:
        self._token = value

    # --- Internal helpers ---
    def _auth_headers(self) -> dict[str, str]:
        if not self._token:
            raise RuntimeError("Bearer token not set. Set GameClient.token first.")
        return {"Authorization": f"Bearer {self._token}"}

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        return self._client.get(path, headers=self._auth_headers(), params=params)

    def _post(self, path: str, *, json: dict[str, Any] | None = None) -> httpx.Response:
        return self._client.post(path, headers=self._auth_headers(), json=json)

    # --- Public API methods ---
    def get_stages(self) -> m.StagesResponse:
        r = self._get("/api/stages")
        r.raise_for_status()
        return m.StagesResponse.model_validate(r.json())

    def get_stage_capabilities(self, stage: int) -> m.StageCapabilitiesResponse:
        r = self._get(f"/api/stage/{stage}/capabilities")
        r.raise_for_status()
        return m.StageCapabilitiesResponse.model_validate(r.json())

    def get_stage_progress(self, stage: int) -> m.StageProgress:
        r = self._get(f"/api/stage/{stage}/progress")
        r.raise_for_status()
        return m.StageProgress.model_validate(r.json())

    def get_tasks(self, stage: int) -> m.TasksResponse:
        r = self._get(f"/api/stage/{stage}/tasks")
        r.raise_for_status()
        return m.TasksResponse.model_validate(r.json())

    def request_next_task(self, stage: int) -> m.RequestTaskResponse:
        r = self._post(f"/api/stage/{stage}/tasks/request")
        if r.status_code == 409:
            # likely {"ok": False, "error": "unfinished_task"}
            try:
                data = r.json()
            except Exception:
                data = {"ok": False, "error": "conflict"}
            return m.RequestTaskResponse.model_validate(data)
        r.raise_for_status()
        return m.RequestTaskResponse.model_validate(r.json())

    def get_task(self, stage: int, task_id: str) -> m.TaskDetail:
        r = self._get(f"/api/stage/{stage}/tasks/{task_id}")
        r.raise_for_status()
        data = r.json()
        # Some implementations wrap the task under a top-level "task" key
        # Normalize so TaskDetail always receives id/level at the top level
        if isinstance(data, dict) and "task" in data and isinstance(data["task"], dict):
            inner = {**data["task"]}
            # propagate ok flag if present
            if "ok" in data and "ok" not in inner:
                inner["ok"] = data["ok"]
            data = inner
        return m.TaskDetail.model_validate(data)

    def submit_plan(self, stage: int, task_id: str, plan: list[list[m.Action]]) -> m.SubmitResult:
        # Convert plan to serializable format
        norm_plan: list[list[dict[str, Any]]] = []
        for turn in plan:
            norm_turn: list[dict[str, Any]] = []
            for action in turn:
                # Actions should already be API Action instances
                if isinstance(action, m.Action):
                    norm_turn.append(action.model_dump())
                else:
                    raise TypeError("Actions must be API Action instances")
            norm_plan.append(norm_turn)
        payload = {"plan": norm_plan}

        r = self._post(f"/api/stage/{stage}/tasks/{task_id}/submit", json=payload)
        # The API returns 200 with an ok flag; raise for other errors
        r.raise_for_status()
        return m.SubmitResult.model_validate(r.json())

    # Schema helper
    def get_level_schema(self) -> dict[str, Any]:
        r = self._get("/api/schema/level")
        r.raise_for_status()
        return r.json()

    # --- Context management ---
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GameClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
