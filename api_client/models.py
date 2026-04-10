from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, RootModel

# Core shared types
Coord = tuple[int, int]


class APIError(BaseModel):
    error: str
    detail: str | None = None


class StageInfo(BaseModel):
    stage: int
    name: str


class StagesResponse(BaseModel):
    ok: bool = True
    stages: list[StageInfo] = Field(default_factory=list)


class ProgressItem(BaseModel):
    # For normal users, UI uses numeric labels, but backend returns type name and status
    type: str
    status: str
    series_index: int | None = None
    series_total: int | None = None
    is_current: bool | None = None


class StageProgress(BaseModel):
    ok: bool = True
    is_superuser: bool | None = False
    items: list[ProgressItem] = Field(default_factory=list)
    # Some helpers the page references
    order: list[str] | None = None
    current_index: int | None = None
    current_series_index: int | None = None
    current_series_total: int | None = None


class TaskSummary(BaseModel):
    id: str
    type: str | None = None
    created_at: str | None = None
    # Any other fields are allowed
    model_config = {"extra": "allow"}


class TasksResponse(BaseModel):
    ok: bool = True
    tasks: list[TaskSummary] = Field(default_factory=list)


class ResourceNode(BaseModel):
    resource: str
    x: int
    y: int


class LevelBoard(BaseModel):
    width: int
    height: int
    grid: list[list[str]]
    resource_nodes: list[ResourceNode] = Field(default_factory=list)


class Structure(BaseModel):
    """API structure model - maintains compatibility with the game API."""

    type: str
    x: int
    y: int
    storage: dict[str, int] | None = None


class LevelGoal(BaseModel):
    target_structure_type: str
    target_resources: dict[str, int]


class LevelDefinition(BaseModel):
    spec_number: int
    max_turns: int
    board: LevelBoard
    structures: list[Structure] = Field(default_factory=list)
    level_goal: LevelGoal


class TaskDetail(BaseModel):
    ok: bool = True
    id: str
    level: LevelDefinition
    # Any other fields are allowed
    model_config = {"extra": "allow"}


class RequestTaskResponse(BaseModel):
    ok: bool
    task_id: str | None = None
    error: str | None = None


# Generic action model for API compatibility
class Action(BaseModel):
    action: str
    args: dict[str, Any] = Field(default_factory=dict)


class Turn(RootModel[list[Action]]):
    pass


class PlanSubmission(BaseModel):
    plan: list[list[Action]]


class SubmitResult(BaseModel):
    ok: bool
    # The backend may return details like success, failure reason, and logs
    success: bool | None = None
    error: str | None = None
    message: str | None = None  # API sometimes uses 'message' instead of 'error'
    accepted: bool | None = None
    attempts: int | None = None
    turns_used: int | None = None
    details: dict[str, Any] | None = None
    turn_index: int | None = None
    action_index: int | None = None

    @property
    def error_message(self) -> str:
        """Get the error message from either 'error' or 'message' field."""
        return self.error or self.message or ""


# Schema/capabilities wrappers
class ActionField(BaseModel):
    type: str
    description: str
    required: bool
    default: Any | None = None


class ActionSchema(BaseModel):
    description: str
    fields: dict[str, ActionField]
    required_fields: list[str]
    model_class: str


class StructureDetail(BaseModel):
    build_cost: dict[str, int] | None = None
    interfaces: list[str] = Field(default_factory=list)
    rate: int | None = None
    resource_allow: list[str] | None = None


class StageCapabilities(BaseModel):
    resources: list[str] = Field(default_factory=list)
    structures: list[str] = Field(default_factory=list)
    structures_details: dict[str, StructureDetail] = Field(default_factory=dict)
    terrains: list[str] = Field(default_factory=list)
    actions: dict[str, ActionSchema] = Field(default_factory=dict)


class StageCapabilitiesResponse(BaseModel):
    ok: bool = True
    stage: int
    new: StageCapabilities
    available: StageCapabilities | dict[str, Any] | None = None

    def available_capabilities(self) -> StageCapabilities | None:
        """Return aggregated capabilities if present and well-formed."""
        if isinstance(self.available, StageCapabilities):
            return self.available
        if isinstance(self.available, dict):
            # Bail out when endpoint intentionally omits data (only_new=1)
            if len(self.available) == 1 and "message" in self.available:
                return None
            try:
                return StageCapabilities.model_validate(self.available)
            except Exception:
                return None
        return None


class LevelSchema(BaseModel):
    # Allow arbitrary schema content
    schema_data: dict[str, Any] = Field(alias="schema")
