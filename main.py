from __future__ import annotations

import argparse
import curses
import json
import os
import sys

import httpx
from api_client.client import GameClient
from api_client.models import SubmitResult, TaskDetail, TaskSummary
from dotenv import load_dotenv
from game_simulation.actions import supported_action_types
from game_simulation.game_state import GameState
from game_simulation.game_types import supported_resource_types
from game_simulation.structures import supported_structure_types
from strategies.base_strategy import BaseStrategy, StrategyFailed
from visualizers.board import print_state
from visualizers.inventory import format_inventory_multiline, format_win_condition
from visualizers.plan import print_plan

_CAPABILITY_CHECK_CACHE: dict[int, str | None] = {}


class CapabilityMismatchError(RuntimeError):
    """Raised when the solver lacks support for stage capabilities."""

    def __init__(self, stage: int, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recruitment Game Server helper CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    advance = subparsers.add_parser(
        "advance",
        help="Progress through stages using starter-style task selection",
    )
    advance.add_argument("strategy", help="Strategy module inside strategies/ (e.g. ramp_up)")
    advance.add_argument("--loop", action="store_true", help="Keep solving tasks until stages finish")
    advance.add_argument("--no-visualize", action="store_true", help="Skip board visualization before solving")
    advance.add_argument("--show-plan", action="store_true", help="Print generated plans while advancing")
    advance.add_argument(
        "--plan-step",
        action="store_true",
        help="Pause after each printed turn; press Esc to continue without pausing",
    )
    advance.add_argument(
        "--ignore-capabilities",
        "--ignore-capabilites",
        action="store_true",
        dest="ignore_capabilities",
        help="Skip capability compatibility checks (developer override)",
    )

    solve_once = subparsers.add_parser(
        "solve-task",
        help="Solve a single task by id",
    )
    solve_once.add_argument("strategy", help="Strategy module inside strategies/ (e.g. ramp_up)")
    solve_once.add_argument("--stage", type=int, required=True, help="Stage number containing the task")
    solve_once.add_argument("--task-id", required=True, help="Task identifier to solve")
    solve_once.add_argument("--show-board", action="store_true", help="Show board before solving")
    solve_once.add_argument("--show-plan", action="store_true", help="Print generated plan actions")
    solve_once.add_argument(
        "--plan-step",
        action="store_true",
        help="Pause after each printed turn; press Esc to continue without pausing",
    )
    solve_once.add_argument("--auto-submit", action="store_true", help="Submit without confirmation")
    solve_once.add_argument(
        "--ignore-capabilities",
        "--ignore-capabilites",
        action="store_true",
        dest="ignore_capabilities",
        help="Skip capability compatibility checks (developer override)",
    )

    subparsers.add_parser(
        "tasks",
        help="List tasks available across all stages",
    )

    interactive = subparsers.add_parser(
        "solve-interactive",
        help="Interactive task browser and solver",
    )
    interactive.add_argument("strategy", help="Strategy module inside strategies/ (e.g. ramp_up)")
    interactive.add_argument("--show-board", action="store_true", help="Show board before solving")
    interactive.add_argument("--auto-submit", action="store_true", help="Submit without confirmation")
    interactive.add_argument(
        "--no-plan-step",
        action="store_false",
        dest="plan_step",
        help="Disable per-turn pause prompts when showing generated plans",
    )
    interactive.add_argument(
        "--ignore-capabilities",
        "--ignore-capabilites",
        action="store_true",
        dest="ignore_capabilities",
        help="Skip capability compatibility checks (developer override)",
    )
    interactive.set_defaults(plan_step=True)

    return parser.parse_args()


def _get_env_var(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def load_environment() -> None:
    load_dotenv()
    if not _get_env_var("GAME_SERVER_TOKEN", "ARBITER_TOKEN"):
        print(
            "Set GAME_SERVER_TOKEN (or legacy ARBITER_TOKEN) in your environment or .env file",
            file=sys.stderr,
        )
        sys.exit(1)


def get_client() -> GameClient:
    base_url = _get_env_var("GAME_SERVER_BASE_URL", "ARBITER_BASE_URL")
    token = _get_env_var("GAME_SERVER_TOKEN", "ARBITER_TOKEN")

    if not base_url:
        print(
            "Set GAME_SERVER_BASE_URL (or legacy ARBITER_BASE_URL) in your environment or .env file",
            file=sys.stderr,
        )
        sys.exit(1)

    return GameClient(base_url=base_url, token=token)


def get_strategy_class(strategy_name: str) -> type[BaseStrategy]:
    import importlib
    import inspect

    explicit_class: str | None = None
    module_path = strategy_name if "." in strategy_name else f"strategies.{strategy_name}"

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        if "." in strategy_name:
            mod_name, _, attr_name = strategy_name.rpartition(".")
            if mod_name:
                try:
                    module = importlib.import_module(mod_name)
                except ImportError as inner_exc:
                    raise SystemExit(f"Strategy module '{strategy_name}' not found") from inner_exc
                explicit_class = attr_name
            else:
                raise SystemExit(f"Strategy module '{strategy_name}' not found") from exc
        else:
            raise SystemExit(f"Strategy module '{strategy_name}' not found") from exc

    if explicit_class:
        obj = getattr(module, explicit_class, None)
        if obj is None:
            raise SystemExit(f"Strategy class '{explicit_class}' not found in module '{module.__name__}'")
        if not inspect.isclass(obj) or not issubclass(obj, BaseStrategy):
            raise SystemExit(f"'{explicit_class}' is not a BaseStrategy subclass")
        return obj

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is BaseStrategy:
            continue
        if issubclass(obj, BaseStrategy) and obj.__module__ == module.__name__:
            return obj

    raise SystemExit(f"No BaseStrategy subclass found in '{module_path}'")


def prompt(text: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    response = input(f"{text} {suffix} ").strip().lower()
    if not response:
        return default
    return response.startswith("y")


def ensure_stage_capabilities_supported(client: GameClient, stage: int) -> None:
    """Check that the solver implements all capabilities exposed for this stage."""
    cached = _CAPABILITY_CHECK_CACHE.get(stage)
    if cached is not None:
        if cached:
            raise CapabilityMismatchError(stage, cached)
        return

    try:
        capabilities = client.get_stage_capabilities(stage)
    except httpx.HTTPStatusError as exc:
        print(
            f"Failed to inspect stage {stage} capabilities: HTTP {exc.response.status_code if exc.response else 'error'}",
            file=sys.stderr,
        )
        raise

    available = capabilities.available_capabilities()
    if available is None:
        _CAPABILITY_CHECK_CACHE[stage] = None
        return

    solver_resources = supported_resource_types()
    solver_structures = supported_structure_types()
    solver_actions = supported_action_types()

    missing_resources = sorted(set(available.resources) - solver_resources)
    missing_structures = sorted(set(available.structures) - solver_structures)
    missing_actions = sorted(set(available.actions.keys()) - solver_actions)

    missing: list[tuple[str, list[str]]] = []
    if missing_resources:
        missing.append(("resources", missing_resources))
    if missing_structures:
        missing.append(("structures", missing_structures))
    if missing_actions:
        missing.append(("actions", missing_actions))

    if missing:
        lines = [
            f"Stage {stage} exposes capabilities not implemented by this solver:",
            *(f"  {label}: {', '.join(items)}" for label, items in missing),
            "Update the game solver to support these capabilities before continuing.",
        ]
        message = "\n".join(lines)
        _CAPABILITY_CHECK_CACHE[stage] = message
        raise CapabilityMismatchError(stage, message)

    _CAPABILITY_CHECK_CACHE[stage] = None


def discover_accessible_stage(client: GameClient) -> int | None:
    stages_resp = client.get_stages()
    stage_numbers = sorted(stage.stage for stage in stages_resp.stages) or [1]

    for stage in stage_numbers:
        try:
            prog = client.get_stage_progress(stage)
            order = prog.order or []
            idx = prog.current_index or 0
            # if stage unfinished, make sure tasks endpoint is reachable
            if not order or idx < len(order):
                client.get_tasks(stage)
                return stage
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code in (403, 404):
                continue
            raise

    return None


def stage_status(client: GameClient, stage: int) -> tuple[str | None, bool]:
    prog = client.get_stage_progress(stage)
    order = prog.order or []
    idx = prog.current_index or 0
    finished = bool(order and idx >= len(order))
    required = order[idx] if (order and idx < len(order)) else None
    return required, finished


def choose_or_request_task(client: GameClient, stage: int, required_type: str | None) -> str | None:
    tasks_resp = client.get_tasks(stage)
    for task in reversed(tasks_resp.tasks):
        if required_type and getattr(task, "type", None) != required_type:
            continue
        if not getattr(task, "solved_at_all", False):
            return task.id

    request = client.request_next_task(stage)
    if request.ok and request.task_id:
        return request.task_id

    # fallback: API might have returned an error despite a pending task existing
    tasks_resp = client.get_tasks(stage)
    if tasks_resp.tasks:
        return tasks_resp.tasks[-1].id

    return None


def solve_with_strategy(
    client: GameClient,
    stage: int,
    task: TaskDetail,
    strategy_cls: type[BaseStrategy],
    *,
    initial_state: GameState,
    show_board: bool,
    show_plan: bool,
    plan_step: bool,
    confirm_submit: bool,
    submit_plan: bool = True,
    show_response: bool = False,
    print_failure_message: bool = True,
    print_summary: bool = True,
    ignore_capabilities: bool = False,
) -> tuple[SubmitResult | None, BaseStrategy | None, str | None]:
    if not ignore_capabilities:
        try:
            ensure_stage_capabilities_supported(client, stage)
        except CapabilityMismatchError as exc:
            print(str(exc), file=sys.stderr)
            return None, None, None

    if show_board:
        print_state(initial_state)

    try:
        strategy = strategy_cls(initial_state.copy())
    except StrategyFailed as exc:
        print(f"Strategy failed: {exc}")
        return None, None, None

    # Provide task context for strategies that can use it.
    try:
        strategy.task_detail = task
    except Exception:
        pass

    attach_method = getattr(strategy, "attach_task", None)
    if callable(attach_method):
        try:
            attach_method(task)
        except Exception:
            pass

    try:
        strategy.generate_plan()
    except StrategyFailed as exc:
        print(f"Strategy failed: {exc}")
        return None, None, None

    plan = strategy.plan

    if show_plan and confirm_submit:
        print("Plan:\n")
        print_plan(initial_state, plan, step_through=plan_step)

    goal = task.level.level_goal
    if not submit_plan:
        return None, strategy, None

    if confirm_submit and not prompt("Submit plan?", default=False):
        return None, strategy, None

    api_plan = [[action.to_api_action() for action in turn] for turn in plan]
    result = client.submit_plan(stage, task.id, api_plan)

    stop_at: tuple[int, int] | None = None
    if result is not None and not result.ok:
        turn_idx = result.turn_index if result.turn_index is not None else -1
        action_idx = result.action_index if result.action_index is not None else -1
        if turn_idx >= 0:
            if action_idx < 0 and turn_idx < len(plan):
                actions = plan[turn_idx]
                action_idx = max(0, len(actions) - 1) if actions else 0
            stop_at = (turn_idx, max(0, action_idx))

    if show_plan:
        if confirm_submit:
            if stop_at is not None:
                print("\nPlan up to failing action:\n")
                print_plan(initial_state, plan, stop_at=stop_at, step_through=plan_step)
        else:
            print("Plan:\n")
            print_plan(initial_state, plan, stop_at=stop_at, step_through=plan_step)

    response_dump = json.dumps(result.model_dump(), indent=2)
    if show_response:
        print(response_dump)
        print("-" * 80)

    if print_summary:
        goal_text = format_win_condition(goal.target_resources, padding=0)
        print(f"Goal: {goal_text}")
        base_inventory = format_inventory_multiline(strategy.game_state.base.storage)
        print("Base after simulation:")
        for line in base_inventory.splitlines():
            print(f"  {line}")

        if result.ok:
            outcome = "SUCCESS" if (result.success or result.accepted) else "SUBMITTED"
            print(f"Plan result: {outcome}")
            if result.error:
                print(f"API message: {result.error}")
        elif print_failure_message and result.error_message:
            print(f"Submission failed: {result.error_message}")

    return result, strategy, response_dump


def fetch_all_tasks(client: GameClient) -> list[tuple[int, TaskSummary]]:
    stages_resp = client.get_stages()
    stage_numbers = sorted(stage.stage for stage in stages_resp.stages) or [1]

    all_tasks: list[tuple[int, TaskSummary]] = []
    for stage in stage_numbers:
        try:
            tasks_resp = client.get_tasks(stage)
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code in (403, 404):
                continue
            raise

        for summary in tasks_resp.tasks:
            all_tasks.append((stage, summary))

    return all_tasks


def command_solve_task(args: argparse.Namespace) -> None:
    load_environment()
    strategy_cls = get_strategy_class(args.strategy)

    with get_client() as client:
        task = client.get_task(args.stage, args.task_id)
        print(f"Selected task: {task.id}")
        initial_state = GameState(task.level)
        result, _, response_dump = solve_with_strategy(
            client,
            args.stage,
            task,
            strategy_cls,
            initial_state=initial_state,
            show_board=args.show_board,
            show_plan=args.show_plan,
            plan_step=args.plan_step,
            confirm_submit=not args.auto_submit,
            show_response=args.show_plan,
            ignore_capabilities=args.ignore_capabilities,
        )
        if result is None:
            return
        if response_dump and not args.show_plan:
            print(response_dump)
            print("-" * 80)


def command_advance(args: argparse.Namespace) -> None:
    load_environment()
    strategy_cls = get_strategy_class(args.strategy)

    with get_client() as client:
        last_stage: int | None = None

        while True:
            stage = discover_accessible_stage(client)
            if stage is None:
                print("No accessible unfinished stages.")
                return

            if stage != last_stage:
                print(f"=== Stage {stage} ===")
                last_stage = stage

            required_type, finished = stage_status(client, stage)
            if finished:
                print(f"Stage {stage} completed.")
                if args.loop:
                    last_stage = None
                    continue
                return

            task_id = choose_or_request_task(client, stage, required_type)
            if not task_id:
                print("No available task for the current gate.")
                return

            task = client.get_task(stage, task_id)
            print(f"Solving task {task.id}")
            initial_state = GameState(task.level)

            result, strategy, response_dump = solve_with_strategy(
                client,
                stage,
                task,
                strategy_cls,
                initial_state=initial_state,
                show_board=not args.no_visualize,
                show_plan=args.show_plan,
                plan_step=args.plan_step,
                confirm_submit=False,
                show_response=args.show_plan,
                print_failure_message=False,
                print_summary=True,
                ignore_capabilities=args.ignore_capabilities,
            )

            if strategy is None:
                cached_msg = _CAPABILITY_CHECK_CACHE.get(stage)
                if cached_msg and not args.ignore_capabilities:
                    print("Halting due to missing solver capabilities.")
                else:
                    print("Halting due to strategy failure.")
                return

            if result is None or not result.ok:
                if result is not None and result.error_message:
                    print(f"Submission failed: {result.error_message}")
                elif response_dump:
                    print(response_dump)
                    print("-" * 80)
                print("Halting due to submission failure.")
                return

            if not args.loop:
                return


def command_list_tasks() -> None:
    load_environment()

    with get_client() as client:
        tasks = fetch_all_tasks(client)
        if not tasks:
            print("No tasks available.")
            return

        tasks_by_stage: dict[int, list[TaskSummary]] = {}
        for stage, summary in tasks:
            tasks_by_stage.setdefault(stage, []).append(summary)

        for stage in sorted(tasks_by_stage.keys()):
            print(f"Stage {stage}:")
            summaries = tasks_by_stage[stage]
            if not summaries:
                print("  (no tasks)")
                continue

            for summary in summaries:
                parts: list[str] = [summary.id]
                task_type = getattr(summary, "type", None)
                if task_type:
                    parts.append(f"type={task_type}")
                created = getattr(summary, "created_at", None)
                if created:
                    parts.append(f"created={created}")
                solved = getattr(summary, "solved_at_all", None)
                if solved is not None:
                    parts.append("solved" if solved else "unsolved")
                print("  " + " | ".join(parts))


def command_solve_interactive(args: argparse.Namespace) -> None:
    load_environment()
    strategy_cls = get_strategy_class(args.strategy)

    with get_client() as client:
        tasks = fetch_all_tasks(client)

        if not tasks:
            print("No tasks available.")
            return

        index = 0

        def refresh_tasks() -> list[tuple[int, TaskSummary]]:
            return fetch_all_tasks(client)

        def format_task_line(stage: int, summary: TaskSummary, width: int) -> str:
            status_flag = getattr(summary, "solved_at_all", None)
            status_text = "solved" if status_flag else "pending"
            task_type = getattr(summary, "type", None)
            label_parts = [f"Stage {stage}", summary.id]
            if task_type:
                label_parts.append(f"type={task_type}")
            label_parts.append(status_text)
            line = " | ".join(label_parts)
            return line[: max(0, width - 1)]

        def run(stdscr):
            nonlocal tasks, index
            curses.curs_set(0)
            curses.use_default_colors()

            while True:
                stdscr.erase()
                max_y, max_x = stdscr.getmaxyx()
                header = "Interactive tasks: ↑/↓ navigate, PgUp/PgDn jump, Enter solve, r refresh, q quit"
                stdscr.addnstr(0, 0, header, max_x - 1)

                if not tasks:
                    stdscr.addnstr(2, 0, "No tasks available.", max_x - 1)
                    stdscr.refresh()
                    key = stdscr.getch()
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    if key in (ord("r"), ord("R")):
                        tasks = refresh_tasks()
                    continue

                visible_rows = max_y - 2
                if visible_rows <= 0:
                    visible_rows = 1
                start = max(0, min(index - visible_rows // 2, max(0, len(tasks) - visible_rows)))

                for row_offset in range(visible_rows):
                    task_idx = start + row_offset
                    if task_idx >= len(tasks):
                        break
                    stage, summary = tasks[task_idx]
                    line = format_task_line(stage, summary, max_x)
                    attr = curses.A_REVERSE if task_idx == index else curses.A_NORMAL
                    stdscr.addnstr(1 + row_offset, 0, line, max_x - 1, attr)

                stdscr.refresh()
                key = stdscr.getch()

                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (curses.KEY_UP, ord("k")):
                    if index > 0:
                        index -= 1
                    continue
                if key in (curses.KEY_DOWN, ord("j")):
                    if index < len(tasks) - 1:
                        index += 1
                    continue
                if key == curses.KEY_PPAGE:
                    if index > 0:
                        index = max(0, index - visible_rows)
                    continue
                if key == curses.KEY_NPAGE:
                    if index < len(tasks) - 1:
                        index = min(len(tasks) - 1, index + visible_rows)
                    continue
                if key in (ord("r"), ord("R")):
                    tasks = refresh_tasks()
                    if tasks:
                        index = min(index, len(tasks) - 1)
                    else:
                        index = 0
                    continue
                if key == curses.KEY_RESIZE:
                    continue
                if key in (curses.KEY_ENTER, 10, 13):
                    if not tasks:
                        continue
                    stage, summary = tasks[index]

                    curses.def_prog_mode()
                    curses.endwin()
                    try:
                        task = client.get_task(stage, summary.id)
                        initial_state = GameState(task.level)
                        print(f"Selected Stage {stage} task {summary.id}")
                        _, _, _ = solve_with_strategy(
                            client,
                            stage,
                            task,
                            strategy_cls,
                            initial_state=initial_state,
                            show_board=args.show_board,
                            show_plan=True,
                            plan_step=args.plan_step,
                            confirm_submit=not args.auto_submit,
                            show_response=True,
                            ignore_capabilities=args.ignore_capabilities,
                        )
                    finally:
                        input("Press Enter to return to interactive list...")
                        tasks = refresh_tasks()
                        if tasks:
                            index = min(index, len(tasks) - 1)
                        else:
                            index = 0
                        curses.reset_prog_mode()
                        curses.curs_set(0)

        curses.wrapper(run)


def main() -> None:
    args = parse_args()

    if args.command == "solve-task":
        command_solve_task(args)
    elif args.command == "advance":
        command_advance(args)
    elif args.command == "tasks":
        command_list_tasks()
    elif args.command == "solve-interactive":
        command_solve_interactive(args)
    else:  # pragma: no cover - defensive
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
