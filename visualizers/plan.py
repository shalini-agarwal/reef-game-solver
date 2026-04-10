import sys
from typing import Any

from game_simulation.actions import BaseAction
from game_simulation.game_state import GameState
from tabulate import tabulate

from visualizers.board import parse_state
from visualizers.inventory import format_inventory_multiline

ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"


def _highlight(text: str, *, failed: bool) -> str:
    if not failed:
        return text
    return f"{ANSI_RED}{text}{ANSI_RESET}"


def _read_single_key(prompt: str) -> str:
    """Read a single keypress from stdin, returning a one-character string."""

    print(prompt, end="", flush=True)

    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_attributes = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attributes)
        print()
        return ch
    except (ImportError, OSError, AttributeError):
        pass

    try:
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ch = msvcrt.getwch()
        print()
        return ch
    except ImportError:
        try:
            response = input()
        except EOFError:
            return "\n"
        return response[:1] if response else "\n"


def _prompt_next_turn() -> bool:
    """Prompt user before continuing to the next turn; return whether to keep pausing."""

    key = _read_single_key("Press any key for next turn (Esc to finish)... ")
    if key == "\x1b":
        print("Continuing without pausing.")
        return False
    if key in {"\x03", "\x04"}:
        raise KeyboardInterrupt
    return True


def print_plan(
    game_state: GameState,
    plan: list[list[BaseAction]],
    stop_at: tuple[int, int] | None = None,
    *,
    step_through: bool = False,
) -> None:
    """Visualize plan execution showing actions per turn and game state after each turn."""
    game_state = game_state.copy()

    stop_turn: int | None = None
    stop_action: int | None = None
    if stop_at is not None:
        stop_turn, stop_action = stop_at

    # Show initial state
    print("=" * 60)
    print("INITIAL STATE")
    print("=" * 60)

    # Show base and connected resources
    base = game_state.base
    base_inventory = format_inventory_multiline(base.storage)
    non_base_resources = game_state.non_base_storage_resources()
    non_base_inventory = format_inventory_multiline(non_base_resources)

    print("Base Resources:")
    for line in base_inventory.splitlines():
        print(f"  {line}")
    print("Non-base Storage:")
    for line in non_base_inventory.splitlines():
        print(f"  {line}")

    # Show win condition
    from visualizers.inventory import format_win_condition

    goal_text = format_win_condition(game_state.goal.target_resources, padding=0)
    print(f"Win Condition:       {goal_text}")
    print()

    print(parse_state(game_state))
    print()

    if not plan:
        print("Plan is empty - no actions to execute")
        return

    # Execute each turn
    remaining_pause = step_through

    if remaining_pause:
        remaining_pause = _prompt_next_turn()

    for turn_idx, actions in enumerate(plan):
        if stop_turn is not None and turn_idx > stop_turn:
            break

        print("=" * 60)
        print(f"TURN {turn_idx}")
        print("=" * 60)
        game_state.turn_start()

        # Show actions for this turn
        if not actions:
            print("No actions in this turn")
        else:
            action_headers = ["Action idx", "Action", "Description"]
            action_rows = []

            limit = len(actions)
            if stop_turn is not None and turn_idx == stop_turn and stop_action is not None:
                limit = min(limit, stop_action + 1)

            for action_idx, action in enumerate(actions):
                if action_idx >= limit:
                    break
                action_desc = str(action)
                action_type_obj = getattr(action, "action_type", None)
                if callable(action_type_obj):
                    action_type = action_type_obj()
                elif isinstance(action_type_obj, str):
                    action_type = action_type_obj
                else:
                    action_type = action.__class__.__name__
                failed = stop_turn is not None and turn_idx == stop_turn and stop_action == action_idx
                action_rows.append(
                    [
                        _highlight(str(action_idx), failed=failed),
                        _highlight(action_type, failed=failed),
                        _highlight(action_desc, failed=failed),
                    ]
                )

                # Apply the action
                action.apply(game_state)

            # Print actions table
            print(tabulate(action_rows, headers=action_headers, tablefmt="grid", maxcolwidths=[None, 20, 40]))
        print()

        # Update turn counter
        game_state.executed_turns += 1

        # Show game state after this turn
        print(parse_state(game_state))

        # Show storage summary
        storage_structures = _get_storage_structures(game_state)
        if storage_structures:
            print("\nStorage summary:")
            storage_headers = ["Structure", "Resources"]
            storage_rows = []
            for s in storage_structures:
                resources = format_inventory_multiline(s["structure"].storage)
                storage_rows.append([s["name"], resources])
            print(
                tabulate(
                    storage_rows,
                    headers=storage_headers,
                    tablefmt="simple",
                    maxcolwidths=[None, 50],
                )
            )

        # Show connected resources to base
        base = game_state.base
        non_base_resources = game_state.non_base_storage_resources()
        non_base_inventory = format_inventory_multiline(non_base_resources)
        print("\nNon-base Storage:")
        for line in non_base_inventory.splitlines():
            print(f"  {line}")

        print()

        if remaining_pause:
            has_more_turns = turn_idx + 1 < len(plan)
            if stop_turn is not None:
                has_more_turns = has_more_turns and turn_idx < stop_turn

            if has_more_turns:
                remaining_pause = _prompt_next_turn()


def _get_storage_structures(game_state: GameState) -> list[dict[str, Any]]:
    """Get all structures that can actually store resources."""
    from game_simulation.structure_mixins import StorageStructure

    storage_structures = []

    for structure in game_state.structures.values():
        # Only include structures that have storage capability
        if isinstance(structure, StorageStructure):
            name = f"{structure.type.value}({structure.x},{structure.y})"
            storage_structures.append({"name": name, "structure": structure, "x": structure.x, "y": structure.y})

    return storage_structures
