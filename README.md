# Game Solver CLI Overview  

You start with a ready-to-use solver package. It contains the command-line interface (`main.py`), a deterministic simulation sandbox that mirrors server rules, example strategies, and simple visualizers.  

This is not a finished solver — the included strategy is deliberately minimal and only clears the trivial ramp-up level. What you *do* have is a working environment: it runs out of the box, talks to the Recruitment Game Server, and lets you fetch tasks, simulate them, and inspect plans. From here, your job is to extend it with real strategies.  

## Running with uv

Use [uv](https://github.com/astral-sh/uv) to avoid managing Python environments manually. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then execute any command from inside the `game_solver` directory:

```bash
uv run python main.py <command> [options]
```

uv creates an isolated environment and resolves dependencies on demand.

Before running anything, copy `.env.example` to `.env` and fill in the required API details (token, base URL, etc.) so the CLI can authenticate against the Recruitment Game Server. The CLI reads `GAME_SERVER_BASE_URL` and `GAME_SERVER_TOKEN` (it still accepts the legacy `ARBITER_*` names for backward compatibility).

## Layout

- `main.py` – command-line interface entry point. Implements `advance`, `solve-task`, `tasks`, and `solve-interactive`.
- `game_simulation/` – deterministic sandbox mirroring server rules. It exposes `GameState`, inventory helpers, and concrete structure models (e.g., base, roads, quarries).
- `strategies/` – in-tree strategies. Each subclass of `BaseStrategy` emits actions through a generator.
- `visualizers/` – helpers that render board states, inventories, and plan tables for debugging.

## How GameState Works

`GameState` ingests the level definition JSON and tracks:

- Structures on the board, keyed by `(x, y)`.
- Inventory contents for every storage-capable structure.
- Turn counters (`executed_turns`, `max_turns`).
- Goal resources as an `Inventory` for quick comparisons.

Calling `turn_start()` resets per-turn flags (for example `ExtractionStructure.extracted_this_turn`) so strategies can account for which structures have already acted. Any action you apply directly mutates this state: builds add structures, extraction increases quarry storage, transfer moves resources between storages, and so on. Because strategies share the same `GameState` instance, they always reason about the latest board snapshot.

## BaseStrategy Flow

`BaseStrategy.generate_more_turn_actions()` is a generator. The framework consumes it inside a loop:

1. `turn_start()` resets per-turn state.
2. The generator yields an action.
3. The framework applies the action immediately, so future yields observe the updated `GameState`.
4. Once the generator stops yielding, that batch of actions is appended to the plan for the turn.

If the loop iterates more than `BaseStrategy.ACTION_LOOP_LIMIT` times (default 100) without stopping, `StrategyFailed` is raised to prevent runaway loops. Raise the constant if you truly need longer turns.

## CLI Commands

Run the CLI through uv:

```bash
uv run python main.py <command> [...]
```

### `advance`

Pulls the next unsolved level for your account and runs the chosen strategy. Example with the minimalist `ramp_up` strategy, pausing between turns:

```bash
uv run python main.py advance ramp_up --plan-step
```

`--plan-step` pauses after the initial state and between turns; press any key to continue, or `Esc` to stop pausing.
Add `--loop` to keep requesting new tasks automatically until one fails.

### `solve-task`

Replays a specific task—useful when testing changes against known levels:

```bash
uv run python main.py solve-task ramp_up --stage 1 --task-id <uuid> --show-plan --plan-step
```

Add `--show-board` to display the level before simulation starts.

### `tasks`

Lists all tasks currently available to your account (grouped by stage). Helpful when auditing backlog or checking that submissions cleared.

```bash
uv run python main.py tasks
```

### `solve-interactive`

Launches a curses-based task browser. Navigation: arrow keys (or `j/k`), `Enter` to solve, `r` to refresh. Step-through pauses are on by default; disable them with `--no-plan-step`:

```bash
uv run python main.py solve-interactive ramp_up
```

## Extending Strategies  

The baseline strategy is intentionally trivial — it only works on the single ramp-up level. Its role is to show you the mechanics: how actions are generated, applied to the `GameState`, and turned into a valid plan.  

Your work starts here. To try your own ideas, copy an existing file (for example `strategies/ramp_up.py`) and adjust `generate_more_turn_actions()`. Keep in mind:  

1. Actions mutate the shared `GameState` immediately, so each new yield sees the latest board.  
2. Per-turn flags (like `extracted_this_turn`) tell you if a structure has already acted.  
3. You must emit a `ClaimWinAction` yourself once the goal is satisfied.  
4. `BaseStrategy.ACTION_LOOP_LIMIT` (default 100) prevents runaway loops; raise it only if you truly need more actions in one turn.  

This package is your environment: it fetches levels, simulates them locally, and gives you the tools to visualize and debug. The strategy is yours to build.  
