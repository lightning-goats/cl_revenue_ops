# cl-revenue-ops

`cl-revenue-ops` is the local execution layer for routing profit and liquidity management on Core Lightning. It watches channel economics, adjusts fees, and executes rebalancing through Sling while keeping the normal operator surface intentionally small.

## What Operators Need To Know

- This is the executor. It owns local fee execution, local rebalance execution, and Sling integration.
- The normal runtime controls are `paused`, `daily_budget_sats`, `min_fee_ppm`, and `max_fee_ppm`.
- The primary operator surfaces are `revenue-status`, `revenue-fee-debug`, `revenue-rebalance-debug`, and `revenue-hive-status`.
- The normal workflow is decision explainability first, knob tuning second.
- Auto fee bands are enabled by default. Manual policy bands are fallback only when an auto band is not yet available.
- Hive-member channels stay at the hive fee, usually `0` ppm. Saturated-hive desaturation works by raising competing non-hive exits, not by repricing the hive channel itself.
- `revenue-policy list|get|find|changes` are diagnostic surfaces. Write actions such as `set` and `delete` remain internal or debug workflows, not the normal operator path.

## Architecture

```text
cl-hive (optional coordination inputs)
    ↓
cl-revenue-ops (local execution layer)
    ↓
Sling (required rebalance engine)
    ↓
Core Lightning
```

With `cl-hive`, the same local executor gets better fleet signals: coordinated corridor hints, peer quality, competition signals, and saturated-hive egress bias. Without `cl-hive`, decisions stay local.

## Install

### Requirements

- Core Lightning `v23.05+`
- Python `3.10+`
- Sling plugin: required and owned by `cl-revenue-ops`
- bookkeeper plugin: recommended for cleaner P&L and cost accounting
- `cl-hive`: optional, for fleet coordination inputs

### Start The Plugin

```bash
cd ~/.lightning/plugins
git clone https://github.com/lightning-goats/cl_revenue_ops.git
cd cl_revenue_ops
pip install -r requirements.txt
chmod +x cl-revenue-ops.py
lightning-cli plugin start "$(pwd)/cl-revenue-ops.py"
```

### Upgrade

```bash
lightning-cli plugin stop cl-revenue-ops
git -C ~/.lightning/plugins/cl_revenue_ops pull
lightning-cli plugin start ~/.lightning/plugins/cl_revenue_ops/cl-revenue-ops.py
```

## Day-1 Operator Workflow

1. Start the plugin and check `revenue-status`.
2. Set only the safety rails you actually want to constrain: `paused`, `daily_budget_sats`, `min_fee_ppm`, `max_fee_ppm`.
3. Let the executor run.
4. Use `revenue-fee-debug` and `revenue-rebalance-debug` to understand holds, clamps, and actions before touching anything else.
5. If you are in a fleet, use `revenue-hive-status` to confirm that coordination inputs are available.

Example runtime adjustments:

```bash
lightning-cli revenue-config get
lightning-cli revenue-config set min_fee_ppm 75
lightning-cli revenue-config set daily_budget_sats 10000
```

## Primary RPCs

| Command | Use |
|---|---|
| `revenue-status` | Health, operator controls, and latest fee/rebalance decisions |
| `revenue-fee-debug` | Why a fee moved, held, or was clamped |
| `revenue-rebalance-debug` | Why a rebalance was selected, skipped, or blocked |
| `revenue-hive-status` | Whether `cl-hive` coordination inputs are available |
| `revenue-config get` | Inspect current runtime controls |
| `revenue-config set <key> <value>` | Change one of the supported runtime controls |
| `revenue-analyze` | Trigger immediate analysis |
| `revenue-wake-all` | Wake the background loops immediately |

## How `cl-hive` Changes Behavior

`cl-hive` does not replace the executor. It augments it.

When available, `cl-revenue-ops` can consume:

- coordinated fee recommendations
- corridor ownership and competition signals
- peer quality and defense data
- shared velocity and traffic intelligence
- saturated-hive egress bias signals

Those inputs adjust local decisions, but local execution still happens here.

## More Detail

- Minimal config example: [config/cl-revenue-ops.conf.minimal](config/cl-revenue-ops.conf.minimal)
- Full config example: [config/cl-revenue-ops.conf.full](config/cl-revenue-ops.conf.full)
- Roadmap: [docs/planning/ROADMAP.md](docs/planning/ROADMAP.md)
