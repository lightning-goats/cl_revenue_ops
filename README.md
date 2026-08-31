# cl-revenue-ops

`cl-revenue-ops` runs routing economics for one Core Lightning node. It adjusts
fees, plans and executes circular rebalances, enforces spend limits, and reports
channel profitability.

It is fully standalone: decisions use only this node's forwards, gossip, and
state. It does not open or close channels, execute swaps, or withdraw funds.

## Requirements

- Core Lightning `v26.06.7+` from a maintainer-verified release artifact
- Python `3.10+`
- Core Lightning's `bookkeeper` plugin (recommended for better cost accounting)

See [Core Lightning compatibility](docs/CORE_LIGHTNING_COMPATIBILITY.md) for the
version floor and smoke-test notes.

## Install

```bash
cd ~/.lightning/plugins
git clone https://github.com/lightning-goats/cl_revenue_ops.git
cd cl_revenue_ops
python3 -m pip install -r requirements.txt
chmod +x cl-revenue-ops.py
lightning-cli plugin start "$(pwd)/cl-revenue-ops.py"
```

For automatic startup, add this to the Core Lightning config, using the full
path on your machine:

```ini
plugin=/home/your-user/.lightning/plugins/cl_revenue_ops/cl-revenue-ops.py
```

The plugin executes by default. To observe without changing fees or sending
rebalances, add this before the first start:

```ini
revenue-ops-dry-run=true
```

Start from [the minimal config](config/cl-revenue-ops.conf.minimal). The
[full config](config/cl-revenue-ops.conf.full) documents advanced settings.

## First checks

```bash
lightning-cli revenue-status
lightning-cli revenue-fee-debug
lightning-cli -k revenue-rebalance-debug summary_only=true
lightning-cli -k revenue-budget section=total_cost
lightning-cli revenue-profitability
```

These are read-only. Use them for decision explainability before changing
settings: they show what the plugin decided, what held it back, and how much
budget remains.

## Safety controls

Pause automated fee and rebalance execution immediately:

```bash
lightning-cli revenue-config set paused true
```

Resume it:

```bash
lightning-cli revenue-config set paused false
```

Inspect the supported runtime controls before changing them:

```bash
lightning-cli revenue-config get
lightning-cli revenue-config list-mutable
```

The main operator controls are:

| Control | Purpose |
| --- | --- |
| `paused` | Kill switch for automated fee and rebalance execution |
| `daily_budget_sats` / `weekly_budget_sats` | Rebalance fee-spend limits |
| `min_fee_ppm` / `max_fee_ppm` | Fee rails |
| `fee_profile` | `active` or `conservative` fee behavior |
| `authority_level` | Maximum authority granted to the plugin |
| `risk_profile` | Named safety bundle; preview it before selecting it |
| `acquisition_experiment_enabled` | Bounded cold-lane fee experiment; enabled by default |

Examples:

```bash
lightning-cli revenue-config set daily_budget_sats 5000
lightning-cli revenue-config set min_fee_ppm 100
lightning-cli revenue-config set acquisition_experiment_enabled false
lightning-cli revenue-profile-preview conservative
```

Runtime overrides are stored in the plugin database and take precedence over
config-file or `setconfig` values. Remove one with
`lightning-cli revenue-config reset <key>`; the response says whether a restart
is required.

`paused` does not disable read-only reports. To disable Python fee authority
separately, follow the
[fee-authority handoff runbook](docs/runbooks/python-fee-authority-handoff.md).

## Daily operation

Useful read-only reports:

```bash
lightning-cli revenue-report summary
lightning-cli revenue-dashboard
lightning-cli -k revenue-budget section=total_cost
lightning-cli revenue-profitability
lightning-cli revenue-health
```

The plugin normally runs its own scheduled cycles. `revenue-cycle`, manual
rebalance/fee commands, spend-ledger mutation commands, and policy writes are
operator actions; do not use them for monitoring. See the
[action RPC inventory](docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md) for
the complete read/action classification.

If behavior looks wrong, pause first, then capture `revenue-status`,
`revenue-fee-debug`, `revenue-rebalance-debug`, and the relevant `getlog`
entries before restarting or changing controls.

## Upgrade

```bash
lightning-cli plugin stop cl-revenue-ops
git -C ~/.lightning/plugins/cl_revenue_ops pull --ff-only
lightning-cli plugin start ~/.lightning/plugins/cl_revenue_ops/cl-revenue-ops.py
```

The SQLite database is kept outside the repository by default at
`~/.lightning/revenue_ops.db`. Back it up before major upgrades.

## Reference

- [v3.0.0 release notes](docs/releases/v3.0.0.md)
- [Public telemetry contracts](docs/contracts/README.md)
- [Action RPC inventory](docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md)
- [Repository maintenance](docs/maintenance.md)
- [License](LICENSE) — BSD 3-Clause
