# Liquidity executor decommission runbook

Date: 2026-08-03. Target contract: cl_revenue_ops 3.0.0.

This runbook removes plugin authority for CapacityPlanner, automatic channel
open/close, planner defibrillation, Boltz, and LN+. It retains revenue
reporting, fee control, profitability analysis, and budget-constrained
circular rebalancing.

Do not perform the production cutover until the Python commit and the matching
Rust retirement have independent Tier-1 approval. Commands below are an
operator procedure; creating this runbook does not authorize running them.

## Production paths and private evidence

The Lightning directory is `/data/lightningd` and its config is
`/data/lightningd/config`. Resolve the loaded plugin entrypoint from
`lightning-cli plugin list`; do not assume its checkout path. Resolve the
database path from the active plugin configuration, expanding it relative to
`/data/lightningd` when needed.

Keep the cutover record private. A suitable location is
`/data/lightningd/private/revenue-ops-decommission-20260803.md` with mode
0600. Record timestamps, reviewed commit IDs, artifact hashes, configuration
hashes, database backup hash, command exit statuses, and aggregate counts.
Never copy node IDs, peer IDs, channel IDs, wallet addresses, credentials, or
raw RPC payloads into repository artifacts.

## 1. Read-only preflight

Run these before changing config or stopping the plugin:

```bash
lightning-cli --lightning-dir=/data/lightningd getinfo
lightning-cli --lightning-dir=/data/lightningd plugin list
lightning-cli --lightning-dir=/data/lightningd revenue-status
lightning-cli --lightning-dir=/data/lightningd revenue-health
lightning-cli --lightning-dir=/data/lightningd revenue-fee-authority-status
lightning-cli --lightning-dir=/data/lightningd revenue-rebalance-debug
lightning-cli --lightning-dir=/data/lightningd revenue-profitability
lightning-cli --lightning-dir=/data/lightningd revenue-total-cost-budget
lightning-cli --lightning-dir=/data/lightningd revenue-spend-ledger
lightning-cli --lightning-dir=/data/lightningd revenue-boltz-status
lightning-cli --lightning-dir=/data/lightningd revenue-lnplus-status
```

Stop if the plugin entrypoint or database path is ambiguous, health is already
red, a rebalance settlement is pending without an understood outcome, Boltz
reports any pending swap, LN+ reports an unresolved application or opening, the
reviewed commits do not match the artifacts, or independent approvals are
missing. A read timeout is not evidence of a safe idle state. The SQLite
preflight does not inspect the external Boltz daemon or journal, so the old
runtime status check is mandatory.

First create the private evidence directory with mode 0700. Run the repository
preflight against the resolved database and write its output
to the private evidence directory. The tool opens SQLite in read-only and
query-only mode, refuses symlinks and overwrites, verifies unresolved LN+ rows,
active retired reservations, and generic no_close tags, and creates a mode-0600
report:

```bash
umask 077
install -d -m 0700 /data/lightningd/private/revenue-ops-decommission-20260803
.venv/bin/python tools/liquidity_decommission_preflight.py \
  --db /resolved/path/to/revenue_ops.db \
  --output /data/lightningd/private/revenue-ops-decommission-20260803/preflight.json
```

## 2. Pin the old runtime inert

Before stopping the old plugin, verify all old executor gates are false through
both CLN config and any database override surface. The exact old names include:

- `planner_enabled=false`
- `planner_execute_closes=false`
- `planner_max_opens_per_cycle=0`
- `planner_max_closes_per_cycle=0`
- `lnplus_swaps_enabled=false`
- `lnplus_execute_applications=false`
- `boltz_enabled=false`
- `boltz_auto_cycle_enabled=false`
- `expansion_treasury_enabled=false`

Use read-only `revenue-config get <key>` and `listconfigs` first. Apply a
change only under the separately approved production cutover. Read back every
value after changing it. Database overrides win over file defaults in the old
runtime, so a false config-file value alone is insufficient.

Render two configs before the outage. Use the repository renderer so the v3
file removes retired options and the rollback file contains one exact false
gate for every retired executor:

```bash
umask 077
.venv/bin/python tools/render_liquidity_decommission_config.py \
  --input /data/lightningd/config \
  --active-output /data/lightningd/private/revenue-ops-decommission-20260803/config.v3 \
  --rollback-output /data/lightningd/private/revenue-ops-decommission-20260803/config.rollback
```

- active v3 config: every removed `revenue-ops-planner-*`,
  `revenue-ops-lnplus-*`, `revenue-ops-boltz-*`, diagnostic-rebalance,
  and expansion-treasury line absent;
- rollback config: compatible with the old plugin, but every executor gate
  explicitly false and every old database override verified false.

Unknown removed options make v3 startup fail, which is a safe failure but still
an outage. Validate the active rendered config before stopping anything.

## 3. Backups and immutable evidence

With restrictive permissions, hash the current entrypoint, tracked plugin
files, and both rendered configs. Back up the revenue database with SQLite
online backup while the old plugin is running, then hash the backup. Record the
reviewed Python and Rust commit IDs.

Capture aggregate before-state only:

```sql
SELECT count(*), coalesce(max(created_at), 0) FROM planner_actions;
SELECT count(*), coalesce(max(applied_at), 0) FROM lnplus_swaps;
SELECT category, count(*), coalesce(max(timestamp), 0)
  FROM spend_events
 WHERE category IN (char(98,111,108,116,122), char(99,104,97,110,110,101,108,95,111,112,101,110), char(99,104,97,110,110,101,108,95,99,108,111,115,101))
 GROUP BY category;
```

The character expressions avoid embedding action labels in copied shell
history. Store only aggregate results in the private record.

## 4. Stop, install, and start

Limit the outage to this plugin:

1. Stop `cl-revenue-ops` through `lightning-cli plugin stop`.
2. Confirm no process for the old plugin remains.
3. Install the exact independently reviewed artifact at the resolved
   entrypoint. Do not deploy an uncommitted worktree.
4. Verify the installed hashes against the approved artifact manifest.
5. Start the resolved entrypoint with `lightning-cli plugin start`.
6. Confirm plugin 3.0.0 appears once in `plugin list`.

Do not restart lightningd and do not change peers, channels, fees, wallets, or
payments as part of this lifecycle.

## 5. Read-only post-cutover verification

Repeat the retained preflight RPCs. Also confirm:

- `revenue-status` contains fee, reporting, profitability, controls, and
  rebalance state without planner, Boltz, or LN+ scheduler state;
- `revenue-rebalance-debug` is available and budgets show daily, weekly,
  global, and per-channel constraints;
- `lightning-cli help` lists exactly the reviewed 39-method plugin surface;
- no `revenue-planner*`, `revenue-boltz*`, or `revenue-lnplus*` method is
  registered;
- `revenue-cycle` accepts only `fees`, `rebalance`, `flow`, and `all`;
- `revenue-budget` has no planner or Boltz section.

Use help enumeration to establish absence. Do not invoke removed action names
against production merely to observe an error.

Inspect the plugin process and thread names without signaling it. No planner,
defibrillation, Boltz, LN+, or obligations-watcher thread may exist. Retained
fee, flow, reporting, and rebalance loops may exist.

Run the aggregate database queries again after a bounded observation window.
Planner/LN+/historical swap-open-close maxima and counts must not advance.
Normal rebalance accounting may advance only if the approved retained runtime
controls permit it.

## 6. Private no_close check

Using the operator private peer/channel record, compare the expected
`no_close` policy set before and after cutover. Do not paste identifiers into
tickets, logs, or this repository. The v3 plugin preserves generic
`no_close` metadata but has no close executor. A mismatch blocks completion
because it can affect manual or external workflows even though v3 cannot close
a channel.

## 7. Rollback

Rollback on startup failure, method-contract mismatch, duplicate plugin
process, missing retained reporting, budget-gate regression, database error, or
unexpected executor thread/action evidence.

1. Stop the v3 plugin only.
2. Restore the reviewed old artifact and the rollback config.
3. Verify all old planner, LN+, Boltz, auto-cycle, and expansion-treasury gates
   remain false in both config and database overrides.
4. Start the old plugin.
5. Repeat the read-only health and aggregate no-action checks.

Never roll back to an old artifact with an executor gate true. If inertness
cannot be proven, leave the plugin stopped and escalate to the operator.

## Completion evidence

Completion requires independent Python and Rust approvals, reviewed artifact
hashes, successful retained read-only RPC checks, absent removed help/thread
surfaces, unchanged historical executor aggregates, preserved private
`no_close` records, and a documented rollback path. Production completion is
separate from code completion.
