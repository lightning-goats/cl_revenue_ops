# Production configuration optimization prompt

Copy the prompt below to the privileged lightning agent on the production node.
It is deliberately evidence-first and config-file-first. Replace no values by
guesswork, and do not paste credentials or private node data into reports.

```text
You are optimizing the production configuration for the standalone Core
Lightning plugin cl_revenue_ops on this node.

Primary objective
-----------------
Maximize durable net routing profit while preserving payment reliability,
liquidity safety, budget enforcement, and operator control. Raw forward count
or raw volume is not the objective unless it improves net profit. Prefer the
persistent lightningd configuration file for every setting that has a real CLN
plugin option. Use `revenue-config` DB overrides only for controls that truly
have no config-file option or for an explicitly approved staged experiment.

Known environment and invariants
--------------------------------
- Production host: 10.8.0.1.
- The running process has used `--lightning-dir=/data/lightningd`; verify this
  live rather than assuming it remains true.
- Work from the freshly pulled `origin/main` checkout of cl_revenue_ops and
  record `git rev-parse HEAD` in the report.
- cl_revenue_ops is standalone. Do not add or configure Sling, Hive, Mycelium,
  fleet coordination, swaps, automatic channel opens/closes, or xrebalance.
- Do not open, close, pay, withdraw, rebalance, set fees, change policies,
  restart anything, or mutate plugin/runtime configuration during discovery.
- Never invoke action RPCs during discovery, including
  `revenue-rebalance-cycle`, `revenue-fee-cycle`, `revenue-set-fee`,
  `revenue-rebalance`, `revenue-spend-*`, `revenue-analyze`,
  `revenue-wake-all`, `revenue-ignore`, `revenue-unignore`,
  `revenue-cleanup-closed`, `revenue-clear-reservations`,
  `revenue-policy set`, or `revenue-config set/reset`.
- Do not transplant Polar's 15-second tournament cadence into production.
- Do not globally collapse fees to laboratory floors. Derive market levels
  from this node's own peers, gossip, forwards, liquidity, and profit history.
- Treat a 25,000-sat daily rebalance budget as high-risk unless live realized
  contribution and measured refill returns clearly support it.

Phase 1: discover the authoritative surfaces
--------------------------------------------
1. Identify the live lightningd process arguments, CLN version, network,
   lightning directory, actual main config file, any include files, plugin
   executable path, plugin checkout, and current plugin commit. Do not assume
   `/data/lightningd/config` is the only file merely because the lightning
   directory is `/data/lightningd`.
2. Read the repository's `AGENTS.md`, `README.md`,
   `config/cl-revenue-ops.conf.full`, `modules/config.py`, and every
   `plugin.add_option` declaration in `cl-revenue-ops.py`.
3. Build an exact mapping of:
   - persistent CLN option name, such as `revenue-ops-...`;
   - internal Config field;
   - current config-file value and its source file;
   - live `listconfigs` value;
   - current `revenue-config get` value;
   - whether an active DB override shadows the file;
   - whether the key is config-file-capable, runtime-only, deprecated, or
     internal.
4. Honor the documented precedence rule: an active `revenue-config set`
   override wins over config-file and `setconfig` values. Never claim that a
   file edit took effect while a DB override still shadows it.

Phase 2: collect read-only evidence
-----------------------------------
Capture timestamped JSON locally on the production node from read-only
surfaces. Use bounded output where supported. At minimum collect:

- `getinfo`, `listconfigs`, `plugin list`, and `listpeerchannels`;
- `listchannels` policies relevant to this node's current channel peers;
- settled forwards over 7-day and 30-day windows, separated by channel and
  direction;
- aggregate completed outgoing payment/rebalance cost evidence over the same
  windows without exposing payment secrets;
- `revenue-status`;
- `revenue-config get` and `revenue-config list-mutable`;
- `revenue-fee-debug`;
- `revenue-rebalance-debug summary_only=true`;
- `revenue-profitability` without an action/refresh flag;
- explicitly read-only budget/ledger views, such as
  `revenue-budget section=total_cost` and `revenue-budget section=ledger`;
- wallet reserve, on-chain feerate, channel uptime/health, pending HTLCs, and
  current advertised base fee, ppm, htlc_min, and htlc_max.

Do not use `revenue-budget` without an explicit read-only section if the local
RPC inventory classifies the no-argument form as mixed. Do not call a refresh,
apply, analyze, cycle, set, reset, cleanup, or wake operation while collecting
evidence. If an RPC is absent or malformed, record it as unknown; do not replace
missing evidence with zero.

Phase 3: derive candidate values
--------------------------------
For every configurable setting, keep the current value unless live evidence
supports a change. Produce a table with: key, current file value, live value,
shadowing override, proposed value, evidence, expected economic effect, safety
bound, confidence, config/runtime surface, and rollback value.

Use these decision rules:

1. Fees
   - Estimate peer-local competing base/ppm distributions from current gossip,
     weighted by relevance to our actual channels and recent forwarding demand.
   - Compare realized revenue, routed volume, elasticity after historical fee
     changes, outbound inventory, refill cost, and stale-gossip failures.
   - Preserve a realistic global operating floor. Use bounded, channel-local
     acquisition/market-boundary mechanisms for conversion rather than a
     network-wide race to 0 or 1 ppm.
   - The bounded one-lane acquisition controller is now default-on after
     crossed CLN/LND validation. Keep
     `acquisition_experiment_enabled=true` unless live evidence identifies a
     concrete reason to opt out; do not loosen its fixed duration, volume,
     opportunity-cost, liquidity, restoration, or cooldown rails.
     Its material-evidence wake path does not justify shortening the configured
     production fee interval; keep cadence decisions evidence-based.
   - Set `min_fee_ppm`, `min_fee_ppm_saturated`, `max_fee_ppm`, fee profile,
     market-boundary values, and related fee controls only from observed bands
     and explicit cost-recovery requirements.

2. HTLC admission
   - The current tested defaults are 0.85 for source, sink, and balanced
     `htlcmax` class caps. The effective target is still capped at 85% of live
     spendable outbound and never above capacity.
   - Keep these truthful defaults unless production failure, concurrency, or
     liquidity evidence demonstrates a safer value. Do not reinstate a low
     fixed class cap merely to shape flow: probabilistic pathfinders interpret
     a low advertised htlc_max as weak route capacity, which can suppress even
     small profitable forwards.

3. Rebalancing and budgets
   - Derive daily and weekly budgets from realized trailing contribution,
     measured successful refill costs, expected incremental routing revenue,
     wallet reserve, and failure rate—not from desired activity.
   - Require positive expected value after complete route fees and opportunity
     cost. Preserve hard spend ceilings, reservation accounting, cooldowns,
     emergency liquidity rails, the 20% emergency floor, and the 60-second
     settlement grace.
   - Do not raise budgets to make the rebalancer appear active. Zero automatic
     spend is correct when no positive-EV route exists.

4. Cadence
   - Choose production fee, flow, reporting, and rebalance intervals from this
     node's forwarding rate, gossip propagation, channel count, database cost,
     and decision churn.
   - Ensure enough new evidence can arrive between learning decisions. Avoid
     fee gossip churn and overlapping rebalance settlement windows. Polar's
     compressed 15-second cadence is forbidden in production.

5. Risk and reliability
   - Preserve authority, pause, dry-run, wallet reserve, force ceilings,
     malformed-data neutrality, and no-action read-only behavior.
   - Prefer conservative values when 7-day and 30-day evidence disagree.
   - Do not enable a default-off experiment globally without a bounded rollout,
     success metric, expiry, and exact rollback.

Phase 4: produce the config-file-first proposal
------------------------------------------------
1. Create a proposed dedicated cl_revenue_ops config include if the live CLN
   configuration supports includes; otherwise edit the actual authoritative
   config file. Preserve unrelated operator settings and comments.
2. Put every supported `revenue-ops-*` plugin option in that persistent file.
   Do not duplicate the same key across files. If duplicates already exist,
   identify precedence and consolidate only after operator approval.
3. Create a second, minimal runtime-only manifest containing only controls
   proven not to have a plugin/config-file option. The governance controls
   documented as runtime-only—such as applicable `econ_*`, `authority_level`,
   and `risk_profile` keys—belong here. Explain why each cannot live in the
   file.
4. For every DB override that shadows a proposed file value, create an ordered
   reset plan. Record its old value first. Do not execute `revenue-config reset`
   until the config file is staged, syntax-reviewed, backed up, and the
   operator approves the exact diff and restart plan.
5. Write these artifacts without secrets:
   - optimized persistent config proposal;
   - minimal runtime-only manifest;
   - evidence/recommendation table;
   - exact apply sequence;
   - exact rollback sequence;
   - post-restart verification checklist.

Phase 5: approval and application gate
--------------------------------------
Stop and show the operator:
- source files and commit inspected;
- evidence window timestamps and completeness;
- exact proposed config diff;
- runtime-only keys and why they cannot be in the file;
- shadowing overrides and ordered resets;
- restart scope;
- expected economic effect and risks;
- exact rollback.

Do not change configuration, reset overrides, pull code, or restart the plugin
until the operator explicitly approves that exact plan. After approval:

1. Back up every changed config/include file with ownership and mode preserved.
2. Apply the persistent config atomically.
3. Reset only the approved shadowing overrides, in the approved order.
4. Apply only the approved runtime-only controls.
5. Restart only the cl_revenue_ops plugin unless a lightningd restart is
   demonstrably required and separately approved.
6. Read back `listconfigs`, `revenue-config get`, `revenue-status`, fee debug,
   and rebalance debug. Prove every intended value is active and unshadowed.
7. Observe at least two native production cycles without forcing a fee or
   rebalance cycle. Confirm no crash, no unexpected action, no duplicate spend,
   no safety violation, and no abnormal gossip churn.
8. If any gate fails, execute the documented rollback immediately and report
   the exact evidence.

Final report format
-------------------
- files changed;
- old and new values;
- evidence and confidence per change;
- config-file values versus unavoidable runtime-only values;
- shadowed overrides reset;
- tests/checks run;
- readback after restart;
- no-Sling/no-coordination confirmation;
- action RPCs invoked, if any, with operator approval reference;
- production compatibility notes;
- rollback status and follow-up risks.
```
