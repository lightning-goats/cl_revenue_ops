# Python Fee Authority Handoff Runbook

This runbook defines the operator contract for the Python fee-authority gate. It
does not authorize a live handoff. A planned cutover requires separate operator
approval after the reviewed source has been deployed and verified default-on.

## Contract

- `revenue-ops-fee-authority-enabled` is a dynamic Core Lightning option. It is
  not a `revenue-config` database override.
- Authority defaults to `true`. While enabled, the existing Python fee
  evaluation and `setchannel` path remains authoritative.
- Disabling authority does not start another implementation, restart a plugin,
  or grant broadcast permission anywhere else. The replacement must remain in
  observer/dry-run/no-broadcast mode unless a separate change authorizes more.
- A disabled gate blocks scheduled and manual fee evaluation, manual fee
  setting, wake requests, the final `setchannel` execution boundary,
  failed-forward fee nudges, and policy-change wakeups. It does not stop
  read-only status or policy inspection, settled-forward accounting, or the
  observational replay-capture control.
- Authority transitions are manual. Initialization adopts the configured value,
  and the dynamic `setconfig` callback applies operator changes. Timers never
  change authority; scheduled fee work only checks and obeys the gate.

## Inspect authority

Run this before and after every transition:

```bash
lightning-cli revenue-fee-authority-status
```

The response schema is `revenue_ops_fee_authority/v1`:

- `enabled` is the current in-process gate value.
- `generation` is an in-process transition counter. It advances only when the
  effective boolean changes; repeating the current value is idempotent.
- `transitioned_at` is the Unix timestamp of the latest effective transition.
- `observed_at` is the status-read timestamp.
- `reason` is `initial`, `init`, or `setconfig`, according to the transition
  source inside the plugin.

`generation` is process-local and must not be compared across plugin restarts.

## Transient disable and restore

These runtime-only commands are for a separately approved rehearsal or an
explicitly temporary response. `transient=true` does not establish the durable
live cutover state.

Disable Python fee authority for the current runtime:

```bash
lightning-cli setconfig \
  config=revenue-ops-fee-authority-enabled val=false transient=true
```

Immediately run `lightning-cli revenue-fee-authority-status` and require
`enabled=false`, `reason=setconfig`, and a generation advance if the prior value
was true.

Restore Python fee authority for the current runtime:

```bash
lightning-cli setconfig \
  config=revenue-ops-fee-authority-enabled val=true transient=true
```

Immediately run `lightning-cli revenue-fee-authority-status` and require
`enabled=true`, `reason=setconfig`, and a generation advance if the prior value
was false. Never leave a rehearsal disabled.

## Separately approved live cutover

A live cutover is manual-only. Do not place it in an executable script, service,
timer, or scheduled job. Deploying the reviewed code does not perform the
cutover; deployment must leave Python authority enabled.

Before cutover, require all of the following:

1. The reviewed Python source is deployed and healthy with authority enabled.
2. A rollback source and checksum are staged through the established deployment
   process.
3. The replacement remains verified observer/dry-run/no-broadcast.
4. A fresh status read reports `enabled=true`.

Only after separate operator approval, run the persistent change manually:

```bash
lightning-cli setconfig \
  config=revenue-ops-fee-authority-enabled val=false transient=false
```

Inspect the direct `setconfig` response before proceeding. Its returned source
must be exactly `config.setconfig`, and its effective value must be `false`.
Stop and roll back if either check fails. Then read
`revenue-fee-authority-status` and require `enabled=false`, `reason=setconfig`,
and the expected generation advance. Re-read status after the next scheduled fee
interval to prove the timer obeyed the disabled gate and did not perform a
transition.

This cutover disables Python fee mutation authority only. It does not authorize
broadcasts by the replacement.

## Persistent rollback

If the cutover is aborted or verification fails, restore the durable default
manually:

```bash
lightning-cli setconfig \
  config=revenue-ops-fee-authority-enabled val=true transient=false
```

Inspect the direct response and require source `config.setconfig` with effective
value `true`. Then require `revenue-fee-authority-status` to report
`enabled=true`, `reason=setconfig`, and the expected generation advance. A
temporary `transient=true` restore is not a substitute for persistent rollback
after a persistent cutover.
