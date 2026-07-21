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
  setting, channel-open initial-fee handling, direct controller fee-evaluation
  and direct controller wake boundaries, dynamic `htlcmax` updates, wake
  requests, the final `setchannel` execution boundary, failed-forward fee
  nudges, and policy-change wakeups. It does not stop read-only status or
  policy inspection, settled-forward accounting, or the observational
  replay-capture control.
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

Only after separate operator approval, run the persistent change manually and
validate Core Lightning's nested response in the same command:

```bash
lightning-cli setconfig \
  config=revenue-ops-fee-authority-enabled val=false transient=false \
  | jq -e '
      .config.value_bool == false
      and (.config.source | type == "string")
      and (.config.source
           | test("(^|/)config[.]setconfig:[1-9][0-9]*$"))
    '
```

The response contract is nested under `.config`. The source is not the literal
string `config.setconfig`; it is a basename or absolute path plus a positive
line number. Live read-only evidence for another dynamic option was
`/data/lightningd/bitcoin/config.setconfig:1`. The regular expression above
accepts that path and `config.setconfig:1`, but rejects another basename,
missing line suffix, line zero, or a non-boolean value.

Then read `revenue-fee-authority-status` and require `enabled=false`,
`reason=setconfig`, and the expected generation advance. Re-read status after
the next scheduled fee interval to prove the timer obeyed the disabled gate and
did not perform a transition.

This cutover disables Python fee mutation authority only. It does not authorize
broadcasts by the replacement.

## Rollback boundaries

### Pre-arm abort (pre-arm and pre-Rust-activation only)

A short Python-only restore is allowed **pre-arm and pre-Rust-activation only**:
no cutover arm has ever been installed for the candidate, Rust fee activation
has never been attempted, Rust remains observer/dry-run/no-broadcast, and its
mutation count is unchanged. Under those conditions only, restore and verify
the persistent Python value:

```bash
lightning-cli setconfig \
  config=revenue-ops-fee-authority-enabled val=true transient=false \
  | jq -e '
      .config.value_bool == true
      and (.config.source | type == "string")
      and (.config.source
           | test("(^|/)config[.]setconfig:[1-9][0-9]*$"))
    '
```

Then require `revenue-fee-authority-status` to report `enabled=true`,
`reason=setconfig`, and the expected generation advance. A transient restore
is not a substitute for this persistent rollback.

### Post-activation rollback

Once an arm has been installed or Rust fee activation has been attempted,
never re-enable Python first. Use this order to prevent overlapping authority:

1. Disable Rust fee broadcasts and positively verify the live executor cannot
   begin another broadcast.
2. Verify that no Rust fee batch is active; wait for or abort the batch through
   the reviewed Rust operational interface.
3. Remove the cutover arm and verify it is absent.
4. Reconcile or quarantine every ambiguous Rust action before another
   authority can act.
5. Restore the checksummed prior Rust shadow artifact if required.
6. Re-enable Python fee authority persistently and validate the nested
   `.config.value_bool == true` and source-path contract with the exact
   `jq -e` check from the pre-arm section.
7. Require positive Python status `enabled=true`, then confirm Rust is
   observer/dry-run/no-broadcast and its mutation count remains stable.
8. Preserve the incident evidence and reset promotion status.

Do not compress this sequence into the pre-arm shortcut merely because the
rollback is urgent.

## Reverting to source that does not register the option

A persistent `config.setconfig` entry can prevent older source from loading if
that source does not register `revenue-ops-fee-authority-enabled`. Prefer a
rollback artifact that retains the dynamic option. If older source is required,
the following is a manual pre-revert safety sequence and must not be automated:

1. Complete the applicable rollback boundary above: Rust broadcasts disabled,
   no batch active, arm absent, and every ambiguity reconciled or quarantined.
2. While the current Python release is still loaded and still registers the
   option, re-enable Python fee authority persistently and pass both the nested
   `jq -e` response check and positive status readback.
3. Use the returned source field to identify the resolved `config.setconfig` path.
   Back it up with ownership, mode, and checksum recorded.
4. Re-read that file immediately before editing, identify by option name rather
   than trusting a possibly shifted line number, and remove exactly the active
   `revenue-ops-fee-authority-enabled` entry with an operator-reviewed atomic
   edit. Abort if the match is absent or ambiguous; preserve all unrelated
   entries and original ownership/mode.
5. Verify the persistent file no longer contains an active entry for the
   option while the still-running plugin continues to report authority enabled.
6. Only then revert to source that does not register the option and follow the
   established checksummed plugin-only rollback procedure.

Do not unload or replace the current plugin before step 5: it is the component
that can positively prove Python authority is enabled while persistent cleanup
is performed.
