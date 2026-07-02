# Deep-Audit Findings Ledger

Per P6 of the deep-audit campaign. One row per finding. `file:line@blob` pins
the finding to a specific line and the git blob it was found in, so coverage
accounting (`tools/audit/deep_manifest.py --coverage`) can map the finding onto
its chunk. A chunk that contains a cited line counts as COVERED by that finding.

`status` values: `OPEN`, `FIXED`, `BEHAVIORAL-HOLD`, `WONTFIX`, `DUPLICATE`.
`severity`: `Critical`, `High`, `Medium`, `Low`, `Info`.
`dimension`: one of the seven never-audited dimensions or `line-audit` /
`docs-conformance` (see the plan).

Rows whose ID contains `EXAMPLE` are ignored by the coverage tool.

| ID | severity | dimension | file:line@blob | description | status | fix_commit | test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXAMPLE-000 | Low | line-audit | modules/utils.py:42@e96115ef457c35700b65329422a01584c43ff6ba | EXAMPLE ROW ONLY — illustrates the citation format; ignored by --coverage. Replace with real findings. | OPEN | | |
| P1-011 | High | line-audit | modules/hive_hints.py:242@9e5eec5b1b40c71eb67431352157c73ab2d5215b | Untrusted ["hive","hints"] datastore value parsed without byte-size/entry-count caps; a multi-MB or >200-entry payload could make the fee path hang. Mirror producer caps (DATASTORE_MAX_BYTES=900_000, MAX_PEERS_IN_SNAPSHOT=200) as read-side defensive caps; oversized -> absent/stale neutral, logged once. | FIXED | 8a4651c | tests/test_hive_hints_payload_caps.py |
| P1-022 | Medium | line-audit | modules/hive_hints.py:1507@9e5eec5b1b40c71eb67431352157c73ab2d5215b | _metabolic_ttl_for did not clamp the metabolic/immune section TTL, so a huge section ttl_seconds could defeat section-freshness. Clamp to HINT_MAX_TTL_SECONDS (86400). | FIXED | 1909664 | tests/test_hive_hints_section_ttl_clamp.py |
| P1-023 | Low | line-audit | modules/hive_hints.py:1035@9e5eec5b1b40c71eb67431352157c73ab2d5215b | get_optimal_fee_estimate lacked the upper bound get_fleet_fee_prior uses; out-of-range untrusted estimate could surface. Bound to [1, MAX_FLEET_FEE_PRIOR_PPM]. | FIXED | 95a302e | tests/test_hive_hints_optimal_fee_bounds.py |
| P1-015 | High | line-audit | modules/boltz_manager.py:1826@b740d8b793156dd2ad954b90d66d1cb232016905 | Free-form positional args (refund swap_id+dest, claim dest, loop_out address, swap_status swap_id, withdraw destination) passed to boltzcli without a `--` end-of-options terminator; a value beginning with `-` could be reparsed as a flag. Insert `--` before positionals (argv-only, no shell). | FIXED | 37f4e99 | tests/test_boltz_argv_terminator.py |
| P1-006 | High | line-audit | modules/boltz_manager.py:1879@b740d8b793156dd2ad954b90d66d1cb232016905 | withdraw() did not validate the destination address format before the subprocess call (address-validation part only; amount cap + budget gate remain BEHAVIORAL-HOLD). Add self-contained bech32/bech32m + base58check + Liquid blech32 structural validation for the wallet currency/network; malformed/wrong-network -> clean rejection. | FIXED | 9637c54 | tests/test_boltz_withdraw_address_validation.py |
| P1-030 | High | line-audit | modules/database.py:3753@d96b136d6ea7485bd08cb235528295b5b8769ce9 | reserve_spend used INSERT OR REPLACE which could resurrect a terminal ('spent'/'released') reservation_id back to 'active', double-counting it. Add a status guard rejecting re-reservation of terminal reservation_ids. | FIXED | 86014e1 | tests/test_database_reserve_spend_guard.py |
