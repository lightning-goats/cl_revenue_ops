# Golden characterization tests (refactor Phase 0)

These freeze the CURRENT behavior of each principal decision class
(docs/planning/refactor.md, Test strategy → Golden behavioral tests) so
the refactor can prove semantic parity.

- Fixtures: `fixtures/<class>/<scenario>.json` — canonical JSON
  (sorted keys) written by `GOLDEN_UPDATE=1`.
- Every module ALSO contains at least one hand-computed assertion (not
  golden) so a recorded fixture full of nonsense can't self-certify.
- Re-recording policy: see `util.py` docstring. Fixture diffs in review
  ARE the behavior-change review.

| Decision class | Test module |
|---|---|
| Fee damping/floor | `test_golden_fee_damping.py` |
| Dynamic htlc_max | `test_golden_htlcmax.py` |
| Rebalance planning | `test_golden_rebalance_planner.py` |
| Profitability class/role | `test_golden_profitability.py` |
| Close protection | `test_golden_close_protection.py` |
| Boltz auto-cycle plan | `test_golden_boltz_cycle.py` |

Deliberately NOT goldened in Phase 0: the unclamped DTS/PID fee target
(`_adjust_channel_fee`) — it samples a Thompson posterior with unseeded
`random` (see `docs/refactor/phase0/portability-hazards.md`). Its
determinism arrives with Phase 1 clock/seed injection.
