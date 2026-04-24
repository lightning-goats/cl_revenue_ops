# Capex Planner Final A/B Comparison

- Code under test: capex budget floor, EV-first open selection, executable-slot sizing, dry-run cooldown filtering, candidate EV diagnostics
- Polar mode: planner dry-run, one cycle each, cl-hive enabled vs disabled
- Safety: planner settings restored to `planner_enabled=false` and `planner_dry_run=false` after each dry-run

| metric | no hive | hive enabled |
| --- | ---: | ---: |
| discovered candidates | 3 | 6 |
| hive open candidates | 0 | 3 |
| fleet-tier capex channels | 0 | 7 |
| fleet exploration budget sats | 5000 | 5000 |
| global capex envelope sats | 5710 | 6400 |
| dry-run opens | 1 | 1 |
| selected open EV sats | 9090 | 9090 |
| selected open amount sats | 3999660 | 3999660 |

Final result: cl-hive no longer degrades the selected open after EV-first
selection and executable-slot sizing. It expands the candidate set and gives
fleet capex treatment to hive-member channels, while the selected open remains
at the same EV/size level as no-hive in this lab snapshot.

Remaining question: the single-cycle EV metric does not yet value strategic
hive topology beyond its rebalance-bias impact. Longer tournaments should
compare realized forwarding/rebalance ROI from hive-selected peers over time.
