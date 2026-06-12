# Intent Contract: modules/__init__.py

## Purpose
Package initializer for the `modules` package that eagerly re-exports the original core API:
`FlowAnalyzer`/`ChannelState`/`FlowMetrics` (flow_analysis), `FeeController`,
`EVRebalancer`/`RebalanceCandidate` (rebalancer), `Config`, `Database`, `DataService`, and
`PolicyManager`/`FeeStrategy`/`RebalanceMode`/`PeerPolicy`. In practice most code imports
submodules directly (`from modules.config import Config`); the package-level re-exports are a
convenience surface frozen at roughly the v1.4 era.

## Consumers / dependencies
- Consumers: `cl-revenue-ops.py` (`from modules import flow_analysis as flow_analysis_mod`, plus
  direct submodule imports), tests importing `modules.*` paths.
- Dependencies (eager imports): `modules/flow_analysis.py`, `modules/fee_controller.py`,
  `modules/rebalancer.py`, `modules/config.py`, `modules/database.py`, `modules/data_service.py`,
  `modules/policy_manager.py` — importing ANY `modules.*` submodule transitively imports all
  seven.

## Invariants
- INIT-1: `import modules` succeeds in a bare environment with only the repo on `sys.path` (the
  eager imports must not require a running CLN plugin or RPC connection at import time).
- INIT-2: `modules.ChannelState` is `flow_analysis.ChannelState`, NOT
  `rebalance_state_v2.ChannelState` — consumers of the package-level name get the flow-analysis
  type.
- INIT-3: `__all__` matches the names actually imported; `from modules import *` raises no
  AttributeError.

## Sanity check
`python3 -c "import modules; assert set(modules.__all__) <= set(dir(modules)); from modules.flow_analysis import ChannelState as A; assert modules.ChannelState is A"`
from the repo root (also implicitly verified by the entire test suite importing `modules.*`).

## Notes
- The docstring and export list are stale: they describe the v1.4-era six-module layout and omit
  the ~25 newer modules (v2/v3 rebalance pipeline, boltz, capex, hive, planner, etc.), which are
  not re-exported. Misleading as documentation of the package.
- Name collision hazard: `ChannelState` exported here is unrelated to the frozen
  `rebalance_state_v2.ChannelState`; auditors and IDE auto-imports can pick the wrong one.
- Eager importing makes every submodule import pull in the heavy v1 modules (rebalancer is
  several thousand lines), which slightly slows test startup but creates no cycles today.
