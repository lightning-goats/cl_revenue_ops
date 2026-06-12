# Intent Contract: modules/demand_flow.py

Tier 2 — medium treatment. Audited 2026-06-12 against commit 9f8f219.

## Purpose

`DemandFlowClassifier` (modules/demand_flow.py:53) labels Lightning nodes as
source/sink/router/unknown for the capacity planner's channel-open candidate scoring.
It has two independent classifiers: `classify_peers` aggregates our own per-channel
`FlowMetrics` into per-peer net-flow profiles (real evidence), and `classify_candidate`
scores arbitrary nodes from gossip heuristics — alias keyword lists (exchanges, wallets,
LSPs), channel-structure shape, fee-policy shape, and liquidity-ad presence (weak
evidence, confidence-weighted). A third helper, `find_sink_adjacent_candidates`,
proposes open targets adjacent to our strongest sink peers, on the theory that liquidity
flowing toward a sink can be monetized one hop upstream. Stateless; no RPC, DB, or
datastore access.

## Inputs / Outputs

- **Caller**: `capacity_planner._discover_from_demand_flow`
  (modules/capacity_planner.py:2088–2120) — instantiates the classifier (:2091), runs
  `classify_peers(all_flow)` (:2094), filters role=="sink", and calls
  `find_sink_adjacent_candidates(sink_profiles, sink_channels, existing_peers)` (:2110).
  Cached profiles are reused for candidate annotation at
  modules/capacity_planner.py:2394–2395.
- **Inputs**: flow analyzer `FlowMetrics` objects (attrs `peer_id`, `sats_in`,
  `sats_out`); gossip `listnodes`-shaped `node_info` and `listchannels`-shaped channel
  dicts supplied by the planner.
- **Output**: `NodeFlowProfile` dataclass (:43–50) — role, confidence, net_flow_ratio,
  gossip_signals, has_liquidity_ads — and candidate dicts
  `{peer_id, source: "demand_flow", score, reason, sink_peer_id, is_sink_adjacent}`
  (:222–229).
- **Note**: `classify_candidate` (:97–191) has **no production caller** — only tests
  reference it; only `classify_peers` and `find_sink_adjacent_candidates` are live.

## Invariants

- **DF-1** Flow-based roles use a ±0.3 net-flow-ratio threshold:
  ratio = (in − out)/(in + out); > 0.3 → source, < −0.3 → sink, else router; zero total
  volume → role "unknown" with confidence 0 (:69–95).
- **DF-2** Flow confidence is volume-scaled and bounded to [0.1, 0.9] via
  `0.3·log10(total)/log10(1e6)` clamped both sides (:85–86) — no peer is ever certain.
  Note the 0.9 ceiling is unreachable in practice: the formula yields 0.3 at 1M sats of
  volume and would need ~1e18 sats to hit 0.9, so realized confidences live near
  [0.1, ~0.5].
- **DF-3** Gossip classification is a normalized argmax: role = highest of
  source/sink/router keyword+structure+fee scores, confidence = winning share of the
  total, and zero total score yields ("unknown", 0.0) (:170–183).
- **DF-4** Malformed gossip cannot raise: numeric fields pass through `_safe_float`
  (:33–40) / `parse_msat`, and only dict-typed active channels are considered (:128);
  bad data degrades scores rather than aborting candidate scoring.
- **DF-5** Sink-adjacency only proposes new, active peers: candidates already in
  `existing_peers` or previously seen are skipped, inactive channels are skipped, at most
  top-5 sinks are expanded and at most 10 candidates returned, sorted by score
  (:203–233).
- **DF-6** Sink-adjacency scores are deterministic and bounded:
  `0.4 · sink_confidence · (1 + (n_sinks − rank)/n_sinks)` (:221) — ≤ 0.8 for any
  confidence ≤ 1.0, and ≤ 0.72 under DF-2's 0.9 flow-confidence cap (in practice ≤ ~0.4
  given DF-2's realistic confidence range).

## Revenue role

Indirect, speculative. It influences only which open candidates the capacity planner
surfaces/scores; a wrong classification wastes open capex on a poorly placed channel, a
right one positions the node upstream of payment demand. No direct spend or fee action.

## Observable surface

Not directly observable. Its fingerprints appear inside capacity-planner candidate
records (`planner_candidates` DB rows with `source="demand_flow"` and reasons mentioning
"Adjacent to sink", via modules/database.py:6553) and any planner status RPC output that
includes those candidates. No hermes artifact isolates this module.

## Uncertainties

- Dead code or future surface? `classify_candidate` and its keyword lists (:16–30) are
  untouched by production paths (verified: only tests/test_demand_flow.py calls it); if
  the planner is meant to gossip-classify open candidates, that wiring is missing.
- Within `classify_candidate`, the `fee_extractive` signal (:159–164) is recorded into
  `gossip_signals` at −0.2 but is never added to any role score — it cannot influence
  the argmax in DF-3. Dead signal inside dead code.
- Keyword lists are static and English-biased; staleness (new exchanges/LSPs) silently
  lowers recall. No process refreshes them.
- The sink-adjacency theory (open to peers one hop upstream of sinks) is asserted, not
  validated against realized routing revenue of past demand_flow-sourced opens.
- `find_sink_adjacent_candidates` needs `sink_channels` supplied by the caller; the
  quality of that gossip snapshot (depth, freshness) bounds DF-5 and was not audited.
