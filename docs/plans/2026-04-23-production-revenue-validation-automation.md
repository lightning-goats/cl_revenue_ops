# Production Revenue Validation Automation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a control-host daily automation pipeline that collects production validation evidence from `lnnode` and `hive-nexus-02`, evaluates rollout-watch checks, and drafts T+14/T+28 reports for the PR #87/#88/#89 validation window.

**Architecture:** Add one repo-managed daily orchestrator plus focused helper scripts for collection, watch evaluation, and report generation. Configure the system with a single YAML file and run it through `systemd --user` at `06:00 America/Denver`, writing immutable raw artifacts under `results/revenue-validation/` and generated drafts under `docs/reports/`.

**Tech Stack:** Python 3.12, `pyln-client`-style CLI probing via `subprocess`, `PyYAML`, `pytest`, `systemd --user`.

---

### Task 1: Add Shared Validation Config And Common Helpers

**Files:**
- Modify: `requirements.txt`
- Create: `config/revenue_validation.yaml`
- Create: `tools/revenue_validation_common.py`
- Test: `tests/test_revenue_validation_common.py`

**Step 1: Write the failing tests**

```python
from tools import revenue_validation_common as mod


def test_load_config_reads_nodes_and_schedule(tmp_path):
    cfg = tmp_path / "revenue_validation.yaml"
    cfg.write_text(
        """
schedule:
  timezone: America/Denver
  run_time: "06:00"
nodes:
  lnnode:
    t0: "2026-04-23T00:00:00Z"
    transport: ["ssh", "lnnode"]
""".strip()
    )
    data = mod.load_config(cfg)
    assert data["schedule"]["run_time"] == "06:00"
    assert data["nodes"]["lnnode"]["transport"] == ["ssh", "lnnode"]


def test_build_command_wraps_remote_transport():
    node = {"transport": ["ssh", "lnnode"]}
    cmd = mod.build_node_command(node, "lightning-cli getinfo")
    assert cmd == ["ssh", "lnnode", "lightning-cli getinfo"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_revenue_validation_common.py -v`
Expected: FAIL with import or attribute errors because the helper module does not exist yet.

**Step 3: Write minimal implementation**

Create `tools/revenue_validation_common.py` with:

- `load_config(path)` using `yaml.safe_load`
- `build_node_command(node_cfg, remote_cmd)` for `ssh ...` and `docker exec ...` transports
- path helpers for dated output directories
- JSON write helpers that create parent directories
- a small `RunResult` shape for command success/failure recording

Add `PyYAML` to `requirements.txt`.

Create `config/revenue_validation.yaml` with:

```yaml
schedule:
  timezone: America/Denver
  run_time: "06:00"
paths:
  results_root: results/revenue-validation
  reports_root: docs/reports
nodes:
  lnnode:
    t0: "REPLACE_ME"
    transport: ["ssh", "lnnode"]
    lightning_cli_prefix: "lightning-cli --lightning-dir=/data/lightningd"
    log_extract_command: "grep -E 'FEE:|REBALANCE_FLOOR|competition_aware|INITIAL_FEE|Hive member|Traceback|Error' /data/lightningd/cln.log | tail -100000"
  hive-nexus-02:
    t0: "REPLACE_ME"
    transport: ["docker", "exec", "cl-hive-node-hive-nexus-02", "sh", "-lc"]
    lightning_cli_prefix: "lightning-cli --lightning-dir=/data/lightning/bitcoin"
    log_extract_command: "grep -E 'FEE:|REBALANCE_FLOOR|competition_aware|INITIAL_FEE|Hive member|Traceback|Error' /data/lightning/bitcoin/bitcoin/log | tail -100000"
thresholds:
  rollback:
    plugin_restart_limit_24h: 3
    revenue_drop_pct: 25
    rebalance_success_floor_pct: 50
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_revenue_validation_common.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add requirements.txt config/revenue_validation.yaml tools/revenue_validation_common.py tests/test_revenue_validation_common.py
git commit -m "feat: add revenue validation config and common helpers"
```

### Task 2: Build The Daily Evidence Collector

**Files:**
- Create: `tools/revenue_validation_collect.py`
- Test: `tests/test_revenue_validation_collect.py`

**Step 1: Write the failing test**

```python
from tools import revenue_validation_collect as mod


def test_collect_writes_expected_snapshot_files(tmp_path, monkeypatch):
    fake_responses = {
        "revenue-dashboard 30": {"financial_health": {}, "period": {}},
        "revenue-report summary": {"status": "ok"},
        "revenue-profitability": {"channels": []},
    }

    def fake_run(node_cfg, command):
        key = command.replace("lightning-cli --lightning-dir=/data/lightningd ", "")
        return mod.CommandResult(ok=True, stdout_json=fake_responses.get(key, {}), stderr="", returncode=0)

    monkeypatch.setattr(mod, "run_json_rpc", fake_run)
    out = mod.collect_node_day(
        node_name="lnnode",
        node_cfg={"lightning_cli_prefix": "lightning-cli --lightning-dir=/data/lightningd"},
        day_dir=tmp_path,
    )
    assert (tmp_path / "revenue-dashboard-30.json").exists()
    assert (tmp_path / "revenue-report-summary.json").exists()
    assert out["status"] == "ok"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_revenue_validation_collect.py -v`
Expected: FAIL because the collector module does not exist yet.

**Step 3: Write minimal implementation**

Create `tools/revenue_validation_collect.py` with:

- one canonical list of per-node collection commands:
  - `revenue-dashboard 30`
  - `revenue-report summary`
  - `revenue-profitability`
  - `revenue-status`
  - `revenue-config get`
  - `listforwards`
  - `listpays`
  - `listpeerchannels`
  - `hive-members`
  - `feerates perkb`
- logic to write immutable files under `results/revenue-validation/YYYY-MM-DD/<node>/`
- a top-level manifest recording node success/failure
- a normalized trend record appended to `results/revenue-validation/trends/<node>.jsonl`

Trend record fields should include at least:

```json
{
  "date": "2026-04-23",
  "node": "lnnode",
  "t0": "2026-04-23T00:00:00Z",
  "days_since_t0": 0,
  "gross_revenue_sats_30d": 18843,
  "net_profit_sats_30d": 12850,
  "opex_sats_30d": 5993,
  "forward_count_30d": 466,
  "volume_sats_30d": 68602516,
  "planner_enabled": true,
  "planner_execute_closes": true,
  "planner_max_opens_per_cycle": 1,
  "planner_max_closes_per_cycle": 1
}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_revenue_validation_collect.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/revenue_validation_collect.py tests/test_revenue_validation_collect.py
git commit -m "feat: add daily revenue validation collector"
```

### Task 3: Implement Rollout Watch Evaluation

**Files:**
- Create: `tools/revenue_validation_watch.py`
- Test: `tests/test_revenue_validation_watch.py`

**Step 1: Write the failing tests**

```python
from tools import revenue_validation_watch as mod


def test_detects_red_flag_when_non_hive_channel_hits_zero_ppm():
    status = {
        "channel_states": [],
        "operator_controls": {"values": {}},
    }
    peerchannels = {
        "channels": [
            {"peer_id": "02peer", "fee_proportional_millionths": 0}
        ]
    }
    hive_members = {"members": []}
    result = mod.check_zero_ppm_non_hive(peerchannels, hive_members)
    assert result["severity"] == "red"


def test_detects_yellow_flag_for_traceback_burst():
    lines = ["Traceback: boom"] * 11
    result = mod.check_traceback_volume(lines)
    assert result["severity"] == "yellow"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_revenue_validation_watch.py -v`
Expected: FAIL because the watch module does not exist yet.

**Step 3: Write minimal implementation**

Create `tools/revenue_validation_watch.py` with:

- helpers for the specific red/yellow checks from the approved plan
- support for evaluating:
  - plugin restart count threshold
  - zero ppm on non-hive channels
  - sustained ceiling pricing
  - rebalance success-rate drop
  - revenue drop threshold
  - traceback volume
  - `REBALANCE_FLOOR` volume
  - competition-aware oscillation signals
- daily findings JSON written to `results/revenue-validation/watch/YYYY-MM-DD.json`
- non-zero exit on red flags or partial node failure

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_revenue_validation_watch.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/revenue_validation_watch.py tests/test_revenue_validation_watch.py
git commit -m "feat: add revenue validation watch checks"
```

### Task 4: Generate Draft T+14 And T+28 Reports

**Files:**
- Create: `tools/revenue_validation_report.py`
- Create: `docs/reports/.gitkeep`
- Test: `tests/test_revenue_validation_report.py`

**Step 1: Write the failing test**

```python
from tools import revenue_validation_report as mod


def test_generates_t14_report_when_checkpoint_reached(tmp_path):
    trends_dir = tmp_path / "trends"
    trends_dir.mkdir()
    (trends_dir / "lnnode.jsonl").write_text(
        '{"date":"2026-05-07","node":"lnnode","days_since_t0":14,"gross_revenue_sats_30d":100}\\n'
    )
    out = tmp_path / "docs" / "reports"
    out.mkdir(parents=True)
    mod.generate_checkpoint_reports(
        trends_root=trends_dir,
        reports_root=out,
        config={"nodes": {"lnnode": {"t0": "2026-04-23T00:00:00Z"}}},
    )
    files = list(out.glob("*-production-t14-findings.md"))
    assert len(files) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_revenue_validation_report.py -v`
Expected: FAIL because the report module does not exist yet.

**Step 3: Write minimal implementation**

Create `tools/revenue_validation_report.py` with:

- logic to read saved trend files and daily watch findings
- per-node checkpoint-state evaluation from configured `T0`
- Markdown generation for:
  - T+14 behavior report
  - T+28 economic report
- explicit separation of:
  - observed evidence
  - computed comparisons
  - inconclusive findings

The generated report structure must follow the approved validation plan sections.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_revenue_validation_report.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/revenue_validation_report.py docs/reports/.gitkeep tests/test_revenue_validation_report.py
git commit -m "feat: add production validation report generator"
```

### Task 5: Add The Daily Orchestrator And Systemd User Units

**Files:**
- Create: `tools/revenue_validation_daily.py`
- Create: `tools/systemd/revenue-validation-daily.service`
- Create: `tools/systemd/revenue-validation-daily.timer`
- Modify: `README.md`
- Test: `tests/test_revenue_validation_daily.py`

**Step 1: Write the failing test**

```python
from tools import revenue_validation_daily as mod


def test_daily_pipeline_runs_collect_watch_and_report_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(mod, "run_collect", lambda *a, **k: calls.append("collect") or 0)
    monkeypatch.setattr(mod, "run_watch", lambda *a, **k: calls.append("watch") or 0)
    monkeypatch.setattr(mod, "run_report", lambda *a, **k: calls.append("report") or 0)
    code = mod.main(["--config", "config/revenue_validation.yaml"])
    assert code == 0
    assert calls == ["collect", "watch", "report"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_revenue_validation_daily.py -v`
Expected: FAIL because the orchestrator module does not exist yet.

**Step 3: Write minimal implementation**

Create `tools/revenue_validation_daily.py` with:

- `main()` that runs collect, watch, and report in sequence
- partial-failure handling that still writes what succeeded
- a final process exit code reflecting red flags or collection failures

Create `tools/systemd/revenue-validation-daily.service`:

```ini
[Unit]
Description=Daily production revenue validation pipeline
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/sat/bin/cl_revenue_ops
ExecStart=/usr/bin/python3 /home/sat/bin/cl_revenue_ops/tools/revenue_validation_daily.py --config /home/sat/bin/cl_revenue_ops/config/revenue_validation.yaml
```

Create `tools/systemd/revenue-validation-daily.timer`:

```ini
[Unit]
Description=Run production revenue validation daily at 06:00 America/Denver

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Update `README.md` with a short operator section covering:

- required `T0` edits in `config/revenue_validation.yaml`
- how to install the user units
- how to trigger a manual run
- where to find saved artifacts and generated reports

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_revenue_validation_daily.py -v`
Expected: PASS

**Step 5: Run focused verification**

Run:

```bash
pytest tests/test_revenue_validation_common.py \
  tests/test_revenue_validation_collect.py \
  tests/test_revenue_validation_watch.py \
  tests/test_revenue_validation_report.py \
  tests/test_revenue_validation_daily.py -v
```

Expected: PASS for all new automation tests.

**Step 6: Commit**

```bash
git add tools/revenue_validation_daily.py tools/systemd/revenue-validation-daily.service tools/systemd/revenue-validation-daily.timer README.md tests/test_revenue_validation_daily.py
git commit -m "feat: add daily production revenue validation scheduler"
```

### Task 6: Install And Verify The User-Level Timer On The Control Host

**Files:**
- Use: `tools/systemd/revenue-validation-daily.service`
- Use: `tools/systemd/revenue-validation-daily.timer`

**Step 1: Install unit files**

Run:

```bash
mkdir -p ~/.config/systemd/user
cp tools/systemd/revenue-validation-daily.service ~/.config/systemd/user/
cp tools/systemd/revenue-validation-daily.timer ~/.config/systemd/user/
systemctl --user daemon-reload
```

Expected: no errors.

**Step 2: Trigger one manual run**

Run:

```bash
systemctl --user start revenue-validation-daily.service
systemctl --user status revenue-validation-daily.service --no-pager
```

Expected: service exits with status `0` when both nodes collect cleanly and no red flags fire, or non-zero with saved partial artifacts if a failure is intentionally surfaced.

**Step 3: Enable the timer**

Run:

```bash
systemctl --user enable --now revenue-validation-daily.timer
systemctl --user list-timers revenue-validation-daily.timer --no-pager
```

Expected: next run scheduled for `06:00`.

**Step 4: Verify artifacts exist**

Run:

```bash
find results/revenue-validation -maxdepth 3 -type f | sort | tail -n 20
```

Expected: daily raw evidence files, manifest, watch JSON, and trend files are present.

**Step 5: Commit installation docs only if repo changes were required**

```bash
git add README.md tools/systemd/revenue-validation-daily.service tools/systemd/revenue-validation-daily.timer
git commit -m "docs: add revenue validation timer install flow"
```
