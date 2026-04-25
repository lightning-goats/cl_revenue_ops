# Repository Maintenance

## Artifact Policy

Source, tests, fixtures, configuration examples, and curated documentation belong in git.

Generated experiment runs should stay local by default. This includes timestamped directories under `results/` such as `fee-*`, `hive-hints-truth-*`, `module-loop-*`, and `rebalancer-polar-mcp-*`. If a run produces a durable conclusion, move the summary into `docs/` or intentionally force-add a small top-level `results/*.md` report.

Raw JSON, cache directories, local databases, virtual environments, vendored checkouts, and local worktrees should not be committed.

## Local Cleanup

Preview local cleanup:

```bash
scripts/clean-local.sh
```

Remove Python caches and pytest cache:

```bash
scripts/clean-local.sh --apply
```

Also remove generated result directories:

```bash
scripts/clean-local.sh --artifacts --apply
```

Also remove expensive-to-recreate local directories such as `.venv`, `.worktrees`, and `vendor`:

```bash
scripts/clean-local.sh --heavy --apply
```

## Tracked Artifact Debt

Some historical artifacts are already tracked. Removing them from future commits should be done deliberately with `git rm --cached` or `git rm` after confirming the data is no longer needed in repository history.

Current candidates:

- `MagicMock/mock/*`
- timestamped raw run output under `results/`
