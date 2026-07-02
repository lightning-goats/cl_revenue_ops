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
