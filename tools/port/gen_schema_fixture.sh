#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:?usage: gen_schema_fixture.sh <cl-revenue-ops-r repo path>}"
TMP=$(mktemp -d)
python3 - "$TMP/fixture.db" <<'EOF'
import sys, os
from unittest.mock import MagicMock
mock = MagicMock(); mock.Plugin = MagicMock; mock.RpcError = Exception
sys.modules.setdefault("pyln", mock); sys.modules.setdefault("pyln.client", mock)
sys.path.insert(0, os.getcwd())
from modules.database import Database
db = Database(sys.argv[1], MagicMock())
db.initialize()
EOF
mkdir -p "$OUT_DIR/fixtures"
sqlite3 "$TMP/fixture.db" .schema > "$OUT_DIR/fixtures/schema.sql"
cp "$TMP/fixture.db" "$OUT_DIR/fixtures/fixture.db"
echo "tables: $(sqlite3 "$TMP/fixture.db" "SELECT count(*) FROM sqlite_master WHERE type='table'")"
