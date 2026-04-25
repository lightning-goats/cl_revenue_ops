#!/usr/bin/env bash
set -euo pipefail

# Collect CLN production-validation artifacts from a node where sat has
# passwordless sudo as the lightningd user for lightning-cli/tail.
#
# Usage:
#   ./tools/production_validation_node_collect.sh hive-nexus-01 10.8.0.1
#
# Output:
#   ./artifacts/production-validation/<node-name>/<timestamp>/

NODE_NAME="${1:?node name required}"
NODE_HOST="${2:?ssh host required}"
STAMP="$(date +%F-%H%M%S)"
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/artifacts/production-validation/${NODE_NAME}/${STAMP}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 sat@${NODE_HOST}"
REMOTE_LOG_PATH="${REMOTE_LOG_PATH:-/data/lightningd/bitcoin/debug.log}"

mkdir -p "$OUT_DIR"

remote_json() {
  local cmd="$1"
  local outfile="$2"
  ${SSH} "sudo -n -u lightningd ${cmd}" > "$OUT_DIR/$outfile"
}

remote_tail() {
  local lines="$1"
  local outfile="$2"
  ${SSH} "sudo -n -u lightningd tail -${lines} ${REMOTE_LOG_PATH}" > "$OUT_DIR/$outfile"
}

remote_json "lightning-cli getinfo" getinfo.json
remote_json "lightning-cli listpeerchannels" listpeerchannels.json
remote_json "lightning-cli listforwards" listforwards.json
remote_json "lightning-cli listpays" listpays.json
remote_json "lightning-cli revenue-config" revenue-config.json || true
remote_json "lightning-cli revenue-profitability" revenue-profitability.json || true
remote_json "lightning-cli hive-members" hive-members.json || true
remote_json "lightning-cli feerates perkb" feerates.json
remote_json "lightning-cli bkpr-listaccountevents" bkpr-listaccountevents.json || true
remote_json "lightning-cli listtransactions" listtransactions.json || true

# Capture recent relevant log surface for rollback-watch and activation checks.
remote_tail 200000 debug-tail.log
grep -E "FEE:|REBALANCE_FLOOR|competition_aware|INITIAL_FEE|Hive member|Traceback|Error|estimated_closure_cost|_estimate_close_cost|coordination_reserved_slots|rebalance_coordination" "$OUT_DIR/debug-tail.log" > "$OUT_DIR/debug-signals.log" || true

cat > "$OUT_DIR/README.txt" <<EOF
node_name=${NODE_NAME}
node_host=${NODE_HOST}
collected_at=${STAMP}
collector_host=$(hostname)
remote_log_path=${REMOTE_LOG_PATH}
EOF

echo "$OUT_DIR"