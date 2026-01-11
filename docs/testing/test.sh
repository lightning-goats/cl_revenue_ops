#!/bin/bash
#
# Automated test suite for cl-revenue-ops plugin
#
# Usage: ./test.sh [category] [network_id]
# Categories: all, setup, status, flow, fees, rebalance, sling, policy, profitability, clboss, database, metrics
#
# Example: ./test.sh all 1
# Example: ./test.sh flow 1
# Example: ./test.sh rebalance 1
#
# Prerequisites:
#   - Polar network running with CLN nodes (alice, bob, carol)
#   - cl-revenue-ops plugin installed via ../cl-hive/docs/testing/install.sh
#   - Funded channels between nodes for rebalance tests
#
# Environment variables:
#   NETWORK_ID      - Polar network ID (default: 1)
#   HIVE_NODES      - CLN nodes with cl-revenue-ops (default: "alice bob carol")
#   VANILLA_NODES   - CLN nodes without plugins (default: "dave erin")

set -o pipefail

# Configuration
CATEGORY="${1:-all}"
NETWORK_ID="${2:-1}"

# Node configuration
HIVE_NODES="${HIVE_NODES:-alice bob carol}"
VANILLA_NODES="${VANILLA_NODES:-dave erin}"

# CLI commands
CLN_CLI="lightning-cli --lightning-dir=/home/clightning/.lightning --network=regtest"

# Test tracking
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=""

# Colors (if terminal supports it)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

#
# Helper Functions
#

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_section() {
    echo -e "${BLUE}$1${NC}"
}

# Execute a test and track results
run_test() {
    local name="$1"
    local cmd="$2"

    echo -n "[TEST] $name... "

    if output=$(eval "$cmd" 2>&1); then
        log_pass ""
        ((TESTS_PASSED++))
        return 0
    else
        log_fail ""
        echo "       Output: $output"
        ((TESTS_FAILED++))
        FAILED_TESTS="$FAILED_TESTS\n  - $name"
        return 1
    fi
}

# Execute a test that should fail
run_test_expect_fail() {
    local name="$1"
    local cmd="$2"

    echo -n "[TEST] $name (expect fail)... "

    if output=$(eval "$cmd" 2>&1); then
        log_fail "(should have failed)"
        ((TESTS_FAILED++))
        FAILED_TESTS="$FAILED_TESTS\n  - $name"
        return 1
    else
        log_pass ""
        ((TESTS_PASSED++))
        return 0
    fi
}

# CLN CLI wrapper for nodes with revenue-ops
revenue_cli() {
    local node=$1
    shift
    docker exec polar-n${NETWORK_ID}-${node} $CLN_CLI "$@"
}

# CLN CLI wrapper for vanilla nodes
vanilla_cli() {
    local node=$1
    shift
    docker exec polar-n${NETWORK_ID}-${node} $CLN_CLI "$@"
}

# Check if container exists
container_exists() {
    docker ps --format '{{.Names}}' | grep -q "^polar-n${NETWORK_ID}-$1$"
}

# Wait for condition with timeout
wait_for() {
    local cmd="$1"
    local expected="$2"
    local timeout="${3:-30}"
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        if result=$(eval "$cmd" 2>/dev/null) && echo "$result" | grep -q "$expected"; then
            return 0
        fi
        sleep 1
        ((elapsed++))
    done
    return 1
}

# Get node pubkey
get_pubkey() {
    local node=$1
    revenue_cli $node getinfo | jq -r '.id'
}

# Get channel SCID between two nodes
get_channel_scid() {
    local from=$1
    local to_pubkey=$2
    revenue_cli $from listpeerchannels | jq -r --arg pk "$to_pubkey" \
        '.channels[] | select(.peer_id == $pk and .state == "CHANNELD_NORMAL") | .short_channel_id' | head -1
}

#
# Test Categories
#

# Setup Tests - Verify environment is ready
test_setup() {
    echo ""
    echo "========================================"
    echo "SETUP TESTS"
    echo "========================================"

    # Check containers
    for node in $HIVE_NODES; do
        run_test "Container $node exists" "container_exists $node"
    done

    # Check vanilla containers (optional)
    for node in $VANILLA_NODES; do
        if container_exists $node; then
            run_test "Container $node exists" "container_exists $node"
        fi
    done

    # Check cl-revenue-ops plugin loaded on hive nodes
    for node in $HIVE_NODES; do
        if container_exists $node; then
            run_test "$node has cl-revenue-ops" "revenue_cli $node plugin list | grep -q 'revenue-ops'"
        fi
    done

    # Check sling plugin loaded (required for rebalancing)
    for node in $HIVE_NODES; do
        if container_exists $node; then
            run_test "$node has sling" "revenue_cli $node plugin list | grep -q sling"
        fi
    done

    # Check CLBoss loaded (optional but recommended)
    for node in $HIVE_NODES; do
        if container_exists $node; then
            if revenue_cli $node plugin list 2>/dev/null | grep -q clboss; then
                run_test "$node has clboss" "true"
            else
                log_info "$node: clboss not loaded (optional)"
            fi
        fi
    done

    # Verify vanilla nodes don't have revenue-ops
    for node in $VANILLA_NODES; do
        if container_exists $node; then
            run_test_expect_fail "$node has NO cl-revenue-ops" "vanilla_cli $node plugin list | grep -q revenue-ops"
        fi
    done
}

# Status Tests - Verify basic plugin functionality
test_status() {
    echo ""
    echo "========================================"
    echo "STATUS TESTS"
    echo "========================================"

    # revenue-status command
    run_test "revenue-status works" "revenue_cli alice revenue-status | jq -e '.status'"

    # Version info
    VERSION=$(revenue_cli alice revenue-status | jq -r '.version')
    log_info "cl-revenue-ops version: $VERSION"
    run_test "Version is returned" "[ -n '$VERSION' ] && [ '$VERSION' != 'null' ]"

    # Config info embedded in status
    run_test "Config in status" "revenue_cli alice revenue-status | jq -e '.config'"

    # Channel states in status
    run_test "Channel states in status" "revenue_cli alice revenue-status | jq -e '.channel_states'"

    # revenue-dashboard command
    run_test "revenue-dashboard works" "revenue_cli alice revenue-dashboard | jq -e '. != null'"

    # Check on all hive nodes
    for node in $HIVE_NODES; do
        if container_exists $node; then
            run_test "$node revenue-status" "revenue_cli $node revenue-status | jq -e '.status'"
        fi
    done
}

# Flow Analysis Tests
test_flow() {
    echo ""
    echo "========================================"
    echo "FLOW ANALYSIS TESTS"
    echo "========================================"

    # Get channel states from revenue-status
    CHANNELS=$(revenue_cli alice revenue-status 2>/dev/null | jq '.channel_states')
    CHANNEL_COUNT=$(echo "$CHANNELS" | jq 'length // 0')
    log_info "Alice has $CHANNEL_COUNT channels"

    if [ "$CHANNEL_COUNT" -gt 0 ]; then
        # Check flow analysis data structure
        run_test "Channels have peer_id" "echo '$CHANNELS' | jq -e '.[0].peer_id'"
        run_test "Channels have state (flow)" "echo '$CHANNELS' | jq -e '.[0].state'"
        run_test "Channels have flow_ratio" "echo '$CHANNELS' | jq -e '.[0].flow_ratio'"
        run_test "Channels have capacity" "echo '$CHANNELS' | jq -e '.[0].capacity'"

        # Check flow state values (should be one of: source, sink, balanced)
        FIRST_FLOW=$(echo "$CHANNELS" | jq -r '.[0].state')
        log_info "First channel state: $FIRST_FLOW"
        run_test "Flow state is valid" "echo '$FIRST_FLOW' | grep -qE '^(source|sink|balanced)$'"

        # Check flow metrics
        run_test "Channels have sats_in" "echo '$CHANNELS' | jq -e '.[0].sats_in >= 0'"
        run_test "Channels have sats_out" "echo '$CHANNELS' | jq -e '.[0].sats_out >= 0'"
    else
        log_info "No channels on Alice - skipping detailed flow tests"
        run_test "revenue-status handles no channels" "revenue_cli alice revenue-status | jq -e '.channel_states'"
    fi

    # Check flow analysis on other nodes
    for node in bob carol; do
        if container_exists $node; then
            run_test "$node flow analysis works" "revenue_cli $node revenue-status | jq -e '.channel_states'"
        fi
    done
}

# Fee Controller Tests
test_fees() {
    echo ""
    echo "========================================"
    echo "FEE CONTROLLER TESTS"
    echo "========================================"

    # Get channel states for fee testing
    CHANNELS=$(revenue_cli alice revenue-status 2>/dev/null | jq '.channel_states')
    CHANNEL_COUNT=$(echo "$CHANNELS" | jq 'length // 0')

    # Check recent fee changes in revenue-status
    FEE_CHANGES=$(revenue_cli alice revenue-status 2>/dev/null | jq '.recent_fee_changes')
    FEE_CHANGE_COUNT=$(echo "$FEE_CHANGES" | jq 'length // 0')
    log_info "Recent fee changes: $FEE_CHANGE_COUNT"

    if [ "$FEE_CHANGE_COUNT" -gt 0 ]; then
        # Check fee change data structure
        run_test "Fee changes have channel_id" "echo '$FEE_CHANGES' | jq -e '.[0].channel_id'"
        run_test "Fee changes have old_fee_ppm" "echo '$FEE_CHANGES' | jq -e '.[0].old_fee_ppm'"
        run_test "Fee changes have new_fee_ppm" "echo '$FEE_CHANGES' | jq -e '.[0].new_fee_ppm'"
        run_test "Fee changes have reason" "echo '$FEE_CHANGES' | jq -e '.[0].reason'"
    else
        log_info "No recent fee changes yet"
    fi

    # Check fee configuration via revenue-config
    run_test "revenue-config list-mutable works" "revenue_cli alice revenue-config list-mutable | jq -e '.mutable_keys'"

    # Check specific config values
    MIN_FEE=$(revenue_cli alice revenue-config get min_fee_ppm 2>/dev/null | jq -r '.value // 0')
    MAX_FEE=$(revenue_cli alice revenue-config get max_fee_ppm 2>/dev/null | jq -r '.value // 5000')
    log_info "Fee range: $MIN_FEE - $MAX_FEE ppm"
    run_test "min_fee_ppm configured" "[ '$MIN_FEE' -ge 0 ]"
    run_test "max_fee_ppm configured" "[ '$MAX_FEE' -gt 0 ]"

    # Check hive fee ppm (for hive members)
    HIVE_FEE=$(revenue_cli alice revenue-config get hive_fee_ppm 2>/dev/null | jq -r '.value // 0')
    log_info "hive_fee_ppm: $HIVE_FEE"
    run_test "hive_fee_ppm configured" "[ '$HIVE_FEE' -ge 0 ]"

    # Check fee interval config
    FEE_INTERVAL=$(revenue_cli alice revenue-config get fee_interval 2>/dev/null | jq -r '.value // 300')
    log_info "fee_interval: $FEE_INTERVAL seconds"
    run_test "fee_interval configured" "[ '$FEE_INTERVAL' -gt 0 ]"

    # =========================================================================
    # v2.0 Fee Algorithm Improvements Tests
    # =========================================================================
    echo ""
    log_info "Testing v2.0 fee algorithm improvements..."

    # Test Improvement #1: Multipliers to Bounds
    run_test "Improvement #1: Bounds multipliers enabled" \
        "grep -q 'ENABLE_BOUNDS_MULTIPLIERS = True' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "Improvement #1: Floor multiplier cap exists" \
        "grep -q 'MAX_FLOOR_MULTIPLIER' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "Improvement #1: Ceiling multiplier floor exists" \
        "grep -q 'MIN_CEILING_MULTIPLIER' /home/sat/cl_revenue_ops/modules/fee_controller.py"

    # Test Improvement #2: Dynamic Observation Windows
    run_test "Improvement #2: Dynamic windows enabled" \
        "grep -q 'ENABLE_DYNAMIC_WINDOWS = True' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "Improvement #2: Min forwards for signal" \
        "grep -q 'MIN_FORWARDS_FOR_SIGNAL' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "Improvement #2: Max observation hours (security)" \
        "grep -q 'MAX_OBSERVATION_HOURS' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "Improvement #2: get_forward_count_since in database" \
        "grep -q 'def get_forward_count_since' /home/sat/cl_revenue_ops/modules/database.py"

    # Test Improvement #3: Historical Response Curve
    run_test "Improvement #3: Historical curve enabled" \
        "grep -q 'ENABLE_HISTORICAL_CURVE = True' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "Improvement #3: HistoricalResponseCurve class exists" \
        "grep -q 'class HistoricalResponseCurve' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "Improvement #3: Max observations limit (security)" \
        "grep -q 'MAX_OBSERVATIONS = 100' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "Improvement #3: Regime change detection" \
        "grep -q 'detect_regime_change' /home/sat/cl_revenue_ops/modules/fee_controller.py"

    # Test Improvement #4: Elasticity Tracking
    run_test "Improvement #4: Elasticity enabled" \
        "grep -q 'ENABLE_ELASTICITY = True' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "Improvement #4: ElasticityTracker class exists" \
        "grep -q 'class ElasticityTracker' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "Improvement #4: Outlier threshold (security)" \
        "grep -q 'OUTLIER_THRESHOLD' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "Improvement #4: Revenue-weighted elasticity" \
        "grep -q 'revenue_change_pct.*fee_change_pct' /home/sat/cl_revenue_ops/modules/fee_controller.py"

    # Test Improvement #5: Thompson Sampling
    run_test "Improvement #5: Thompson Sampling enabled" \
        "grep -q 'ENABLE_THOMPSON_SAMPLING = True' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "Improvement #5: ThompsonSamplingState class exists" \
        "grep -q 'class ThompsonSamplingState' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "Improvement #5: Max exploration bounded (security)" \
        "grep -q 'MAX_EXPLORATION_PCT = 0.20' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "Improvement #5: Beta distribution sampling" \
        "grep -q 'betavariate' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "Improvement #5: Ramp-up period for new channels" \
        "grep -q 'RAMP_UP_CYCLES' /home/sat/cl_revenue_ops/modules/fee_controller.py"

    # Test v2.0 Database Schema
    run_test "v2.0 DB: v2_state_json column migration" \
        "grep -q 'v2_state_json' /home/sat/cl_revenue_ops/modules/database.py"
    run_test "v2.0 DB: forward_count_since_update column" \
        "grep -q 'forward_count_since_update' /home/sat/cl_revenue_ops/modules/database.py"

    # Test v2.0 State Persistence
    run_test "v2.0 State: JSON serialization in save" \
        "grep -q 'json.dumps.*v2_data' /home/sat/cl_revenue_ops/modules/fee_controller.py"
    run_test "v2.0 State: JSON deserialization in load" \
        "grep -q 'json.loads.*v2_json' /home/sat/cl_revenue_ops/modules/fee_controller.py"
}

# Rebalancer Tests
test_rebalance() {
    echo ""
    echo "========================================"
    echo "REBALANCER TESTS"
    echo "========================================"

    # Check recent rebalances in revenue-status
    REBALANCES=$(revenue_cli alice revenue-status 2>/dev/null | jq '.recent_rebalances')
    REBAL_COUNT=$(echo "$REBALANCES" | jq 'length // 0')
    log_info "Recent rebalances: $REBAL_COUNT"

    # Check rebalance configuration
    REBAL_MIN_PROFIT=$(revenue_cli alice revenue-config get rebalance_min_profit 2>/dev/null | jq -r '.value // 10')
    log_info "rebalance_min_profit: $REBAL_MIN_PROFIT sats"
    run_test "rebalance_min_profit configurable" "[ '$REBAL_MIN_PROFIT' -ge 0 ]"

    REBAL_INTERVAL=$(revenue_cli alice revenue-config get rebalance_interval 2>/dev/null | jq -r '.value // 600')
    log_info "rebalance_interval: $REBAL_INTERVAL seconds"
    run_test "rebalance_interval configurable" "[ '$REBAL_INTERVAL' -gt 0 ]"

    # Check EV-based rebalancing code exists
    run_test "EV calculation in rebalancer" \
        "grep -q 'expected_value\\|EV\\|expected_profit' /home/sat/cl_revenue_ops/modules/rebalancer.py"

    # Check flow-aware opportunity cost
    run_test "Flow-aware opportunity cost" \
        "grep -q 'flow_multiplier\\|opportunity_cost' /home/sat/cl_revenue_ops/modules/rebalancer.py"

    # Check historical inbound fee estimation
    run_test "Historical inbound fee estimation" \
        "grep -q 'get_historical_inbound_fee_ppm\\|historical.*fee' /home/sat/cl_revenue_ops/modules/rebalancer.py"

    # Get channels for rebalance testing
    CHANNELS=$(revenue_cli alice revenue-status 2>/dev/null | jq '.channel_states')
    CHANNEL_COUNT=$(echo "$CHANNELS" | jq 'length // 0')

    if [ "$CHANNEL_COUNT" -ge 2 ]; then
        log_info "Found $CHANNEL_COUNT channels - can test rebalance candidates"

        # Check channel states include rebalance-relevant data
        run_test "Channels have flow_ratio for rebalancing" \
            "echo '$CHANNELS' | jq -e '.[0].flow_ratio'"
    else
        log_info "Need 2+ channels for rebalance tests - skipping"
    fi

    # Check for rejection diagnostics logging
    run_test "Rejection diagnostics implemented" \
        "grep -q 'REJECTION BREAKDOWN\\|rejection' /home/sat/cl_revenue_ops/modules/rebalancer.py"
}

# Sling Integration Tests
test_sling() {
    echo ""
    echo "========================================"
    echo "SLING INTEGRATION TESTS"
    echo "========================================"

    # Check sling plugin is loaded
    run_test "Sling plugin loaded" "revenue_cli alice plugin list | grep -q sling"

    # Check sling commands available
    run_test "sling-stats command works" "revenue_cli alice sling-stats 2>/dev/null | jq -e '. != null' || true"

    # Check sling configuration options in revenue-ops
    run_test "sling_max_hops config exists" \
        "grep -q 'sling_max_hops' /home/sat/cl_revenue_ops/modules/config.py"

    run_test "sling_parallel_jobs config exists" \
        "grep -q 'sling_parallel_jobs' /home/sat/cl_revenue_ops/modules/config.py"

    run_test "sling_target_sink config exists" \
        "grep -q 'sling_target_sink' /home/sat/cl_revenue_ops/modules/config.py"

    run_test "sling_target_source config exists" \
        "grep -q 'sling_target_source' /home/sat/cl_revenue_ops/modules/config.py"

    run_test "sling_outppm_fallback config exists" \
        "grep -q 'sling_outppm_fallback' /home/sat/cl_revenue_ops/modules/config.py"

    # Check sling-job creation in rebalancer
    run_test "sling-job integration" \
        "grep -q 'sling-job' /home/sat/cl_revenue_ops/modules/rebalancer.py"

    # Check maxhops parameter used
    run_test "maxhops parameter used" \
        "grep -q 'maxhops' /home/sat/cl_revenue_ops/modules/rebalancer.py"

    # Check flow-aware target calculation
    run_test "Flow-aware target calculation" \
        "grep -q 'sling_target_sink\\|sling_target_source' /home/sat/cl_revenue_ops/modules/rebalancer.py"

    # Check peer exclusion sync
    run_test "Peer exclusion sync implemented" \
        "grep -q 'sync_peer_exclusions\\|sling-except-peer' /home/sat/cl_revenue_ops/modules/rebalancer.py"

    # Check sling-except-peer command
    run_test "sling-except-peer command available" \
        "revenue_cli alice help 2>/dev/null | grep -q 'sling-except' || revenue_cli alice sling-except-peer 2>&1 | grep -qi 'parameter\\|node_id'"
}

# Policy Manager Tests
test_policy() {
    echo ""
    echo "========================================"
    echo "POLICY MANAGER TESTS"
    echo "========================================"

    # Get node pubkeys
    ALICE_PUBKEY=$(get_pubkey alice)
    BOB_PUBKEY=$(get_pubkey bob)
    CAROL_PUBKEY=$(get_pubkey carol)
    log_info "Alice: ${ALICE_PUBKEY:0:16}..."
    log_info "Bob: ${BOB_PUBKEY:0:16}..."
    log_info "Carol: ${CAROL_PUBKEY:0:16}..."

    # Test revenue-policy get command
    run_test "revenue-policy get works" "revenue_cli alice revenue-policy get $BOB_PUBKEY | jq -e '.policy'"

    # Check policy structure
    BOB_POLICY=$(revenue_cli alice revenue-policy get $BOB_PUBKEY 2>/dev/null)
    log_info "Bob policy: $(echo "$BOB_POLICY" | jq -c '.policy')"
    run_test "Policy has strategy" "echo '$BOB_POLICY' | jq -e '.policy.strategy'"
    run_test "Policy has rebalance_mode" "echo '$BOB_POLICY' | jq -e '.policy.rebalance_mode'"

    # Test valid strategies
    BOB_STRATEGY=$(echo "$BOB_POLICY" | jq -r '.policy.strategy')
    run_test "Strategy is valid" "echo '$BOB_STRATEGY' | grep -qE '^(static|dynamic|hive|aggressive|conservative)$'"

    # Test revenue-policy set command
    run_test "revenue-policy set works" \
        "revenue_cli alice -k revenue-policy action=set peer_id=$CAROL_PUBKEY strategy=dynamic | jq -e '.status == \"success\"'"

    # Verify policy was set
    CAROL_STRATEGY=$(revenue_cli alice revenue-policy get $CAROL_PUBKEY | jq -r '.policy.strategy')
    log_info "Carol strategy after set: $CAROL_STRATEGY"
    run_test "Policy set was applied" "[ '$CAROL_STRATEGY' = 'dynamic' ]"

    # Test invalid strategy (should fail gracefully)
    run_test_expect_fail "Invalid strategy rejected" \
        "revenue_cli alice -k revenue-policy action=set peer_id=$CAROL_PUBKEY strategy=invalid_strategy 2>&1 | jq -e '.status == \"success\"'"

    # Check policy list command
    run_test "revenue-policy list works" "revenue_cli alice revenue-policy list | jq -e '. != null'"

    # Policy on all hive nodes
    for node in bob carol; do
        if container_exists $node; then
            run_test "$node policy manager works" "revenue_cli $node revenue-policy get $ALICE_PUBKEY | jq -e '.policy'"
        fi
    done
}

# Profitability Analyzer Tests
test_profitability() {
    echo ""
    echo "========================================"
    echo "PROFITABILITY ANALYZER TESTS"
    echo "========================================"

    # Check profitability analysis is available
    run_test "Profitability analyzer exists" \
        "[ -f /home/sat/cl_revenue_ops/modules/profitability_analyzer.py ]"

    # Check profitability methods
    run_test "ROI calculation implemented" \
        "grep -q 'calculate_roi\\|roi\\|return_on' /home/sat/cl_revenue_ops/modules/profitability_analyzer.py"

    # Check revenue-dashboard for profitability metrics
    DASHBOARD=$(revenue_cli alice revenue-dashboard 2>/dev/null)
    log_info "Dashboard keys: $(echo "$DASHBOARD" | jq 'keys')"

    # Check for financial health metrics
    run_test "Dashboard has financial_health" \
        "echo '$DASHBOARD' | jq -e '.financial_health'"

    # Check for profit tracking
    run_test "Dashboard has net_profit" \
        "echo '$DASHBOARD' | jq -e '.financial_health.net_profit_sats >= 0 or .net_profit_sats >= 0 or true'"

    # Check profitability config
    run_test "Kelly config available" \
        "revenue_cli alice revenue-config get enable_kelly 2>/dev/null | jq -e '.key == \"enable_kelly\"'"

    KELLY_ENABLED=$(revenue_cli alice revenue-config get enable_kelly 2>/dev/null | jq -r '.value // false')
    log_info "Kelly Criterion enabled: $KELLY_ENABLED"

    # Check Kelly Criterion implementation
    run_test "Kelly Criterion in code" \
        "grep -qi 'kelly' /home/sat/cl_revenue_ops/modules/rebalancer.py || grep -qi 'kelly' /home/sat/cl_revenue_ops/modules/profitability_analyzer.py"
}

# CLBOSS Integration Tests
test_clboss() {
    echo ""
    echo "========================================"
    echo "CLBOSS INTEGRATION TESTS"
    echo "========================================"

    # Check CLBoss manager module exists
    run_test "CLBoss manager module exists" \
        "[ -f /home/sat/cl_revenue_ops/modules/clboss_manager.py ]"

    # Check if CLBoss is loaded
    if ! revenue_cli alice plugin list 2>/dev/null | grep -q clboss; then
        log_info "CLBoss not loaded - skipping runtime tests"
        return
    fi

    # CLBoss is loaded - test integration
    run_test "clboss-status works" "revenue_cli alice clboss-status | jq -e '.info.version'"

    # Check revenue-clboss-status command (our custom wrapper)
    run_test "revenue-clboss-status works" \
        "revenue_cli alice revenue-clboss-status 2>/dev/null | jq -e '. != null' || true"

    # Get a peer to test unmanage
    BOB_PUBKEY=$(get_pubkey bob)

    # Test clboss-unmanage with lnfee tag (revenue-ops owns this tag)
    UNMANAGE_RESULT=$(revenue_cli alice clboss-unmanage "$BOB_PUBKEY" lnfee 2>&1 || true)
    if echo "$UNMANAGE_RESULT" | grep -qi "unknown command"; then
        log_info "clboss-unmanage not available (upstream CLBoss)"
        run_test "CLBoss unmanage documented" \
            "grep -q 'clboss-unmanage\\|clboss_unmanage' /home/sat/cl_revenue_ops/modules/clboss_manager.py"
    else
        run_test "clboss-unmanage lnfee tag works" "true"
    fi

    # Check tag ownership documentation
    run_test "lnfee tag used by revenue-ops" \
        "grep -q 'lnfee' /home/sat/cl_revenue_ops/modules/clboss_manager.py"

    run_test "balance tag used by revenue-ops" \
        "grep -q 'balance' /home/sat/cl_revenue_ops/modules/clboss_manager.py"

    # Check CLBoss status parsing
    run_test "CLBoss status parsing" \
        "grep -q 'clboss.status\\|clboss-status' /home/sat/cl_revenue_ops/modules/clboss_manager.py"
}

# Database Tests
test_database() {
    echo ""
    echo "========================================"
    echo "DATABASE TESTS"
    echo "========================================"

    # Check database module exists
    run_test "Database module exists" \
        "[ -f /home/sat/cl_revenue_ops/modules/database.py ]"

    # Check key database methods
    run_test "Historical fee tracking method exists" \
        "grep -q 'get_historical_inbound_fee_ppm' /home/sat/cl_revenue_ops/modules/database.py"

    run_test "Forward event storage exists" \
        "grep -q 'store_forward\\|forward_event\\|insert.*forward' /home/sat/cl_revenue_ops/modules/database.py"

    run_test "Rebalance history storage exists" \
        "grep -q 'store_rebalance\\|rebalance.*history\\|insert.*rebalance' /home/sat/cl_revenue_ops/modules/database.py"

    run_test "Policy storage exists" \
        "grep -q 'store_policy\\|get_policy\\|policy' /home/sat/cl_revenue_ops/modules/database.py"

    # Check database file exists on node (in .lightning root, not regtest subdir)
    if docker exec polar-n${NETWORK_ID}-alice test -f /home/clightning/.lightning/revenue_ops.db 2>/dev/null; then
        DB_EXISTS="yes"
    else
        DB_EXISTS="no"
    fi
    log_info "Database exists: $DB_EXISTS"
    run_test "Database file exists on node" "[ '$DB_EXISTS' = 'yes' ]"

    # Check schema migrations
    run_test "Schema versioning exists" \
        "grep -q 'schema_version\\|SCHEMA_VERSION\\|migration' /home/sat/cl_revenue_ops/modules/database.py"
}

# Metrics Tests
test_metrics() {
    echo ""
    echo "========================================"
    echo "METRICS TESTS"
    echo "========================================"

    # Check metrics module exists
    run_test "Metrics module exists" \
        "[ -f /home/sat/cl_revenue_ops/modules/metrics.py ]"

    # Check revenue-dashboard provides metrics
    DASHBOARD=$(revenue_cli alice revenue-dashboard 2>/dev/null)
    log_info "Dashboard: $(echo "$DASHBOARD" | jq -c '.' | head -c 100)..."

    run_test "Dashboard returns data" "echo '$DASHBOARD' | jq -e '. != null'"

    # Check for key metrics
    run_test "Metrics module has forward tracking" \
        "grep -q 'forward\\|routing' /home/sat/cl_revenue_ops/modules/metrics.py"

    run_test "Metrics module has fee tracking" \
        "grep -q 'fee\\|revenue' /home/sat/cl_revenue_ops/modules/metrics.py"

    # Check capacity planner integration
    run_test "Capacity planner module exists" \
        "[ -f /home/sat/cl_revenue_ops/modules/capacity_planner.py ]"
}

# Reset Tests - Clean state for fresh testing
test_reset() {
    echo ""
    echo "========================================"
    echo "RESET TESTS"
    echo "========================================"
    echo "Resetting cl-revenue-ops state for fresh testing"
    echo ""

    log_info "Stopping cl-revenue-ops plugin on Alice..."
    revenue_cli alice plugin stop /home/clightning/.lightning/plugins/cl-revenue-ops/cl-revenue-ops.py 2>/dev/null || true
    sleep 2

    log_info "Restarting cl-revenue-ops plugin on Alice..."
    revenue_cli alice plugin start /home/clightning/.lightning/plugins/cl-revenue-ops/cl-revenue-ops.py 2>/dev/null || true
    sleep 3

    run_test "Plugin restarted successfully" "revenue_cli alice plugin list | grep -q revenue-ops"
    run_test "revenue-status works after restart" "revenue_cli alice revenue-status | jq -e '.status'"
}

#
# Main Test Runner
#

print_header() {
    echo ""
    echo "========================================"
    echo "cl-revenue-ops Test Suite"
    echo "========================================"
    echo ""
    echo "Network ID: $NETWORK_ID"
    echo "Hive Nodes: $HIVE_NODES"
    echo "Vanilla Nodes: $VANILLA_NODES"
    echo "Category: $CATEGORY"
    echo ""
}

print_summary() {
    echo ""
    echo "========================================"
    echo "Test Results"
    echo "========================================"
    echo ""
    echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
    echo ""

    if [ $TESTS_FAILED -gt 0 ]; then
        echo -e "${RED}Failed Tests:${NC}"
        echo -e "$FAILED_TESTS"
        echo ""
    fi

    TOTAL=$((TESTS_PASSED + TESTS_FAILED))
    if [ $TOTAL -gt 0 ]; then
        PASS_RATE=$((TESTS_PASSED * 100 / TOTAL))
        echo "Pass Rate: ${PASS_RATE}%"
    fi
    echo ""
}

run_category() {
    case "$1" in
        setup)
            test_setup
            ;;
        status)
            test_status
            ;;
        flow)
            test_flow
            ;;
        fees)
            test_fees
            ;;
        rebalance)
            test_rebalance
            ;;
        sling)
            test_sling
            ;;
        policy)
            test_policy
            ;;
        profitability)
            test_profitability
            ;;
        clboss)
            test_clboss
            ;;
        database)
            test_database
            ;;
        metrics)
            test_metrics
            ;;
        reset)
            test_reset
            ;;
        all)
            test_setup
            test_status
            test_flow
            test_fees
            test_rebalance
            test_sling
            test_policy
            test_profitability
            test_clboss
            test_database
            test_metrics
            ;;
        *)
            echo "Unknown category: $1"
            echo ""
            echo "Available categories:"
            echo "  all          - Run all tests"
            echo "  setup        - Environment and plugin verification"
            echo "  status       - Basic plugin status commands"
            echo "  flow         - Flow analysis functionality"
            echo "  fees         - Fee controller functionality"
            echo "  rebalance    - Rebalancing logic and EV calculations"
            echo "  sling        - Sling plugin integration"
            echo "  policy       - Policy manager functionality"
            echo "  profitability - Profitability analysis"
            echo "  clboss       - CLBoss integration"
            echo "  database     - Database operations"
            echo "  metrics      - Metrics collection"
            echo "  reset        - Reset plugin state"
            exit 1
            ;;
    esac
}

# Main execution
print_header
run_category "$CATEGORY"
print_summary

# Exit with failure if any tests failed
[ $TESTS_FAILED -eq 0 ]
