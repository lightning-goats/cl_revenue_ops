#!/bin/bash
#
# Install cl-revenue-ops and cl-hive plugins on Polar CLN nodes
#
# Usage: ./install.sh <network-id> [nodes]
# Example: ./install.sh 1
# Example: ./install.sh 1 "alice bob"
#

set -e

NETWORK_ID="${1:-1}"
NODES="${2:-alice bob carol}"
REVENUE_OPS_PATH="${REVENUE_OPS_PATH:-/home/sat/cl_revenue_ops}"
HIVE_PATH="${HIVE_PATH:-/home/sat/cl-hive}"

echo "Installing plugins on Polar network $NETWORK_ID"
echo "Nodes: $NODES"
echo "cl-revenue-ops: $REVENUE_OPS_PATH"
echo "cl-hive: $HIVE_PATH"
echo ""

for node in $NODES; do
    CONTAINER="polar-n${NETWORK_ID}-${node}"

    echo "=== Installing on $CONTAINER ==="

    # Check container exists
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
        echo "ERROR: Container $CONTAINER not found. Is Polar running?"
        exit 1
    fi

    # Install dependencies
    echo "  Installing dependencies..."
    docker exec -u root $CONTAINER apt-get update -qq
    docker exec -u root $CONTAINER apt-get install -y -qq python3 python3-pip git > /dev/null
    docker exec -u root $CONTAINER pip3 install -q pyln-client 2>/dev/null

    # Create plugins directory
    docker exec $CONTAINER mkdir -p /home/clightning/.lightning/plugins

    # Copy cl-revenue-ops
    echo "  Copying cl-revenue-ops..."
    docker cp "$REVENUE_OPS_PATH" $CONTAINER:/home/clightning/.lightning/plugins/cl-revenue-ops

    # Copy cl-hive
    echo "  Copying cl-hive..."
    docker cp "$HIVE_PATH" $CONTAINER:/home/clightning/.lightning/plugins/cl-hive

    # Set permissions
    echo "  Setting permissions..."
    docker exec -u root $CONTAINER chown -R clightning:clightning /home/clightning/.lightning/plugins
    docker exec $CONTAINER chmod +x /home/clightning/.lightning/plugins/cl-revenue-ops/cl-revenue-ops.py
    docker exec $CONTAINER chmod +x /home/clightning/.lightning/plugins/cl-hive/cl-hive.py

    # Load plugins
    echo "  Loading plugins..."
    if docker exec $CONTAINER lightning-cli plugin start /home/clightning/.lightning/plugins/cl-revenue-ops/cl-revenue-ops.py 2>/dev/null; then
        echo "    cl-revenue-ops loaded"
    else
        echo "    cl-revenue-ops failed to load (check logs)"
    fi

    if docker exec $CONTAINER lightning-cli plugin start /home/clightning/.lightning/plugins/cl-hive/cl-hive.py 2>/dev/null; then
        echo "    cl-hive loaded"
    else
        echo "    cl-hive failed to load (check logs)"
    fi

    echo "  Done with $node"
    echo ""
done

echo "=== Installation Complete ==="
echo ""
echo "Verify with:"
echo "  docker exec polar-n${NETWORK_ID}-alice lightning-cli plugin list | grep -E '(revenue|hive)'"
echo ""
echo "View logs with:"
echo "  docker exec polar-n${NETWORK_ID}-alice tail -50 /home/clightning/.lightning/regtest/log"
