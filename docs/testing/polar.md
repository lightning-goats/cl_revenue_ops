# Polar Testing Guide for cl-revenue-ops and cl-hive

This guide covers installing and testing cl-revenue-ops, cl-hive, and dependencies on a Polar regtest environment.

## Prerequisites

- Polar installed with 3 CLN nodes (v25.12 recommended)
- Docker running
- Plugin repositories cloned locally

## Architecture

```
Node 1 (Alice)          Node 2 (Bob)           Node 3 (Carol)
├── cl-revenue-ops      ├── cl-revenue-ops     ├── cl-revenue-ops
├── cl-hive             ├── cl-hive            ├── cl-hive
└── sling               └── sling              └── sling
```

---

## Installation

### Option A: Quick Install Script

Use the provided installation script:

```bash
# Find your Polar network ID (usually 1, 2, etc.)
ls ~/.polar/networks/

# Run installer (replace 1 with your network ID)
./install.sh 1
```

### Option B: Manual Installation

#### Step 1: Identify Container Names

```bash
docker ps --filter "ancestor=polarlightning/clightning" --format "{{.Names}}"
```

Typical names: `polar-n1-alice`, `polar-n1-bob`, `polar-n1-carol`

#### Step 2: Install Dependencies

```bash
CONTAINER="polar-n1-alice"

docker exec -u root $CONTAINER apt-get update
docker exec -u root $CONTAINER apt-get install -y python3 python3-pip git
docker exec -u root $CONTAINER pip3 install pyln-client
```

#### Step 3: Copy Plugins

```bash
docker cp /home/sat/cl_revenue_ops $CONTAINER:/home/clightning/.lightning/plugins/
docker cp /home/sat/cl-hive $CONTAINER:/home/clightning/.lightning/plugins/

docker exec -u root $CONTAINER chown -R clightning:clightning /home/clightning/.lightning/plugins
docker exec $CONTAINER chmod +x /home/clightning/.lightning/plugins/cl-revenue-ops/cl-revenue-ops.py
docker exec $CONTAINER chmod +x /home/clightning/.lightning/plugins/cl-hive/cl-hive.py
```

#### Step 4: Load Plugins

```bash
docker exec $CONTAINER lightning-cli plugin start /home/clightning/.lightning/plugins/cl-revenue-ops/cl-revenue-ops.py
docker exec $CONTAINER lightning-cli plugin start /home/clightning/.lightning/plugins/cl-hive/cl-hive.py
```

### Option C: Docker Volume Mount (Persistent)

Create `~/.polar/networks/<network-id>/docker-compose.override.yml`:

```yaml
version: '3'
services:
  alice:
    volumes:
      - /home/sat/cl_revenue_ops:/home/clightning/.lightning/plugins/cl-revenue-ops:ro
      - /home/sat/cl-hive:/home/clightning/.lightning/plugins/cl-hive:ro
  bob:
    volumes:
      - /home/sat/cl_revenue_ops:/home/clightning/.lightning/plugins/cl-revenue-ops:ro
      - /home/sat/cl-hive:/home/clightning/.lightning/plugins/cl-hive:ro
  carol:
    volumes:
      - /home/sat/cl_revenue_ops:/home/clightning/.lightning/plugins/cl-revenue-ops:ro
      - /home/sat/cl-hive:/home/clightning/.lightning/plugins/cl-hive:ro
```

Restart the network in Polar UI after creating this file.

---

## Configuration

### cl-revenue-ops (Testing Config)

```ini
revenue-ops-flow-interval=300
revenue-ops-fee-interval=120
revenue-ops-rebalance-interval=60
revenue-ops-min-fee-ppm=1
revenue-ops-max-fee-ppm=1000
revenue-ops-daily-budget-sats=10000
revenue-ops-clboss-enabled=false
```

### cl-hive (Testing Config)

```ini
hive-governance-mode=advisor
hive-probation-days=0
hive-min-vouch-count=1
hive-heartbeat-interval=60
```

---

## Testing

### Test 1: Verify Plugin Loading

```bash
for node in alice bob carol; do
    echo "=== $node ==="
    docker exec polar-n1-$node lightning-cli plugin list | grep -E "(revenue|hive)"
done
```

### Test 2: cl-revenue-ops Status

```bash
docker exec polar-n1-alice lightning-cli revenue-status
docker exec polar-n1-alice lightning-cli revenue-channels
docker exec polar-n1-alice lightning-cli revenue-dashboard
```

### Test 3: Hive Genesis

```bash
# Alice creates a Hive
docker exec polar-n1-alice lightning-cli hive-genesis

# Verify
docker exec polar-n1-alice lightning-cli hive-status
```

### Test 4: Hive Join

```bash
# Alice generates invite
TICKET=$(docker exec polar-n1-alice lightning-cli hive-invite | jq -r '.ticket')

# Bob joins
docker exec polar-n1-bob lightning-cli hive-join "$TICKET"

# Verify
docker exec polar-n1-bob lightning-cli hive-status
docker exec polar-n1-alice lightning-cli hive-members
```

### Test 5: State Sync

```bash
ALICE_HASH=$(docker exec polar-n1-alice lightning-cli hive-status | jq -r '.state_hash')
BOB_HASH=$(docker exec polar-n1-bob lightning-cli hive-status | jq -r '.state_hash')
echo "Alice: $ALICE_HASH"
echo "Bob: $BOB_HASH"
# Hashes should match
```

### Test 6: Fee Policy Integration

```bash
BOB_PUBKEY=$(docker exec polar-n1-bob lightning-cli getinfo | jq -r '.id')
docker exec polar-n1-alice lightning-cli revenue-policy get $BOB_PUBKEY
# Should show strategy: hive
```

### Test 7: Three-Node Hive

```bash
TICKET=$(docker exec polar-n1-alice lightning-cli hive-invite | jq -r '.ticket')
docker exec polar-n1-carol lightning-cli hive-join "$TICKET"
docker exec polar-n1-alice lightning-cli hive-members
# Should show 3 members
```

---

## Troubleshooting

### Plugin Fails to Load

```bash
# Check Python dependencies
docker exec polar-n1-alice pip3 list | grep pyln

# Check plugin permissions
docker exec polar-n1-alice ls -la /home/clightning/.lightning/plugins/
```

### View Plugin Logs

```bash
docker exec polar-n1-alice tail -100 /home/clightning/.lightning/regtest/log | grep -E "(revenue|hive)"
```

### Permission Issues

```bash
docker exec -u root polar-n1-alice chown -R clightning:clightning /home/clightning/.lightning/plugins
```

---

## Cleanup

### Stop Plugins

```bash
for node in alice bob carol; do
    docker exec polar-n1-$node lightning-cli plugin stop cl-revenue-ops || true
    docker exec polar-n1-$node lightning-cli plugin stop cl-hive || true
done
```

### Reset Databases

```bash
for node in alice bob carol; do
    docker exec polar-n1-$node rm -f /home/clightning/.lightning/regtest/revenue_ops.db
    docker exec polar-n1-$node rm -f /home/clightning/.lightning/regtest/cl_hive.db
done
```
