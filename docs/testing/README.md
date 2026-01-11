# cl-revenue-ops Testing

Automated test suite for the cl-revenue-ops plugin.

## Prerequisites

1. **Polar Network** running with CLN nodes (alice, bob, carol)
2. **Plugins installed** via cl-hive's install script:
   ```bash
   cd /home/sat/cl-hive/docs/testing
   ./install.sh <network_id>
   ```
3. **Funded channels** between nodes (for rebalance tests)

## Quick Start

```bash
# Run all tests
./test.sh all 1

# Run specific category
./test.sh flow 1
./test.sh rebalance 1
```

## Test Categories

| Category | Description |
|----------|-------------|
| `setup` | Environment and plugin verification |
| `status` | Basic plugin status commands |
| `flow` | Flow analysis functionality |
| `fees` | Fee controller functionality |
| `rebalance` | Rebalancing logic and EV calculations |
| `sling` | Sling plugin integration |
| `policy` | Policy manager functionality |
| `profitability` | Profitability analysis |
| `clboss` | CLBoss integration |
| `database` | Database operations |
| `metrics` | Metrics collection |
| `reset` | Reset plugin state |
| `all` | Run all tests |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NETWORK_ID` | `1` | Polar network ID |
| `HIVE_NODES` | `alice bob carol` | CLN nodes with cl-revenue-ops |
| `VANILLA_NODES` | `dave erin` | CLN nodes without plugins |

## Test Coverage

### Core Functionality
- Plugin loading and status
- Revenue channel analysis
- Dashboard metrics

### Flow Analysis
- Channel flow state detection (source/sink/balanced)
- Forward event tracking
- Balance monitoring

### Fee Controller
- Dynamic fee adjustment
- Fee range configuration (min/max PPM)
- Hive member fee policy (0 PPM)

### Rebalancer
- EV-based candidate selection
- Flow-aware opportunity cost
- Historical inbound fee estimation
- Rejection diagnostics

### Sling Integration
- sling-job creation with maxhops
- Flow-aware target calculation
- Peer exclusion synchronization
- outppm fallback configuration

### Policy Manager
- Per-peer strategy assignment
- Strategy validation (static/dynamic/hive)
- Rebalance mode configuration

### Profitability Analyzer
- ROI calculation
- Revenue tracking
- Cost tracking

### CLBoss Integration
- Status monitoring
- Tag management (lnfee, balance)
- unmanage/manage operations

### Database
- Forward event storage
- Rebalance history
- Policy persistence
- Schema versioning

## Running Tests

### Full Test Suite
```bash
./test.sh all 1
```

### Individual Categories
```bash
# Test sling integration
./test.sh sling 1

# Test rebalancer
./test.sh rebalance 1

# Test fee controller
./test.sh fees 1
```

### Reset Plugin State
```bash
./test.sh reset 1
```

## Integration with cl-hive Tests

The cl-revenue-ops tests complement the cl-hive test suite. For full integration testing:

```bash
# 1. Install plugins
cd /home/sat/cl-hive/docs/testing
./install.sh 1

# 2. Run cl-hive tests
./test.sh all 1

# 3. Run cl-revenue-ops tests
cd /home/sat/cl_revenue_ops/docs/testing
./test.sh all 1
```

## Troubleshooting

### Plugin Not Loaded
```bash
# Check plugin status
docker exec polar-n1-alice lightning-cli --network=regtest plugin list | grep revenue
```

### No Channels
Some tests require funded channels. Create channels in Polar:
1. Open Polar
2. Right-click nodes to create channels
3. Mine blocks to confirm

### Database Missing
```bash
# Check database file
docker exec polar-n1-alice ls -la /home/clightning/.lightning/regtest/revenue_ops.db
```

### CLBoss Not Available
CLBoss tests are optional. If not loaded, runtime tests are skipped and only code verification tests run.
