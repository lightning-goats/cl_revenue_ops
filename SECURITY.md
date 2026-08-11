# Security Policy

`cl_revenue_ops` operates in a security-sensitive environment.

The plugin interacts with Core Lightning and may analyze or influence actions involving channel liquidity, routing fees, rebalancing, channel management, swaps, and other operations involving real Bitcoin.

Security issues should therefore be reported responsibly and **must not be disclosed through public GitHub issues before a fix is available**.

## Supported Versions

Security fixes are provided for the latest maintained version of `cl_revenue_ops`.

| Version                 | Supported   |
| ----------------------- | ----------- |
| `main` / latest release | ✅           |
| Older releases          | Best effort |
| Unmaintained forks      | ❌           |

Users should run the latest stable release whenever practical.

## Reporting a Vulnerability

**Do not open a public GitHub issue for suspected security vulnerabilities.**

Preferred reporting method:

1. Use GitHub's **Private Vulnerability Reporting** feature for this repository, if available.
2. If private vulnerability reporting is unavailable, contact the maintainers privately before publishing technical details.

A useful vulnerability report should include:

* A clear description of the vulnerability.
* The affected version or commit.
* The affected component or plugin module.
* Preconditions required to exploit the issue.
* Reproduction steps or a minimal proof of concept.
* Expected behavior versus observed behavior.
* Potential impact.
* Whether funds, credentials, node availability, or privacy may be affected.
* Any suggested mitigation or patch, if known.

Please avoid including real private keys, macaroon-equivalent credentials, RPC credentials, wallet secrets, seed material, `hsm_secret`, personally identifying information, or other production secrets in reports.

## Response Process

Maintainers will make a reasonable effort to:

1. Acknowledge receipt of a security report.
2. Reproduce and assess the reported issue.
3. Determine severity and affected versions.
4. Develop and test a mitigation or fix.
5. Coordinate disclosure with the reporter where appropriate.
6. Publish a security advisory when users need to take action.

Complex issues involving Core Lightning behavior, protocol interactions, third-party services, or unusual node configurations may require additional investigation.

Please allow maintainers a reasonable remediation period before public disclosure.

## Security Scope

Issues are considered particularly important when they could cause or contribute to:

### Loss of Funds

Examples include:

* Unauthorized Lightning or on-chain spending.
* Incorrect destination or amount construction.
* Repeated or duplicate financial actions.
* Failure of idempotency protections.
* Incorrect channel open or close execution.
* Unsafe rebalance behavior.
* Unsafe swap execution.
* Fee-policy behavior capable of causing substantial unintended economic loss.
* Bypassing configured spending or capital limits.
* Integer, unit, or denomination errors involving BTC, satoshis, millisatoshis, ppm, percentages, or fee rates.

### Authorization or Policy Bypass

Examples include:

* Bypassing configured safety policies.
* Bypassing intent validation or arbitration.
* Executing actions that should have remained advisory or suppressed.
* Circumventing budget, profitability, liquidity, concentration, or risk limits.
* Converting malformed or untrusted data into executable actions.
* Executing an action without the expected authorization path.

### Intent and Idempotency Failures

Where actions are represented as typed or idempotent intents, vulnerabilities include:

* Replaying an already-executed intent.
* Executing the same logical operation multiple times.
* Intent-ID collisions.
* Mutating an intent after authorization.
* Executing an expired or stale intent.
* Confusing one intent type for another.
* Incorrect serialization or deserialization that changes financial meaning.
* Failure to bind execution to the intended node, peer, channel, amount, or operation.

### Core Lightning RPC Security

Examples include:

* Command injection through RPC parameters.
* Unsafe construction of RPC calls.
* Unexpected invocation of privileged RPC methods.
* Authorization boundary violations.
* Malicious RPC responses causing unsafe financial behavior.
* Trusting unvalidated plugin or datastore data.

### Secret Exposure

Examples include accidental disclosure of:

* `hsm_secret`
* wallet seed material
* private keys
* RPC credentials
* API tokens
* authentication cookies
* access tokens
* swap-service credentials
* Nostr private keys
* database credentials
* backup credentials
* personally identifying node-operator information

Secrets should never be written to normal application logs.

### Unsafe External Input Handling

Potentially hostile input may originate from:

* Lightning peers.
* Gossip data.
* invoices.
* BOLT messages.
* Core Lightning RPC responses.
* plugin notifications.
* datastore contents.
* configuration files.
* environment variables.
* external APIs.
* swap providers.
* market-data providers.
* Nostr or other messaging systems.
* files imported for analysis or recovery.

Input crossing these boundaries should be treated as untrusted unless explicitly proven otherwise.

### Data Integrity

Security-relevant integrity failures include:

* Corruption of profitability history.
* Corruption of channel accounting.
* Incorrect attribution of revenue or cost.
* Stale data being presented as current.
* Mixing snapshots from different evaluation cycles.
* Inconsistent state being consumed by different policies.
* Incorrect persistence or restoration of policy state.
* Maliciously crafted persisted data changing executable decisions.

Financial actions should preferably be derived from a consistent, versioned view of node state.

### Concurrency and Race Conditions

Examples include:

* Two controllers acting on the same channel simultaneously.
* Duplicate actions caused by overlapping evaluation cycles.
* State changing between authorization and execution.
* Race conditions around budgets or capital limits.
* Rebalance, fee, swap, open, or close operations conflicting with one another.

Security-sensitive actions should be serialized, arbitrated, locked, or otherwise protected where concurrent execution could change their meaning.

### Denial of Service

Examples include:

* Crashing `lightningd`.
* Blocking the Core Lightning plugin event loop.
* Unbounded memory consumption.
* Unbounded database growth.
* Infinite or extreme retry behavior.
* Excessive RPC traffic.
* Pathological external input causing sustained CPU consumption.
* Repeated failed financial operations that degrade node availability.

### Dependency and Supply-Chain Vulnerabilities

Reports involving dependencies are in scope when they create a realistic attack path against `cl_revenue_ops`.

This includes:

* Compromised packages.
* Dependency confusion.
* Unsafe dependency updates.
* Malicious build or installation scripts.
* Insecure GitHub Actions permissions.
* Artifact tampering.
* Installation processes that execute untrusted code.

## Out of Scope

The following normally do not qualify as vulnerabilities in `cl_revenue_ops`:

* Ordinary routing losses inherent to operating a Lightning node.
* Expected fee-market volatility.
* Poor profitability resulting from otherwise correct policy decisions.
* Counterparty behavior that the Lightning protocol permits.
* Previously documented operational risks.
* Attacks requiring an already fully compromised host or unrestricted root access, unless the vulnerability materially expands that compromise.
* Vulnerabilities exclusively affecting unsupported versions.
* Social-engineering attacks against maintainers without a software component.
* Generic dependency CVEs with no demonstrated impact on this project.
* Purely theoretical attacks without a plausible execution path.
* Reports generated solely by automated scanners without validation.

Unexpected financial behavior may nevertheless indicate a security issue and should be reported when the behavior violates an explicit safety boundary.

## Security Model

`cl_revenue_ops` should be considered a **financial control system**, not merely an analytics application.

The security model should assume:

* The Core Lightning node controls real funds.
* Input data may be stale, malformed, adversarial, or incomplete.
* External services may fail or return incorrect information.
* Network peers are untrusted.
* APIs may be unavailable or compromised.
* Processes may restart at arbitrary points.
* An operation may succeed even when the caller does not receive confirmation.
* Duplicate execution is dangerous.
* Configuration errors are possible.
* Policy algorithms can be logically incorrect even when the software behaves exactly as implemented.

Consequently, executable actions should follow a model similar to:

**observe → normalize → evaluate → construct typed intent → validate → arbitrate → authorize → execute → verify → record**

Where practical, financial decisions should be separated from the mechanisms that execute them.

## Fail-Safe Behavior

When state is ambiguous, inconsistent, stale, malformed, or unavailable, the preferred behavior is generally:

**do not perform a new capital-moving action.**

The plugin should fail closed for operations capable of materially moving or locking funds.

Examples include refusing execution when:

* Required channel information is missing.
* A snapshot is stale.
* Amount validation fails.
* Policy evaluation fails.
* Intent validation fails.
* Budget information is unavailable.
* An operation conflicts with another active intent.
* The expected node or channel identity cannot be verified.

Read-only analysis may continue when safe to do so.

## Financial Units

Lightning software frequently handles several easily confused units.

Security-sensitive code should make units explicit, including:

* BTC
* sat
* msat
* ppm
* basis points
* percentages
* fiat-denominated reference values

Avoid passing ambiguous bare integers between security-sensitive components.

Conversions should be centralized and tested, particularly:

* sat ↔ msat
* BTC ↔ sat
* ppm fee calculations
* percentage calculations
* maximum-spend calculations

Overflow, underflow, rounding, sign handling, and floating-point behavior should be considered part of the security boundary.

## Idempotency

Any operation that can spend, move, lock, or materially alter capital should be designed assuming that execution may be retried.

A network timeout does **not** prove that an operation failed.

Where applicable:

* Give every executable operation a stable unique identity.
* Persist execution state before or atomically with execution where possible.
* Detect duplicate requests.
* Reconcile ambiguous results with Core Lightning before retrying.
* Never blindly retry non-idempotent financial RPCs.
* Verify the actual resulting state after execution.

## Configuration Security

Production operators should:

* Run the plugin as the same minimally privileged account required for Core Lightning operation.
* Restrict filesystem permissions on configuration and database files.
* Avoid storing secrets directly in source-controlled configuration.
* Never commit production credentials.
* Protect `hsm_secret` independently of the plugin.
* Maintain tested backups appropriate for the underlying Core Lightning installation.
* Review configuration changes before deployment.
* Use conservative capital and spending limits.
* Introduce new executable features gradually.

Where available, new capital-affecting capabilities should initially operate in observation, shadow, dry-run, or advisory modes.

## Logging

Logs are useful for auditing but can themselves become a security risk.

Logs should contain enough information to reconstruct important decisions, such as:

* policy evaluated;
* intent type;
* intent identifier;
* affected channel or peer;
* authorization result;
* suppression reason;
* execution result.

Logs should **not** contain:

* private keys;
* seeds;
* `hsm_secret`;
* passwords;
* authentication tokens;
* complete sensitive credentials.

Sensitive external responses should be sanitized before logging.

## Database and Persistence Security

Persisted state used for financial decisions should be considered security-sensitive.

Implementations should account for:

* partial writes;
* process crashes;
* schema migrations;
* stale records;
* duplicated records;
* corrupted state;
* concurrent writers;
* restoration from backup.

Where practical, records representing executable actions should maintain enough history to determine:

* what was proposed;
* why it was authorized;
* whether it was executed;
* what Core Lightning reported;
* whether the resulting state was verified.

## External Services

Third-party information should generally be advisory unless independently validated.

A compromised or incorrect external service should not by itself be sufficient to cause unrestricted movement of funds.

External integrations should use:

* explicit timeouts;
* bounded retries;
* response validation;
* size limits where appropriate;
* TLS verification;
* conservative failure behavior.

The plugin should remain operationally safe when an external service becomes unavailable.

## Testing Expectations

Changes affecting capital-moving functionality should receive stronger testing than ordinary presentation or analytics changes.

Important areas include:

* boundary-value tests;
* malformed-input tests;
* unit-conversion tests;
* duplicate-execution tests;
* restart/recovery tests;
* concurrency tests;
* stale-state tests;
* permission and policy-bypass tests;
* maximum/minimum amount tests;
* external-service failure tests.

Where feasible, financial execution paths should be tested against regtest, signet, Polar, or another isolated Lightning environment before production deployment.

## Production Deployment

Operators should treat upgrades affecting execution logic conservatively.

Recommended practice includes:

1. Review the release diff.
2. Back up the relevant Core Lightning state.
3. Verify configuration.
4. Run the test suite.
5. Deploy in advisory, shadow, dry-run, or otherwise constrained mode when supported.
6. Observe decisions before enabling new autonomous actions.
7. Start with conservative budgets and capital limits.
8. Monitor logs and node state after activation.
9. Maintain a straightforward method to disable the plugin.

Production nodes holding substantial funds should not be the first environment in which new capital-moving behavior is exercised.

## Responsible Disclosure

We appreciate researchers who:

* Give maintainers an opportunity to remediate vulnerabilities before publication.
* Avoid accessing funds or data beyond what is necessary to demonstrate an issue.
* Avoid degrading production Lightning infrastructure.
* Use regtest, signet, test nodes, or controlled environments whenever possible.
* Provide clear reproduction information.
* Coordinate disclosure of vulnerabilities that could affect downstream users.

Researchers must not intentionally steal, lock, destroy, or place third-party Bitcoin at risk as part of vulnerability testing.

## Security Is a Process

No security policy guarantees that software controlling Bitcoin is safe.

Node operators remain responsible for:

* understanding the software they deploy;
* establishing appropriate capital limits;
* maintaining backups;
* protecting Core Lightning secrets;
* reviewing configuration;
* monitoring autonomous behavior;
* updating dependencies;
* applying security fixes.

If you discover behavior that could cause unauthorized fund movement, bypass a configured safety boundary, expose secrets, corrupt economically important state, or otherwise compromise a Lightning node, please report it privately.
