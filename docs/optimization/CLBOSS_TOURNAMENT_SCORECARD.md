# CLBOSS tournament scorecard

Coverage: 43 replicas, 71 blocks, 2525 attempted / 2522 settled payments. Enhanced strict-schema blocks: 54; safety-eligible: 41.

| Comparable area | Revenue Ops | CLBOSS | Current leader |
|---|---:|---:|---|
| Routing volume (msat) | 15992724010 | 36555803657 | clboss |
| Forward count | 701 | 1676 | clboss |
| Gross routing fees (msat) | 2029312 | 813276 | revenue_ops |
| Rebalance cost (msat) | 57732 | 61924 | revenue_ops |
| Net routing profit (msat) | 1971580 | 751352 | revenue_ops |
| Gross yield (ppm) | 126.890 | 22.248 | revenue_ops |
| Volume share (%) | 30.434 | 69.566 | clboss |
| Mean worst imbalance (ppm; lower is better) | 804182.2 | 823745.6 | revenue_ops |

Formal verdict: **not ready**. It requires at least three fresh replicas and six enhanced cold/warm blocks per league per replica.

This table describes observed lab outcomes; it does not treat historical smoke blocks as decisive evidence.

## Current functional comparison

| Comparable functional area | Revenue Ops evidence | CLBOSS evidence | Current result |
|---|---|---|---|
| Fee setting | Higher aggregate yield and profit; final-image post-refill pricing won linked net across both identities | Higher aggregate raw volume; natural low quotes still win untreated and acquisition/retention lanes | Split; Revenue wins controlled monetization, CLBOSS wins aggregate conversion breadth |
| Cold rebalance responsiveness | 50,000 sats delivered in every crossed 120-second cold window | No delivery in the same windows despite native 120/hour cadence | Revenue Ops |
| Warm rebalance delivery | Final build delivered 285,000 sats in each eligible warm epoch | Delivered 235,000 sats in the same epochs | Revenue Ops |
| Warm rebalance efficiency | 4,636 msat for 285,000 sats (16.27 ppm) | 4,085 msat for 235,000 sats (17.38 ppm) | Revenue Ops per delivered sat; CLBOSS lower absolute spend |
| Liquidity-to-demand conversion | Four final-image blocks won 1.14B/0.94B msat and 155,061/132,830 msat linked net | Lower final-image post-refill volume and linked net | Revenue Ops |
| Channel open/close management | Intentionally out of scope; no open/close caller | Disabled/unmanaged in the harness to keep shared scope | Not comparable |
| Budget/safety enforcement | Production budget enforced; all new controlled blocks safety-clean | Intentionally uncapped; all new controlled blocks safety-clean | Different policies; no safety regression |
| Aggregate net routing profit | 1,971,580 msat | 751,352 msat | Revenue Ops |

## Safety-eligible results by market profile

Only enhanced blocks with no fallback traffic and no block-level or
contender-level safety violations contribute here.

| Profile / phase / scope | Revenue volume (msat) | CLBOSS volume (msat) | Revenue net (msat) | CLBOSS net (msat) | Current result |
|---|---:|---:|---:|---:|---|
| `realistic` / replicas 47-48 crossed cold / both | 2330000000 | 3870000000 | 340545 | 3870 | Revenue wins profit/balance; CLBOSS wins volume |
| `realistic` / replica 44 warm pressure / both | 645000000 | 15000000 | 91915 | 15 | Revenue wins routes, volume, profit, and ending balance |
| `realistic` / replica 44 cold pressure / both | 1145000000 | 1955000000 | 163790 | 1955 | Revenue wins profit/balance; CLBOSS wins cold volume |
| `realistic` / 100-payment crossed cold / both | 1790000000 | 1970000000 | 253755 | 1970 | Near-parity routes; Revenue wins profit and ending balance |
| `acquisition` / native positive-base retention / CLN | 90000000 | 160000000 | 767 | 1590 | CLBOSS wins combined volume, routes, and profit across two paid blocks |
| `acquisition` / mixed acquisition-to-retention transition / CLN | 120000000 | 5000000 | 628 | 5 | Revenue wins, but the in-window phase transition makes this diagnostic only |
| `acquisition` / native paid retention / CLN | 45000000 | 80000000 | 275 | 632 | Forward routes tie 5-5; CLBOSS wins weighted volume and profit |
| `acquisition` / paid retention / LND | 255000000 | 195000000 | 2225 | 1419 | Revenue wins volume and profit |
| `acquisition` / paid retention / CLN | 10000000 | 115000000 | 60 | 943 | CLBOSS wins volume and profit |
| `realistic` / forward pressure / both | 180000000 | 1370000000 | 27000 | 1370 | Revenue wins profit; CLBOSS wins volume/balance |
| `realistic` / forward pressure / CLN | 5000000 | 325000000 | 730 | 325 | Revenue wins profit; CLBOSS wins volume/balance |
| `realistic` / forward pressure / LND | 0 | 330000000 | 0 | 330 | CLBOSS wins volume, profit, and balance |
| `realistic` / 50-ppm treatment / LND | 0 | 330000000 | 0 | 330 | CLBOSS wins; ordinary floor cut buys no volume |
| `realistic` / crossed post-refill / LND | 570000000 | 470000000 | 66291 | 56400 | Revenue wins volume and linked net profit |
| `realistic` / replica 74 fresh conversion + post-refill / LND | 285000000 | 235000000 | 38894 | 35250 | Revenue wins routes (12-3), volume, linked net profit, and exact 25%→30% refill |
| `realistic` / replica 75 crossed fresh conversion + post-refill / LND | 285000000 | 235000000 | 38894 | 35250 | Exact crossed-identity replication of replica 74 |
| `realistic` / final admission-refresh image, replicas 83-84 / LND | 1140000000 | 940000000 | 155061 | 132830 | Revenue wins volume and linked net across four clean blocks and both identities |
| `realistic` / crossed post-refill / CLN | 1140000000 | 940000000 | 155752 | 1645 | Revenue wins volume and linked net profit |
| `legacy_low_fee` / baseline / both | 5000000 | 445000000 | 75 | 4870 | CLBOSS wins volume and profit |

## Fee-market regimes

The tournament no longer treats the original 10-ppm startup policy as a
general market model. It now records one of two explicit profiles in every new
traffic block:

| Profile | Initial base / rate | Traffic amounts | Purpose |
|---|---:|---:|---|
| `acquisition` | 1 msat / 10 ppm | fixed 5k sat by default | Isolate low-price route acquisition and paid retention. |
| `realistic` | 500 msat / 150 ppm | deterministic 5k, 15k, 35k, 100k sat mix | Primary fee-setting, liquidity, and net-profit comparison. |

The realistic seed is a rounded, dated snapshot of the public announced graph,
not a claim that one fee fits every channel. On 2026-08-28, 1ML reported a
0.437-sat median base fee and a 150-ppm median rate; its 25th, 75th, and 95th
fee-rate percentiles were approximately 1, 633, and 2,863 ppm. Public graph
statistics omit private channels, so tournament conclusions must remain robust
across the full distribution rather than optimize to the median alone.

CLBOSS is intentionally not spend-capped in full-stack competition. It runs
its native xrebalancer with grant mode off and its fastest exposed rate of 120
attempts per hour. Revenue Ops retains its production budget enforcement; the
controlled rounds grant it a 1,000-sat rebalance allowance. The Polar-only
cadence is compressed to 15 seconds for Revenue Ops fee, flow, and rebalance
cycles. Production defaults remain unchanged.

## Controlled native rebalancing

These observations are separate from the aggregate traffic table because the
fixture payments create the starting liquidity state rather than scored route
demand. Both controllers start with matching selected-family lanes at
approximately 75% local on the source and 25% local on the destination, receive
equal 2M-sat neutral return paths, and resume simultaneously. No manual cycle
RPC is used.

| Replica / identity assignment | Revenue delivered / cost | CLBOSS delivered / cost | Safety | Result |
|---|---:|---:|---|---|
| 52 / Revenue B, CLBOSS A | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Revenue completed a profitable native refill; CLBOSS did nothing |
| 53 / Revenue A, CLBOSS B | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Crossed replication of the Revenue win |
| 54 / Revenue B, CLBOSS A | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Revenue refill converted into a post-refill volume win |
| 55 / Revenue A, CLBOSS B | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Crossed post-refill volume replication |
| 57 / Revenue A, CLBOSS B | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Fixed-image post-refill volume win |
| 58 / Revenue B, CLBOSS A | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Crossed fixed-image post-refill volume win |
| 60 / Revenue B, CLBOSS A | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Exact 90-sat evidence-band validation over 180 seconds |
| 61 / Revenue A, CLBOSS B | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Crossed exact-band validation |
| 62 / Revenue B, CLBOSS A / LND | 50,000 sats / 2.052 sats | 0 / 0 | no violations | Native LND-facing refill and post-refill win |
| 63 / Revenue A, CLBOSS B / LND | 50,000 sats / 2.052 sats | 0 / 0 | no violations | Crossed native LND-facing replication |
| 64 / Revenue B, CLBOSS A / repeated LND | first 50,000 sats / 2.052 sats; later 0 | later 155,000 sats / 1.311 sats | no violations | 10% emergency floor left a profitable 14.5% destination cooling while CLBOSS renewed |
| 65 / Revenue A, CLBOSS B / diagnostic | first 50,000 sats / 2.052 sats; later 0 | 0 | no violations | Discarded: plugin startup option still injected the old 10% default |
| 67 / Revenue A, CLBOSS B / equal pressure | 310,000 sats / 4.314 sats | 155,000 sats / 1.311 sats | no violations | 20% floor activated, but stale balance scheduled a duplicate Revenue refill |
| 68 / Revenue B, CLBOSS A / final equal pressure | 155,000 sats / 2.157 sats | 155,000 sats / 1.311 sats | no violations | Final image renewed once to exactly 30%; duplicate refill eliminated |
| 69 / Revenue A, CLBOSS B / diagnostic | 50,000 sats / 2.052 sats | 0 / 0 | no violations | Excluded from promotion: stale derived gossip cache hid the controlled market band |
| 70 / Revenue B, CLBOSS A / diagnostic | 50,000 sats / 2.052 sats | 0 / 0 | no violations | Excluded from promotion: the canonical 30-day snapshot lagged the settled fee window |
| 71 / Revenue A, CLBOSS B / diagnostic | 50,000 sats / 2.052 sats | 0 / 0 | no violations | Excluded from promotion: disconnected background nodes left 10-ppm policies stale in gossip and the 1-ppm return fixture polluted p25 |
| 72 / Revenue B, CLBOSS A / 120-ppm route | 0 / 0 | 0 / 0 | no violations | Revenue moved 120→118 ppm but correctly rejected a 9-sat route worth only 3–6 sats |
| 73 / Revenue A, CLBOSS B / 10-ppm route, 120-ppm market | 0 / 0 | 0 / 0 | no violations | Revenue repeated 120→118 ppm and rejected a borderline −0.025-sat planner score |
| 74 / Revenue B, CLBOSS A / 10-ppm route, 150-ppm market | 50,000 sats / 3.001 sats | 0 / 0 | no violations | Revenue moved 150→147 ppm and restored the earned destination from 25% to exactly 30%; uncapped CLBOSS did nothing |
| 75 / Revenue A, CLBOSS B / crossed 10-ppm route, 150-ppm market | 50,000 sats / 3.001 sats | 0 / 0 | no violations | Crossed replication: Revenue repeated 150→147 ppm, exact 25%→30% refill, and the post-refill demand win; uncapped CLBOSS again did nothing |
| 76 / Revenue B, CLBOSS A / warm 155k renewal | first 50,000 sats / 3.001 sats; warm 155,000 sats / 4.051 sats | warm 155,000 sats / 3.205 sats | no violations | Revenue renewed exactly once from 14.5%→30%, but CLBOSS matched the warm refill for 0.846 sats less; diagnostic equal-pressure epoch |
| 77 / Revenue A, CLBOSS B / pre-fix crossed warm | first 50,000 sats / 3.001 sats; warm 155,000 sats / 4.051 sats | warm 155,000 sats / 3.205 sats | no violations | Re-quote found no cheaper equally reliable route; reproduced the 0.846-sat cost gap |
| 78 / Revenue B, CLBOSS A / pre-fix crossed warm | first 50,000 sats / 3.001 sats; warm 155,000 sats / 4.051 sats | warm 155,000 sats / 3.205 sats | no violations | Exact crossed replication proved the cost gap was identity-independent |
| 79 / Revenue A, CLBOSS B / exact-msat fix | first 50,000 sats / 2.051 sats; warm 155,000 sats / 3.206 sats | warm 155,000 sats / 3.205 sats | no violations | Cold cost fell 31.7%; warm cost reached economic parity while retaining the responsiveness win |
| 80 / Revenue B, CLBOSS A / crossed exact-msat fix | first 50,000 sats / 2.051 sats; warm 155,000 sats / 3.206 sats | warm 155,000 sats / 3.205 sats | no violations | Exact crossed replication of the cold cost reduction and warm parity |
| 81 / Revenue A, CLBOSS B / cache diagnostic | first 50,000 sats / 2.051 sats; warm 285,000 sats / 4.636 sats | warm 235,000 sats / 4.085 sats | observations clean; second demand block excluded | Repeat demand exposed a stale 10M-msat admission ceiling after the warm refill |
| 82 / Revenue B, CLBOSS A / settlement-cache fix diagnostic | first 50,000 sats / 2.051 sats; warm 255,000 sats / 4.306 sats; later 285,000 sats / 4.636 sats | warm 230,000 sats / 4.030 sats; later 235,000 sats / 4.085 sats | observations clean; first demand block excluded | Cache invalidation restored the first warm ceiling, but a later fee-window wait reset it to 10M msat |
| 83 / Revenue A, CLBOSS B / final admission-refresh image | first 50,000 sats / 2.051 sats; warm 285,000 sats / 4.636 sats twice | warm 235,000 sats / 4.085 sats twice | no violations | Three clean demand epochs; live ceiling remained 242,839,900 msat through repeated same-process warm cycles |
| 84 / Revenue B, CLBOSS A / crossed final image | first 50,000 sats / 2.051 sats; warm 285,000 sats / 4.636 sats | warm 235,000 sats / 4.085 sats | observations clean; second demand block excluded | Crossed identity reproduced the cold/warm delivery and live 250M-msat admission ceiling; first demand block won linked net |

Across the ten clean cold observations in replicas 52-63, Revenue delivered
500,000 sats for 12.520
sats while uncapped CLBOSS delivered zero. The refill moved each selected
Revenue destination from below the 30% threshold into the operating band;
untouched lanes in the other client family mean the contender-wide
worst-imbalance metric remains unsuitable for this fixture.

Repeated warm demand then exposed a renewal gap hidden by the original
single-epoch fixture. In replica 64, settled demand drained Revenue's profitable
LND destination from 30% to 14.5%. The 24-hour cooldown still blocked it because
the emergency floor was 10% and anchor drift was only 15.5 points; uncapped
CLBOSS restored 155,000 sats in the corresponding warm epoch. Revision
`43e006b` raised the configurable emergency refill floor to 20% while preserving
the ordinary cooldown, value, EV, and budget gates. Replica 65 proved that the
CLN plugin option still overrode the dataclass with 10%; revision `12a2baf`
aligned the actual startup surface and added a default-parity regression test.

An equal-pressure, noncompetitive functional lane then sent the same realistic
5k/15k/35k/100k mix through each held controller. Replica 67 proved the 20%
override activated, but Revenue sent two 155,000-sat refills because the next
15-second cycle still saw the stale pre-settlement balance. Revision `f201b22`
adds a 60-second post-success grace that suppresses only the emergency shortcut.
Final-image replica 68 started Revenue/CLBOSS destinations at 14.5%/9.5%; each
controller autonomously restored exactly 155,000 sats. Revenue spent 2.157 sats,
CLBOSS spent 1.311 sats, Revenue ended exactly at its 30% floor, and no forced
cycle, fallback, reservation leak, or safety violation occurred. This closes
the duplicate-refill bug but leaves CLBOSS ahead on equal-refill cost.

The same warm rounds identify the next fee-setting target. In corrected crossed
replicas 66-67, CLBOSS's natural 120-ppm LND destination carried every offered
155,000-sat competitive epoch while Revenue's roughly 133-ppm destination
carried none. Revenue remains the aggregate net-profit and yield leader, but it
cannot yet claim decisive fee-setting or route-share superiority.

Replicas 69-74 close the two evidence-freshness defects that prevented a
bounded conversion response. Revision `9e7113f` permits a profitable earned
quote to retain a 10% edge below the fresh corridor p25. Revision `d479600`
bypasses the derived gossip cache only for that fully eligible earned window,
and `fdbecc4` performs one backoff-protected canonical profitability refresh
when settled fee evidence contradicts a real zero-forward 30-day snapshot.
Absent, malformed, negative, unearned, depleted, or exploring evidence remains
neutral and cannot authorize the quote.

The harness now reconnects isolated background CLN nodes before retuning and
requires exact directional gossip readback from both contenders. Its default
synthetic return policy is 500 msat / 120 ppm; explicit cheap-route scenarios
remain positive-fee and must pass the same crossed readback. This correction
made the fee result repeat in replicas 72-75: p25 was 120/120/150/150 ppm, the
profitability ceiling was 108/108/135/135 ppm, and Revenue applied bounded first
steps of 118/118/147/147 ppm. Replicas 74-75 then established the corresponding
profitability decision: Revenue spent 3.001 sats to restore 50,000 sats of a
150-ppm earning lane from 25% to 30%, while uncapped CLBOSS spent nothing and
left its lane at 25% in both identities. At the applied 147-ppm quote, one fully utilized refill
has a 7.35-sat gross fee opportunity and a 4.349-sat spread over the measured
route cost. After the synthetic paths were cooperatively closed and confirmed
absent, both demand blocks settled 15/15 without fallback and produced the exact
same split: Revenue won 12 routes and 285M msat versus CLBOSS's 3 routes and
235M msat. Per-replica linked net was 38.894 versus 35.250 sats after charging
Revenue's refill. The crossed end-to-end result is now replicated; longer warm
windows and the formal multi-block coverage threshold still prevent a decisive
all-area verdict.

Replica 76 extends the same image into a held equal-pressure renewal epoch.
After the first 50,000-sat Revenue refill, identical 155,000-sat demand placed
Revenue at 14.5% and CLBOSS at 9.5%. Both controllers then autonomously restored
exactly 155,000 sats once. Revenue paid 4.051 sats versus CLBOSS's 3.205 sats,
so CLBOSS retains a 0.846-sat execution-cost advantage on an equal refill.
Revenue's targeted demand earned 22.785 sats gross versus CLBOSS's 23.250 sats;
after charging the warm refill, CLBOSS therefore leads this noncompetitive
epoch by 1.311 sats net. This is evidence to improve route-cost selection, not
to loosen Revenue's EV or budget gates. The observation also caught a host
wall-clock step: the runner now reports interval duration from the monotonic
scheduler clock while preserving wall time only for block identity.

Replicas 77-78 tested revision `a456e3d`, which asks V3 once for a strictly
cheaper route at no lower estimated success probability before executing an
already selected refill. The extra quote returned the same 2,050-msat middle
path (or correctly failed its tighter bound), so the original route remained
unchanged and the 4,051/3,205-msat Revenue/CLBOSS gap repeated exactly across
identities. The negative result localized the deficit to route construction,
not alternate-path ranking.

The route decomposition exposed premature sat rounding: Revenue converted the
destination peer's exact 1,155-msat final-hop fee to 2,000 msat before building
the route. Revision `fe2c25c` preserves millisatoshi precision in both V2 and
V3 and rounds only at sat-denominated budget/reporting boundaries. Crossed
replicas 79-80 reduced the 50,000-sat cold refill from 3,001 to 2,051 msat and
the 155,000-sat warm refill from 4,051 to 3,206 msat. CLBOSS spent 3,205 msat
in each warm block. Thus Revenue retains its cold response lead and closes the
warm execution-cost gap to economic parity; the remaining 1 msat is not a
decisive difference. All four post-change observation blocks were autonomous,
single-shot, fully delivered where attempted, and safety-clean.

Repeated same-process demand in replica 81 exposed a second execution defect:
after Revenue refilled the destination, its advertised HTLC maximum remained at
10M msat, so larger customer payments bypassed the contender. Revision
`502e0ae` invalidates the local `listpeerchannels` and `listfunds` caches after
both synchronous and confirmed-late native settlements. Replica 82 proved that
this repaired the immediate post-settlement read, but also showed the ceiling
could regress on a later warm cycle because dynamic admission updates were
downstream of the fee controller's observation-window return.

Revision `f3d2b0e` separates admission refresh from fee learning. Sleeping or
waiting fee cycles may now update only `htlcmax`, preserving the exact current
base fee and ppm and leaving the fee-evidence cursor/window untouched. Missing,
malformed, deadband, or RPC-error inputs remain neutral. Final-image replica 83
then completed three safety-clean demand epochs in one process. Revenue carried
285M msat and earned 41,895 msat in each block versus CLBOSS's 235M and 35,250;
the linked three-block totals were 114,362/97,580 msat after rebalance costs.
Its live ceiling stayed at 242,839,900 msat after both warm refills. Crossed
replica 84 reproduced the 50,000-sat cold and 285,000/235,000-sat warm delivery,
advertised a live 250M-msat ceiling on identity B, and added a clean linked-net
win of 40,699/35,250 msat. Across the four eligible final-image blocks, Revenue
therefore won volume 1.14B/0.94B msat and linked net 155,061/132,830 msat. A
second replica-84 demand block had one fallback settlement and remains excluded.

The tournament runtime is now CLN v26.06.7. Because the release notes warn
that the initially published `elementsproject/lightningd:v26.06.7` image lacks
the fixes, the harness overlays the official Ubuntu 22.04 amd64 release
tarball, SHA256
`53ddf124fe7058b6a2fc059d104976cc54ba5be21dc55b295cd82d01cabeb39c`,
on the known v26.06.6 filesystem base. The checksum matched the published
manifest and all four maintainer signatures verified. The image build asserts
both `lightningd` and `lightning-cli` report v26.06.7, and runner preflight
fails closed on version, artifact-digest, or product-revision drift.

The four safety-eligible post-refill demand blocks (54, 55, 57, and 58) then
routed 1.14B msat through Revenue versus 0.94B through CLBOSS. Revenue's routing
fees were 159.960 sats versus 1.645 sats. Charging the four linked 1.052-sat
refills to this phase leaves Revenue at 155.752 sats net versus 1.645 sats for
CLBOSS. Every block settled 15/15 payments with no fallback, and the 285M/235M
volume split repeated exactly across both identities and both product images.

The crossed LND-facing replicas 62-63 use the same exact 120-ppm evidence band
and multipart fixture payments pinned to one outgoing contender and the same
contender as last hop. Revenue again refilled 50,000 sats while CLBOSS did
nothing. Their two post-refill LND blocks settled 30/30 without fallback or
safety violations and repeated the 285M/235M split across identities. Revenue
earned 70,395 msat gross and 66,291 msat after both linked refills, versus
56,400 msat for CLBOSS. Across all six eligible post-refill blocks, linked net
is therefore 222,043 msat for Revenue versus 58,045 msat for CLBOSS.

Revenue Ops' bounded acquisition experiment remains default-off and may quote
0 ppm on only one capped episode. It now admits competitor observations from
1 through 10 ppm instead of requiring exactly 1 ppm; all duration, volume,
opportunity-cost, liquidity, and cooldown rails remain unchanged. After 50,000
acquired sats it may run a one-hour, 250,000-sat paid validation phase at
0 ppm plus a positive base fee. New transitions charge no more than half the
competitor's proportional fee at the smallest acquired payment. If no positive
strict undercut exists, it exits. Both phases share the 25-sat opportunity-cost
cap and restore the exact captured base and proportional fees on exit.

## What the tournament has established

- Revenue Ops extracts more fee per routed sat, but CLBOSS wins far more routing volume. The main economic gap is conversion and retained demand, not fee arithmetic alone.
- Replicas 47-48 add a fresh crossed realistic repeat: Revenue earned 340,545 msat from 2.33B msat while CLBOSS earned 3,870 msat from 3.87B msat. Revenue finished materially better balanced in both runs (270,178 and 130,170 worst-imbalance ppm versus 970,000 for CLBOSS), with zero safety violations. CLBOSS still wins raw volume.
- Controlled replicas 52-53 establish a crossed native-rebalance win. Revenue completed one positive-EV 50,000-sat refill in each 90-second observation for 1.052 sats; uncapped CLBOSS completed none despite its 120/hour setting. Both runs were safety-clean. This is evidence for execution responsiveness and profitability discipline, not yet long-horizon profit superiority.
- Replicas 54-55 and fixed-image replicas 57-58 connect that refill to customer demand: Revenue won post-refill volume 285M to 235M msat in every run and linked net profit 155,752 to 1,645 msat across the four eligible blocks. The direct fixture paths were cooperatively closed and confirmed absent before scoring, so no payment bypassed the contenders.
- Crossed replicas 62-63 extend the same result to LND-facing liquidity. Revenue repeated the 50,000-sat native refill and exact 285M/235M post-refill volume win in both identities; aggregate linked net was 66,291 versus 56,400 msat with 30/30 settlements. This reverses the earlier unrepaired LND corridor loss under a causal liquidity fixture rather than a global fee-floor cut.
- Repeated warm replicas 64-68 changed the rebalance decision path twice: the default emergency floor is now 20% at both the dataclass and CLN option surfaces, and a 60-second settlement grace prevents stale-balance duplicate refills. Final-image replica 68 restored exactly 155,000 sats once, with clean safety and no forced cycle. CLBOSS still achieved the same refill for 0.846 sats less.
- Corrected crossed replicas 66-67 also show the current fee-setting loss directly: CLBOSS at 120 ppm won every offered LND demand payment against Revenue near 133 ppm. The next fee experiment must improve conversion without returning to globally unrealistic fee floors.
- Replicas 69-75 implement and validate that bounded fee experiment without lowering global floors. Fresh canonical ROI and fresh directional gossip are required; the profitable conversion ceiling repeated at 108/108/135/135 ppm in replicas 72-75, with bounded applied moves to 118/118/147/147 ppm and no safety violations.
- Replicas 72-75 also separate profit discipline from raw activity. Revenue rejected 9-sat and 4-sat routes when their complete modeled scores were negative, then executed the 150-ppm lane once the measured 3.001-sat cost cleared the opportunity gate. Crossed replicas 74-75 each moved the destination from 25% to exactly 30%; uncapped CLBOSS did nothing. With fixture paths absent, Revenue repeated the exact 12-3 route, 285M-235M msat, and 38.894-35.250 sats linked-net win across both identities.
- Replicas 76-80 close the observed warm route-cost gap. A bounded equally reliable re-quote did not find a cheaper path, while exact-msat final-hop construction reduced Revenue's cold/warm costs by 0.950/0.845 sats. Crossed final-image replicas retain Revenue's cold response win and reach warm economic parity (3.206 versus 3.205 sats) without changing refill size, cadence, EV gates, or safety.
- Replicas 81-84 close the observed admission-freshness gap. Settlement cache invalidation alone repaired only the first warm read; the final admission-only refresh keeps `htlcmax` synchronized outside fee-learning windows without changing fees or consuming evidence. Replica 83 repeated three clean demand epochs in one process, replica 84 crossed the identity, and four eligible final-image blocks won Revenue 1.14B/0.94B msat of volume and 155,061/132,830 msat of linked net profit. Aggregate raw volume and acquisition/retention remain CLBOSS advantages, so the all-area verdict is still not ready.
- The scorer now resolves every post-refill smoke block to its exact native observation, fails closed on missing or mismatched lineage, charges the linked rebalance cost, and publishes eligible single-family phase results. Historical aggregate profit no longer silently treats native refills as free.
- Replica 56 exposed an arbitrary early-channel capex cliff: a channel with four forwards, positive canonical contribution, and a profitable classification received zero budget because it had neither more than five forwards nor more than 100 sats contribution. Revision `4c26e11` now admits an early active tier funded only by the configured reinvestment share of realized 30-day contribution and capped by the existing bootstrap rail. Zero, absent, negative, malformed, and DB-degraded evidence still grants nothing.
- Replicas 60-61 validate `4c26e11` in the exact repaired band across identities. Equal 120-ppm fixture pricing produced approximately 90 sats of contribution, Revenue received 88 sats of combined allocation and completed the same 50,000-sat/1.052-sat refill, while CLBOSS completed none during each 180-second observation. Both observations were safety-clean.
- Diagnostic replicas 56, 59, and 60 demand blocks remain excluded: 56 and 60 each had one terminal failed payment, while 59 allowed a delayed CLBOSS circular payment to overlap scored traffic. The runner now freezes both controllers after the native observation and before retiring return paths, preventing later circular forwards from contaminating customer-demand attribution.
- The realistic 100-payment result now repeats across crossed identities. Replica 41 produced Revenue/CLBOSS volume of 1.79B/1.97B msat and fees of 253,755/1,970 msat; replica 44's eligible 80-payment cold plus 20-payment warm blocks produced the same 1.79B/1.97B volume and 255,705/1,970 fees. Combined, Revenue captured 92/200 routes, 90.9% of CLBOSS' volume, 129.3x its fees, and ended both crossed runs near 790k versus CLBOSS' 970k worst-imbalance ppm. Global fee cuts would sacrifice this replicated advantage; improvements must target missed lanes selectively.
- Replica 44's warm continuation exposed the sustainability difference: after the cold block, Revenue served 19/20 routes and 645M msat while CLBOSS served 1/20 and 15M. Revenue earned 91,915 versus 15 msat and remained less imbalanced. CLBOSS' xrebalancer was healthy but moved no liquidity.
- The crossed realistic CLN block repeated the profit result: Revenue earned 2.25x CLBOSS' fees from one of ten routes. On the LND corridor, however, Revenue won no routes at either 150 ppm or the 50-ppm safety floor while CLBOSS quoted 1 ppm. The 50-ppm cut therefore produced no conversion benefit.
- Native paid retention is mechanically verified but not a decisive strategy. In replica 39, Revenue Ops autonomously moved one CLN lane from 0 to the observed 1-ppm floor after exactly 50,000 acquired sats, persisted the phase, and restored its captured baseline during cleanup. The safety-eligible paid block split forward routes 5-5, while CLBOSS won weighted volume 80M to 45M msat and net fees 632 to 275 msat. Replica 40 then tested a strict positive-base undercut: Revenue moved from 0 to 4 msat + 0 ppm against 1 ppm, but across two paid blocks the treated lane split routes 8-12 and fees 32-110 msat; whole-contender profit was 767-1590 msat. The implementation and exact restoration are sound, but the economic hypothesis is not supported. Earlier manual retention won the LND block in replica 27 but lost the crossed CLN block in replica 34.
- A bounded 0-ppm acquisition quote can win a lane, but placement matters: observed lane share has ranged from 40% to 100% across client and peer identities.
- A 1-ppm tie did not acquire traffic in an earlier round. A zero-fee quote acquired 80% of the treated lane in replica 25 at an opportunity cost of 1.5 sats, then restored the captured 15-ppm baseline exactly.
- Autonomous rebalancing correctly refuses uneconomic routes below its contribution-margin hold. Product revision `9805b04` additionally prices one lower-ranked source fallback per selected destination, so an expensive first choice no longer suppresses a profitable alternative. It preserves the EV, budget, and one-pair-per-destination rails. Replicas 52-53 show that bounded fallback completing the same profitable route across crossed identities.
- Product revisions `3df9ad3` and `0aa7da8` fix the profitability-cache contradiction found in replicas 42 and 50. Any newly settled forward count can now trigger a backoff-protected canonical refresh, and Polar can genuinely run 15-second cycles instead of silently clamping them to 60 seconds. Only canonical profitability and capex output can create value or budget; production defaults are unchanged.
- After replica 44's warm block, Revenue selected and attempted a positive-EV 20,000-sat refill quoted at 2 sats with 98% estimated success. Four attempts failed on depleted lab return-path channels, spent zero, and left no reservation. The controlled fixture now supplies equal neutral return lanes after pressure and validates their balances before observation.
- Tournament preflight now pins the default image to the verified Revenue Ops revision and rejects a mismatched label before scored traffic; an unscored replica exposed the stale default tag.
- Replica 41's attempted warm replication stopped fail-closed after six settled payments when one CLN dispatch timed out and could not be reconciled immediately. A later read-only lookup found the invoice unpaid, but the partial block remains excluded rather than using hindsight to weaken the no-replay contract.
- Product revision `3df9ad3` and harness revision `5325003` add canonical profitability refresh on settled-flow contradiction, hardened timeout reconciliation, and the pinned contender image. Full tests passed before replicas 43-44 (3878 passed, 5 skipped, 2 xfailed).

## Active improvement loop

| Step | Evidence sought | Implementation or decision gate |
|---|---|---|
| Family attribution | CLN versus LND volume, fees, and forwards | Runner blocks map every contender SCID to a client family and fail closed on unmapped activity. |
| Automatic acquisition | Whether the default-off product selects and wins a natural lane | Enable the gate and wait for native fee cycles; never force a scored fee cycle. |
| Paid retention | Whether a positive base-fee undercut converts the 1-ppm tie into retained volume and profit | Keep experimental until positive lift repeats across crossed identities and both client families. |
| Retention curve | Whether any paid quote beats free acquisition on net profit under CLN route randomization | Measure multiple bounded price points with enough routes for confidence; optimize net contribution, not raw route count. |
| Liquidity pressure | Whether each controller restores depleted earning liquidity profitably | Run one-way traffic with equal spend caps; compare net fees, cost, and ending imbalance. |
| Controlled depletion | Whether each native controller repairs the same exact 75/25 liquidity state | Ten clean observations across CLN/LND and crossed identities; Revenue leads 10-0 while CLBOSS remains uncapped. |
| Reserved return lane | Whether a controller can complete a profitable circular refill after pressure | Equal post-pressure 2M-sat CLN/LND paths are removed and confirmed absent before demand scoring. |
| Post-refill demand | Whether repaired liquidity produces more routed volume and linked net profit | Six eligible CLN/LND blocks repeat Revenue's 285M/235M volume win; aggregate linked net is 222,043/58,045 msat. Extend to longer warm demand. |
| Warm renewal | Whether profitable outbound inventory renews inside the normal 24-hour cooldown without duplicate spend | Final-image replicas 83-84 deliver 285k versus 235k sats at 16.27/17.38 ppm cost efficiency; replica 83 repeats twice in one process. Extend to more replicas and both client families. |
| Admission freshness | Whether settled liquidity immediately becomes usable even while fee learning waits | Complete for the observed LND corridor: `502e0ae` invalidates settled-state caches and `f3d2b0e` refreshes only `htlcmax` outside fee windows; live crossed readbacks remained 242,839,900/250,000,000 msat. |
| Fee conversion | Whether Revenue can beat CLBOSS's earned quote without a global low-fee policy | Four final-image eligible LND blocks win 1.14B/0.94B volume and 155,061/132,830 linked net. Remaining target is untreated CLN/LND breadth plus acquisition/retention, where aggregate CLBOSS volume still leads. |
| Profit threshold | Whether Revenue spends only when the full refill economics clear opportunity cost | Replicas 72-73 hold negative 9-sat/4-sat routes; crossed replicas 74-75 each spend 3.001 sats on the clearly positive 150-ppm lane while uncapped CLBOSS remains idle. Extend the demand and renewal window. |
| CLN 26.06.7 compatibility | Whether both contenders and all read-only/action surfaces remain compatible with the 2026-08-28 security point release | Complete: signed official amd64 binary overlay, exact digest/version/revision preflight, full tests green, and crossed replicas 77-80 safety-clean. Do not use the release's warned-bad Docker image. |
| Evidence freshness | Whether settled forwards become canonical value/budget evidence before a 15-minute cache TTL expires | Implemented through `0aa7da8`; keep the analyzer refresh canonical, read-only, and backoff protected. |
| Product change | Repeatable positive net lift across crossed identities and clients | Admission refresh is promoted by crossed LND evidence; next bounded treatment must target aggregate route breadth or acquisition/retention and repeat across CLN and LND. |

Regenerate the aggregate observation separately before reconciling the narrative table:

```bash
.venv/bin/python tools/polar_clboss_scorecard.py --format markdown
```
