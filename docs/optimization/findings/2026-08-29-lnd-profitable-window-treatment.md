# LND profitable-window treatment

## Question

Could Revenue Ops convert a locally proven profitable LND quote into more net
profit than uncapped CLBOSS, or was CLBOSS's historical volume lead evidence
that Revenue Ops should lower ordinary fee floors?

## Fixture correction

The first crossed pair exposed a protocol mismatch, not a product failure. The
15-second fee and rebalance cadences were accelerated, but Revenue Ops' active
fee profile still requires either three settled forwards or 0.25 hours before
closing an observation window. A single 500,000-sat earning counterflow could
therefore leave both contenders at the equal 800-ppm seed throughout the
120-second observation.

The corrected fixture preserves exactly 500,000 sats of counterflow and the
same 800-ppm quote for each contender, but settles it as 166,667, 166,667, and
166,666 sats. This satisfies the existing three-forward safety threshold; it
does not add a tournament-only fee profile or force a Revenue fee cycle. For
LND, the runner also applies the same truthful admission rule to both paused
contenders: advertise the real post-fixture spendable balance, never more, and
require exact LND gossip readback before paying. Missing or malformed policy
evidence fails closed.

## Crossed result

Replicas 108 and 109 crossed Revenue Ops across both identities. In each:

- Revenue Ops autonomously changed the earning lane from 800 to 760 ppm;
- Revenue Ops delivered about 49,900 sats of native refill for about 7.54 sats;
- uncapped CLBOSS retained 800 ppm and delivered no refill;
- all eight realistic LND payments settled with no fallback, unattributed
  volume, forced controller cycle, or safety violation;
- Revenue Ops captured seven forwards and 210,000,000 msat at 159,600 msat
  gross, while CLBOSS captured one and 100,000,000 msat at 80,000 msat gross.

Across the crossed pair, Revenue Ops earned 304,123 msat linked net after
15,077 msat of refill cost, versus CLBOSS's 160,000 msat: a 1.901x net-profit
win. Revenue also carried 420,000,000 versus 200,000,000 msat, but that 2.10x
volume result is supporting evidence rather than the objective.

## Decision

Do not lower ordinary fee floors to chase CLBOSS volume. Preserve realistic
fee levels and let sufficient settled evidence close the ordinary observation
window. A bounded 5% downshift from a proven profitable quote produced both
more profit and more volume across identities. Continue scoring risk-adjusted
net profit first; treat raw volume and forward count as diagnostics unless the
incremental traffic remains profitable after rebalance and capital costs.
