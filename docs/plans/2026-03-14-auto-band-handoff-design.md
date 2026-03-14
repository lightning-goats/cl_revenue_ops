# Auto Band Handoff Design

## Goal

Keep manual dynamic fee bands as a fallback, but let a channel's learned auto band take control as soon as that channel has enough Thompson data to support one.

## Problem

Manual dynamic fee bands are stored per peer, while learned auto bands are stored per channel. The current precedence is `manual > auto > none`, which blocks channel-level auto bands whenever a peer-level manual band exists.

That creates the wrong handoff behavior:

- Channels with enough data cannot start using their learned band.
- Removing the manual peer policy would remove the fallback for every channel to that peer, including channels that still lack enough data.

## Design

Change effective autoband resolution to:

1. Learned channel auto band, when present and enabled
2. Manual peer policy band
3. No band

This preserves the operator's configured manual band as a fallback for channels that have not learned enough yet, while allowing ready channels to self-tune immediately.

## Scope

Modify only effective band resolution and operator/debug visibility. Do not change policy storage, add new schema fields, or mutate peer policies automatically.

## Expected Behavior

- If a channel has no learned auto band yet, the manual peer band still constrains dynamic pricing.
- Once the channel learns a valid auto band, that auto band becomes the effective band for that channel.
- The manual peer band remains in policy storage and continues to act as fallback for other channels to the same peer.
- If auto bands are disabled, manual peer bands continue to behave as before.
- Initial fee setting for a newly normal channel also uses the same effective precedence.

## Risks

- Operators expecting manual peer bands to permanently override channel learning will now see learned channel bands take over once available.

This is acceptable because it matches the intended "manual fallback until sufficient data exists" behavior.

## Testing

Add regression coverage for:

- effective precedence switching from manual to auto
- channels without learned auto band still using manual fallback
- `_adjust_channel_fee()` clamping to auto band when both manual and auto exist
- `set_initial_fee()` clamping to auto band when both manual and auto exist
