"""Resource-growth regression: _boltz_balance_last_action must not grow unbounded.

_boltz_balance_last_action maps channel_id -> unix ts of the last Boltz balance
action, used purely as a per-channel cooldown gate. Before this fix nothing ever
removed entries, so a long-running node accumulated one entry per channel ever
swapped (including closed channels) for the process lifetime.

The fix adds a TTL-based prune. The TTL (30 days) dwarfs any realistic Boltz
balance cooldown (default 4h; hint overrides are hours), so pruning an entry can
never change a cooldown decision: any pruned entry is already far older than any
cooldown window and is therefore behaviorally equivalent to absent.
"""

import time

from tests.plugin_test_utils import load_plugin_module


class TestBoltzBalanceActionEviction:
    def test_prune_removes_stale_and_keeps_recent(self):
        mod = load_plugin_module()
        now = int(time.time())
        ttl = mod._BOLTZ_BALANCE_ACTION_TTL_SECONDS

        mod._boltz_balance_last_action.clear()
        mod._boltz_balance_last_action.update({
            "recent": now - 60,                 # 1 min old -> keep
            "just_over_ttl": now - ttl - 1,     # just past TTL -> evict
            "ancient": now - (400 * 86400),     # 400 days old (closed chan) -> evict
        })

        removed = mod._prune_boltz_balance_actions(now)

        assert removed == 2
        assert set(mod._boltz_balance_last_action.keys()) == {"recent"}
        assert mod._boltz_balance_last_action["recent"] == now - 60

    def test_ttl_dwarfs_realistic_cooldown(self):
        # Safety invariant: the eviction TTL must be far larger than any
        # plausible cooldown window, so eviction is provably non-behavioral.
        mod = load_plugin_module()
        # Default cooldown is 4h; even a generous 7-day override is < TTL.
        assert mod._BOLTZ_BALANCE_ACTION_TTL_SECONDS >= 7 * 86400
        assert mod._BOLTZ_BALANCE_ACTION_TTL_SECONDS > 4 * 3600 * 10

    def test_prune_on_empty_is_noop(self):
        mod = load_plugin_module()
        mod._boltz_balance_last_action.clear()
        assert mod._prune_boltz_balance_actions(int(time.time())) == 0
        assert mod._boltz_balance_last_action == {}
