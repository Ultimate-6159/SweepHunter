"""ดูว่า filter ตัวไหนเปิด/ปิดอยู่ตอนนี้"""
import json, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

c = json.load(open("config.json", encoding="utf-8"))

def status(enabled, name, detail=""):
    em = "🟢 ON " if enabled else "🔴 OFF"
    print(f"  {em}  {name:<35} {detail}")

print("="*80)
print("🔍 ACTIVE FILTERS (entry-blocking layers)")
print("="*80)

h = c.get("hyper_frequency", {})
rf = c.get("risk_filters", {})
rgm = c.get("regime_filter", {})

# Confidence
print("\n🎯 AI Confidence:")
mc = h.get("min_confidence", 0)
status(True, "min_confidence", f"≥ {mc}")
dt = h.get("directional_threshold", {})
if dt.get("enabled"):
    bm = dt.get("buy_max", "∞"); sm = dt.get("sell_max", "∞")
    status(True, "directional_threshold", f"BUY [{dt.get('buy')}-{bm}]  SELL [{dt.get('sell')}-{sm}]")
else:
    status(False, "directional_threshold")

# Trend & momentum
print("\n📈 Trend / Momentum:")
status(h.get("trend_filter_enabled", True), "trend_filter (EMA dist)",
       f"min_dist={h.get('trend_min_ema_dist_atr')} ATR")
ex = h.get("exhaustion_filter", {})
status(ex.get("enabled", True), "exhaustion_filter (FOMO guard)",
       f"max_vel={ex.get('max_velocity_5_atr')} ATR")
tc = h.get("tick_confirmation", {})
status(tc.get("enabled", False), "tick_confirmation")

# Risk filters
print("\n🛡️  Risk Filters:")
ats = rf.get("atr_spike", {})
status(ats.get("enabled"), "atr_spike (volatile guard)",
       f"max ratio {ats.get('max_atr_ratio')}×")
df = rf.get("direction_flip", {})
status(df.get("enabled", False), "direction_flip (block side after losses)",
       f"after {df.get('min_consec_same_dir_losses')} L, {df.get('cooldown_minutes')}min")
sc = rf.get("series_loss_cap", {})
status(sc.get("enabled"), "series_loss_cap",
       f"{sc.get('max_loss_pct_of_balance')}% / {sc.get('action')}")

# Regime
print("\n🌊 Regime Filter:")
status(rgm.get("enabled"), "regime_filter (master)")
if rgm.get("enabled"):
    htf = rgm.get("htf_trend", {})
    status(htf.get("enabled"), "  └─ htf_trend (block counter-trend)",
           f"strong≥{htf.get('strong_trend_threshold')}")
    dd = rgm.get("daily_drawdown", {})
    status(dd.get("enabled"), "  └─ daily_drawdown halt",
           f"≥{dd.get('max_dd_pct_of_balance')}% halt {dd.get('halt_hours_after_trip',4)}h")

# Recovery
print("\n💰 Recovery:")
rec = c.get("recovery", {})
status(rec.get("enabled"), "recovery escalation",
       f"max steps={rec.get('max_steps')}")

# News
print("\n📰 News:")
nf = c.get("news_filter", {})
status(nf.get("enabled", True), "news_filter")

# Spread
print("\n💧 Spread:")
sp = c.get("spread_guard", {})
status(True, "spread_guard", f"max {sp.get('max_spread_points')}p")

# Smart trailing
print("\n🔒 Smart Trailing:")
st = c.get("smart_trailing", {})
status(st.get("enabled", False), "smart_trailing",
       f"trigger {st.get('be_trigger_atr')} ATR")

# Cooldown
print("\n⏱️  Cooldown:")
cd = h.get("min_seconds_between_entries", 0)
status(cd > 0, "min_seconds_between_entries", f"{cd}s")

print("\n" + "="*80)
print("📌 รวม filter ที่ ACTIVE block AI signals จริงๆ:")
print("="*80)
active_filters = []
if h.get("min_confidence", 0) > 0: active_filters.append(f"min_conf {h['min_confidence']}")
if dt.get("enabled"): active_filters.append("directional_threshold (BUY/SELL ranges)")
if h.get("trend_filter_enabled", True): active_filters.append("EMA trend filter")
if ex.get("enabled", True): active_filters.append("exhaustion (FOMO)")
if ats.get("enabled"): active_filters.append("ATR spike")
if df.get("enabled", False): active_filters.append("direction_flip ⚠️")
if rgm.get("enabled"):
    if rgm.get("htf_trend", {}).get("enabled"): active_filters.append("HTF trend")
    if rgm.get("daily_drawdown", {}).get("enabled"): active_filters.append("daily DD")
print(f"  → {len(active_filters)} filters: {', '.join(active_filters)}")
