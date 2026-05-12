"""ดู schema + snapshot ปัจจุบัน เพื่อออกแบบระบบ data lineage"""
import sqlite3
DB = r'C:\SweepHunter - Copy\data\db\hyper_trades.sqlite'
out = open(r'C:\SweepHunter - Copy\_schema.txt', 'w', encoding='utf-8')
def w(*a): out.write(" ".join(str(x) for x in a)+"\n")
c = sqlite3.connect(DB); c.row_factory = sqlite3.Row

w("="*80); w("SCHEMA: config_snapshots"); w("="*80)
for r in c.execute("PRAGMA table_info(config_snapshots)").fetchall():
    w(f"  {r['name']:<30} {r['type']}")

w("\n" + "="*80); w("SCHEMA: decisions (relevant cols)"); w("="*80)
for r in c.execute("PRAGMA table_info(decisions)").fetchall():
    if r['name'] in ('id','ts_utc','config_snapshot_id','status','pnl','prediction','confidence','role'):
        w(f"  {r['name']:<30} {r['type']}")

w("\n" + "="*80); w("CURRENT SNAPSHOTS (full)"); w("="*80)
for r in c.execute("""
    SELECT id, ts_utc, label, config_hash, risk_per_trade_pct, sl_atr_mult, tp_atr_mult,
           smart_trailing_enabled, be_trigger_atr,
           series_loss_cap_pct, series_loss_cap_action, hourly_lot_mult_enabled
    FROM config_snapshots ORDER BY id
""").fetchall():
    w(f"\n  #{r['id']} {r['ts_utc'][:19]} hash={r['config_hash']}")
    w(f"     label: {r['label']}")
    w(f"     risk={r['risk_per_trade_pct']}% sl={r['sl_atr_mult']} tp={r['tp_atr_mult']} trail={r['smart_trailing_enabled']} trigger={r['be_trigger_atr']}")
    w(f"     loss_cap={r['series_loss_cap_pct']}% action={r['series_loss_cap_action']} hmul={r['hourly_lot_mult_enabled']}")
out.close(); print("done")
