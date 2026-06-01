# -*- coding: utf-8 -*-
"""วิเคราะห์ว่า recovery สำเร็จส่วนใหญ่ที่ step ไหน"""
import sqlite3
from collections import defaultdict
from pathlib import Path

DB = Path("data/db/hyper_trades.sqlite")

def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    # Series ที่ปิดด้วย win (มีอย่างน้อย 1 WIN และปิด series)
    series_rows = conn.execute("""
        SELECT s.id, s.status, s.final_pnl, s.steps, s.side,
               datetime(s.opened_at_utc, 'unixepoch') opened,
               datetime(s.closed_at_utc, 'unixepoch') closed
        FROM series s
        WHERE s.status NOT IN ('OPEN')
        ORDER BY s.id
    """).fetchall()

    decisions = conn.execute("""
        SELECT series_id, step, status, pnl, volume,
               CASE prediction WHEN 2 THEN 'BUY' WHEN 0 THEN 'SELL' ELSE '?' END side
        FROM decisions
        WHERE series_id IS NOT NULL AND status IN ('WIN', 'LOSS')
        ORDER BY series_id, step
    """).fetchall()

    by_series = defaultdict(list)
    for d in decisions:
        by_series[d["series_id"]].append(dict(d))

    # --- 1) Series ที่กู้สำเร็จ (จบด้วย WIN ไม้สุดท้าย) ---
    recovered = []  # win_step, losses_before, cum_loss_before_win, win_pnl, total_steps
    failed_max = []  # closed without full recovery
    all_multi = []  # series with 2+ trades

    for s in series_rows:
        sid = s["id"]
        trades = by_series.get(sid, [])
        if not trades:
            continue
        wins = [t for t in trades if t["status"] == "WIN"]
        losses = [t for t in trades if t["status"] == "LOSS"]
        if len(trades) < 2:
            continue  # skip single-trade series (not recovery path)

        all_multi.append(len(trades))

        # หาไม้ WIN แรกที่ทำให้ series กู้ได้ (มักเป็นไม้สุดท้ายที่ชนะ)
        # กรณี recovery: เสียหลายไม้แล้วชนะไม้หนึ่งปิด series
        last = trades[-1]
        if last["status"] == "WIN" and len(losses) >= 1:
            cum_loss = sum(abs(t["pnl"] or 0) for t in losses)
            win_pnl = last["pnl"] or 0
            net_recovery = win_pnl - cum_loss  # ถ้า > 0 = กู้ครบ+กำไร
            recovered.append({
                "sid": sid,
                "win_step": last["step"],
                "n_losses": len(losses),
                "n_trades": len(trades),
                "cum_loss": cum_loss,
                "win_pnl": win_pnl,
                "net": net_recovery,
                "full_cover": win_pnl >= cum_loss - 0.01,
                "volume_win": last["volume"],
                "status": s["status"],
            })
        elif len(losses) >= 1 and not wins:
            failed_max.append({"sid": sid, "n_losses": len(losses), "cum_loss": sum(abs(t["pnl"] or 0) for t in losses)})

    print("=" * 70)
    print("  สถิติ Recovery — กู้สำเร็จที่ step ไหน?")
    print("=" * 70)
    print(f"\n  Series หลายไม้ (≥2 trades): {len(all_multi)}")
    print(f"  Series กู้สำเร็จ (เสีย≥1 แล้ว WIN ไม้สุดท้าย): {len(recovered)}")
    print(f"  Series เสียหมดไม่มี WIN: {len(failed_max)}")

    if not recovered:
        print("\n  ไม่มีข้อมูล recovery ที่ชนะปิด series")
        conn.close()
        return

    # Distribution by win step
    by_step = defaultdict(list)
    for r in recovered:
        by_step[r["win_step"]].append(r)

    print("\n" + "-" * 70)
    print("  ไม้ที่ชนะแล้วปิด series (win step) — กี่ % ของการกู้สำเร็จ")
    print("-" * 70)
    print(f"  {'Step':>6} | {'จำนวน':>6} | {'%':>6} | {'กู้ครบหนี้':>10} | {'หนี้เฉลี่ยก่อน win':>18} | {'lot ชนะเฉลี่ย':>12}")
    print(f"  {'-'*6} + {'-'*6} + {'-'*6} + {'-'*10} + {'-'*18} + {'-'*12}")

    total = len(recovered)
    for step in sorted(by_step.keys()):
        items = by_step[step]
        n = len(items)
        pct = 100 * n / total
        full = sum(1 for x in items if x["full_cover"])
        avg_debt = sum(x["cum_loss"] for x in items) / n
        avg_vol = sum(x["volume_win"] for x in items) / n
        print(f"  {step:6d} | {n:6d} | {pct:5.1f}% | {full:4d}/{n:4d}   | ${avg_debt:16.2f} | {avg_vol:10.3f}")

    # จำนวน loss ก่อน win
    by_n_loss = defaultdict(int)
    for r in recovered:
        by_n_loss[r["n_losses"]] += 1

    print("\n" + "-" * 70)
    print("  จำนวนไม้ที่แพ้ก่อนชนะ (n_losses) — ใกล้เคียง step ชนะ")
    print("-" * 70)
    for k in sorted(by_n_loss.keys()):
        n = by_n_loss[k]
        print(f"    แพ้ {k} ไม้แล้วชนะ → {n} ครั้ง ({100*n/total:.1f}%)")

    # Median / mode
    steps = [r["win_step"] for r in recovered]
    steps.sort()
    median = steps[len(steps)//2]
    mode_step = max(by_step.keys(), key=lambda k: len(by_step[k]))

    print("\n" + "-" * 70)
    print("  สรุปสำหรับตั้งค่า 'กู้ทีเดียว'")
    print("-" * 70)
    print(f"    Step ที่ชนะบ่อยสุด (mode):     step {mode_step} ({len(by_step[mode_step])}/{total} = {100*len(by_step[mode_step])/total:.1f}%)")
    print(f"    Step กลาง (median):            step {median}")
    cum_pct = 0
    rec_steps = []
    for step in sorted(by_step.keys()):
        cum_pct += 100 * len(by_step[step]) / total
        rec_steps.append(step)
        if cum_pct >= 80:
            print(f"    80% กู้สำเร็จภายใน step ≤ {step}")
            break
    if cum_pct < 80 and rec_steps:
        print(f"    (ข้อมูลมีถึง step {max(rec_steps)} เท่านั้น)")

    # หนี้สะสมก่อน win ที่ step ต่างๆ
    print("\n" + "-" * 70)
    print("  หนี้สะสมก่อนไม้ชนะ (USD) — ใช้ตั้ง one_shot_max_debt_pct")
    print("-" * 70)
    debts = [r["cum_loss"] for r in recovered]
    debts.sort()
    if debts:
        print(f"    min: ${debts[0]:.2f}  |  median: ${debts[len(debts)//2]:.2f}  |  max: ${debts[-1]:.2f}")
        for pct in [50, 75, 90, 95]:
            idx = int(len(debts) * pct / 100) - 1
            idx = max(0, min(idx, len(debts)-1))
            print(f"    p{pct}: ${debts[idx]:.2f}")

    # กู้ไม่ครบในครั้งเดียว
    not_full = [r for r in recovered if not r["full_cover"]]
    if not_full:
        print(f"\n  ⚠ ชนะแต่กู้ไม่ครบหนี้ใน 1 ไม้: {len(not_full)}/{total} ({100*len(not_full)/total:.1f}%)")
        by_step_nf = defaultdict(int)
        for r in not_full:
            by_step_nf[r["win_step"]] += 1
        for st in sorted(by_step_nf.keys()):
            print(f"      step {st}: {by_step_nf[st]} ครั้ง")

    # Recent series 508 style
    print("\n" + "-" * 70)
    print("  ตัวอย่าง 10 series ล่าสุดที่กู้สำเร็จ")
    print("-" * 70)
    for r in sorted(recovered, key=lambda x: -x["sid"])[:10]:
        cover = "ครบ" if r["full_cover"] else "ไม่ครบ"
        print(f"    #{r['sid']:4d} step{r['win_step']} แพ้{r['n_losses']}ไม้ "
              f"หนี้${r['cum_loss']:.0f} win${r['win_pnl']:.0f} lot{r['volume_win']:.2f} [{cover}]")

    conn.close()

if __name__ == "__main__":
    main()
