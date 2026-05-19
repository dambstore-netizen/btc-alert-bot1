"""
╔══════════════════════════════════════════════════════════╗
║         BTC ALERT BOT — Pronto para Railway.app          ║
║   As credenciais são lidas das variáveis de ambiente     ║
╚══════════════════════════════════════════════════════════╝
"""

import os
import time
import logging
import requests
import numpy as np
from datetime import datetime, timezone

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO — lida das variáveis de ambiente
# ─────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

CHECK_INTERVAL   = 900   # 15 minutos
ALERT_COOLDOWN   = 3600  # não repete o mesmo alerta em 1h
MIN_SCORE        = 2     # score mínimo para alertar

BB_PERIOD      = 20
BB_MULT        = 2.0
BB_SQUEEZE_PCT = 3.0
ATR_PERIOD     = 14
VOL_SMA        = 20
VOL_SPIKE_X    = 2.0
RSI_PERIOD     = 14

SR_LEVELS    = [70000, 72000, 74000, 76000, 78000, 80000, 82000, 85000, 90000, 95000, 100000]
SR_TOLERANCE = 0.015

API_BASE   = "https://api.crypto.com/exchange/v1"
INSTRUMENT = "BTC_USD"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("BTCBot")


# ─────────────────────────────────────────────
#  FETCH DE DADOS
# ─────────────────────────────────────────────
def fetch_candles(timeframe="1h", limit=100):
    url = f"{API_BASE}/public/get-candlestick"
    params = {"instrument_name": INSTRUMENT, "timeframe": timeframe, "count": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("result", {}).get("data", [])
        return (
            [float(c["c"]) for c in data],
            [float(c["h"]) for c in data],
            [float(c["l"]) for c in data],
            [float(c["v"]) for c in data],
        )
    except Exception as e:
        log.error(f"Erro ao buscar candles: {e}")
        return None, None, None, None


# ─────────────────────────────────────────────
#  INDICADORES
# ─────────────────────────────────────────────
def sma(arr, p):
    a = np.array(arr, dtype=float)
    r = np.full(len(a), np.nan)
    for i in range(p - 1, len(a)):
        r[i] = a[i - p + 1 : i + 1].mean()
    return r

def calc_rsi(closes, p=14):
    c = np.array(closes, dtype=float)
    d = np.diff(c)
    gains  = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)
    ag, al = gains[:p].mean(), losses[:p].mean()
    rsis = [np.nan] * (p + 1)
    rsis.append(100 if al == 0 else 100 - 100 / (1 + ag / al))
    for i in range(p, len(gains)):
        ag = (ag * (p - 1) + gains[i]) / p
        al = (al * (p - 1) + losses[i]) / p
        rsis.append(100 if al == 0 else 100 - 100 / (1 + ag / al))
    return np.array(rsis)

def calc_bb(closes, p=20, m=2.0):
    c   = np.array(closes, dtype=float)
    mid = sma(c, p)
    width = np.full(len(c), np.nan)
    for i in range(p - 1, len(c)):
        sd = c[i - p + 1 : i + 1].std()
        width[i] = (2 * m * sd) / mid[i] * 100
    return mid, width

def calc_atr(highs, lows, closes, p=14):
    h, l, c = np.array(highs), np.array(lows), np.array(closes)
    tr  = np.maximum(h[1:] - l[1:],
          np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    atr = np.full(len(c), np.nan)
    atr[p] = tr[:p].mean()
    for i in range(p + 1, len(c)):
        atr[i] = (atr[i - 1] * (p - 1) + tr[i - 1]) / p
    return atr

def rsi_divergence(closes, rsis, lb=10):
    valid = [(c, r) for c, r in zip(closes[-lb:], rsis[-lb:]) if not np.isnan(r)]
    if len(valid) < 4:
        return None
    p  = [v[0] for v in valid]
    rs = [v[1] for v in valid]
    if p[-1] > max(p[:-1]) and rs[-1] <= max(rs[:-1]): return "bearish"
    if p[-1] < min(p[:-1]) and rs[-1] >= min(rs[:-1]): return "bullish"
    return None

def near_sr(price):
    for lvl in SR_LEVELS:
        if abs(price - lvl) / price <= SR_TOLERANCE:
            return True, lvl
    return False, None


# ─────────────────────────────────────────────
#  ANÁLISE
# ─────────────────────────────────────────────
def analyse():
    closes, highs, lows, vols = fetch_candles(limit=100)
    if not closes or len(closes) < 30:
        return None

    _, bbw    = calc_bb(closes, BB_PERIOD, BB_MULT)
    atr       = calc_atr(highs, lows, closes, ATR_PERIOD)
    atr_m     = sma(np.nan_to_num(atr), 20)
    vol_m     = sma(vols, VOL_SMA)
    rsi       = calc_rsi(closes, RSI_PERIOD)

    price     = closes[-1]
    sig_sq    = not np.isnan(bbw[-1])  and bbw[-1]  < BB_SQUEEZE_PCT
    sig_atr   = not np.isnan(atr[-1])  and not np.isnan(atr_m[-1]) and atr[-1] > atr_m[-1]
    sig_vol   = not np.isnan(vol_m[-1]) and vols[-1] > vol_m[-1] * VOL_SPIKE_X
    sig_div   = rsi_divergence(closes, rsi)
    sig_sr, sr_lvl = near_sr(price)

    score = sum([sig_sq, sig_atr, sig_vol, sig_div is not None, sig_sr])

    return dict(
        price=price, bbw=bbw[-1], atr=atr[-1], atr_m=atr_m[-1],
        vol=vols[-1], vol_m=vol_m[-1], rsi=rsi[-1],
        sig_sq=sig_sq, sig_atr=sig_atr, sig_vol=sig_vol,
        sig_div=sig_div, sig_sr=sig_sr, sr_lvl=sr_lvl,
        score=score,
        ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


# ─────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────
def send_telegram(msg):
    if not TELEGRAM_TOKEN:
        log.warning("TELEGRAM_TOKEN não definido.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        }, timeout=10)
        log.info("Alerta Telegram enviado.")
    except Exception as e:
        log.error(f"Erro Telegram: {e}")

def build_msg(d):
    icon = "🔴" if d["score"] >= 4 else "🟠" if d["score"] >= 2 else "🟢"
    lvl  = "ALTO" if d["score"] >= 4 else "MÉDIO" if d["score"] >= 2 else "BAIXO"
    sr_txt = f"✅ Zona S/R  (nível: ${d['sr_lvl']:,})" if d["sig_sr"] else "⬜ Zona S/R  (longe de S/R)"
    interp = (
        "⚠️ Múltiplos sinais! Movimento ≥10% possível. Define stops apertados." if d["score"] >= 4
        else "👀 Alguns sinais activos. Monitoriza e prepara o plano." if d["score"] >= 2
        else "✅ Mercado calmo. Boa altura para planear entradas."
    )
    return "\n".join([
        f"<b>⚡ BTC ALERT — {icon} {lvl}  (score {d['score']}/5)</b>",
        f"💰 <b>Preço:</b> ${d['price']:,.0f}   🕐 {d['ts']}",
        "",
        f"{'✅' if d['sig_sq']  else '⬜'} BB Squeeze  (BB Width: {d['bbw']:.2f}%)",
        f"{'✅' if d['sig_atr'] else '⬜'} ATR elevado  (${d['atr']:.0f} vs média ${d['atr_m']:.0f})",
        f"{'✅' if d['sig_vol'] else '⬜'} Volume spike  ({d['vol']:.0f} vs média {d['vol_m']:.0f})",
        f"{'✅' if d['sig_div'] else '⬜'} RSI divergência  {d['sig_div'] or '—'}  (RSI: {d['rsi']:.1f})",
        sr_txt, "",
        interp, "",
        "⚠️ Não é conselho financeiro. Gere sempre o risco.",
    ])


# ─────────────────────────────────────────────
#  LOOP PRINCIPAL
# ─────────────────────────────────────────────
def main():
    log.info("BTC Alert Bot iniciado.")
    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN não está definido! Define a variável de ambiente e reinicia.")
        return

    send_telegram(
        f"🤖 <b>BTC Alert Bot online!</b>\n"
        f"A monitorizar BTC/USD de {CHECK_INTERVAL // 60} em {CHECK_INTERVAL // 60} minutos.\n"
        f"Recebes alertas quando score ≥ {MIN_SCORE}/5."
    )

    last_score = 0
    last_alert = 0

    while True:
        try:
            log.info("A analisar mercado...")
            d = analyse()

            if d is None:
                log.warning("Sem dados — a tentar em 60s.")
                time.sleep(60)
                continue

            log.info(
                f"Preço: ${d['price']:,.0f} | Score: {d['score']}/5 | "
                f"BBW: {d['bbw']:.2f}% | RSI: {d['rsi']:.1f} | Vol: {d['vol']:.0f}"
            )

            now = time.time()
            if d["score"] >= MIN_SCORE and (
                d["score"] != last_score or now - last_alert > ALERT_COOLDOWN
            ):
                send_telegram(build_msg(d))
                last_score = d["score"]
                last_alert = now

        except KeyboardInterrupt:
            log.info("Bot parado.")
            break
        except Exception as e:
            log.error(f"Erro inesperado: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
