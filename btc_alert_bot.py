"""
╔══════════════════════════════════════════════════════════╗
║         BTC ALERT BOT v3 — PythonAnywhere               ║
║   100% Crypto.com API — sem Binance, sem bloqueios       ║
║   8 indicadores: BB, ATR, Volume, RSI, S/R,              ║
║                  Funding Rate, VWAP, EMA Cross           ║
╚══════════════════════════════════════════════════════════╝
"""

import os
import time
import logging
import requests
import numpy as np
from datetime import datetime, timezone

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO — variáveis de ambiente
# ─────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

CHECK_INTERVAL  = 900   # 15 minutos
ALERT_COOLDOWN  = 3600  # não repete o mesmo alerta em 1h
MIN_SCORE       = 3     # score mínimo para alertar (em 8)

# Parâmetros técnicos
BB_PERIOD       = 20
BB_MULT         = 2.0
BB_SQUEEZE_PCT  = 3.0
ATR_PERIOD      = 14
VOL_SMA         = 20
VOL_SPIKE_X     = 2.0
RSI_PERIOD      = 14
EMA_FAST        = 9
EMA_SLOW        = 21
VWAP_DEV_PCT    = 2.0
FUNDING_BULL    = -0.005   # funding negativo → sinal bullish
FUNDING_BEAR    = 0.010    # funding positivo alto → sinal bearish

SR_LEVELS       = [70000, 72000, 74000, 76000, 78000, 80000, 82000, 85000, 90000, 95000, 100000]
SR_TOLERANCE    = 0.015

# Crypto.com API — spot e perpetual
API_BASE        = "https://api.crypto.com/exchange/v1"
INSTRUMENT_SPOT = "BTC_USD"
INSTRUMENT_PERP = "BTCUSD-PERP"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("BTCBot")


# ─────────────────────────────────────────────
#  FETCH DE DADOS — tudo na Crypto.com
# ─────────────────────────────────────────────
def fetch_candles(instrument=INSTRUMENT_SPOT, timeframe="1h", limit=100):
    """Candlesticks spot ou perpetual."""
    url = f"{API_BASE}/public/get-candlestick"
    params = {"instrument_name": instrument, "timeframe": timeframe, "count": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("result", {}).get("data", [])
        if not data:
            return None, None, None, None
        return (
            [float(c["c"]) for c in data],
            [float(c["h"]) for c in data],
            [float(c["l"]) for c in data],
            [float(c["v"]) for c in data],
        )
    except Exception as e:
        log.error(f"Erro candles ({instrument}): {e}")
        return None, None, None, None


def fetch_mark_price():
    """Mark price do perpetual — usado para calcular funding implícito."""
    url = f"{API_BASE}/public/get-mark-price-history"
    params = {"instrument_name": INSTRUMENT_PERP, "timeframe": "1h", "count": 2}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("result", {}).get("data", [])
        if data:
            return float(data[-1].get("v", 0))
        return None
    except Exception as e:
        log.warning(f"Mark price indisponível: {e}")
        return None


def fetch_funding_rate():
    """
    Funding rate estimado: diferença % entre preço perpetual e spot.
    Positivo = longs a pagar (bearish). Negativo = shorts a pagar (bullish).
    """
    try:
        # Spot
        r_spot = requests.get(
            f"{API_BASE}/public/get-candlestick",
            params={"instrument_name": INSTRUMENT_SPOT, "timeframe": "1h", "count": 1},
            timeout=10
        )
        spot_data = r_spot.json().get("result", {}).get("data", [])
        spot_price = float(spot_data[-1]["c"]) if spot_data else None

        # Perpetual
        r_perp = requests.get(
            f"{API_BASE}/public/get-candlestick",
            params={"instrument_name": INSTRUMENT_PERP, "timeframe": "1h", "count": 1},
            timeout=10
        )
        perp_data = r_perp.json().get("result", {}).get("data", [])
        perp_price = float(perp_data[-1]["c"]) if perp_data else None

        if spot_price and perp_price and spot_price > 0:
            basis = (perp_price - spot_price) / spot_price * 100
            return basis, spot_price, perp_price
        return None, None, None
    except Exception as e:
        log.warning(f"Funding estimado indisponível: {e}")
        return None, None, None


# ─────────────────────────────────────────────
#  INDICADORES
# ─────────────────────────────────────────────
def sma(arr, p):
    a = np.array(arr, dtype=float)
    r = np.full(len(a), np.nan)
    for i in range(p - 1, len(a)):
        r[i] = a[i - p + 1 : i + 1].mean()
    return r


def calc_ema(arr, p):
    a = np.array(arr, dtype=float)
    k = 2 / (p + 1)
    e = np.full(len(a), np.nan)
    e[0] = a[0]
    for i in range(1, len(a)):
        e[i] = a[i] * k + e[i - 1] * (1 - k)
    return e


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
    if p < len(tr):
        atr[p] = tr[:p].mean()
        for i in range(p + 1, len(c)):
            atr[i] = (atr[i - 1] * (p - 1) + tr[i - 1]) / p
    return atr


def calc_vwap(highs, lows, closes, volumes, periods=24):
    h = np.array(highs[-periods:])
    l = np.array(lows[-periods:])
    c = np.array(closes[-periods:])
    v = np.array(volumes[-periods:])
    typical = (h + l + c) / 3
    return np.sum(typical * v) / np.sum(v)


def calc_ema_cross(closes):
    ema_fast = calc_ema(closes, EMA_FAST)
    ema_slow = calc_ema(closes, EMA_SLOW)
    prev_diff = ema_fast[-2] - ema_slow[-2]
    curr_diff = ema_fast[-1] - ema_slow[-1]
    if prev_diff < 0 and curr_diff >= 0: return "bullish", ema_fast[-1], ema_slow[-1]
    if prev_diff > 0 and curr_diff <= 0: return "bearish", ema_fast[-1], ema_slow[-1]
    return None, ema_fast[-1], ema_slow[-1]


def rsi_divergence(closes, rsis, lb=10):
    valid = [(c, r) for c, r in zip(closes[-lb:], rsis[-lb:]) if not np.isnan(r)]
    if len(valid) < 4: return None
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
#  ANÁLISE PRINCIPAL
# ─────────────────────────────────────────────
def analyse():
    closes, highs, lows, vols = fetch_candles(INSTRUMENT_SPOT, "1h", 100)
    if not closes or len(closes) < 30:
        return None

    # Indicadores base
    _, bbw  = calc_bb(closes, BB_PERIOD, BB_MULT)
    atr     = calc_atr(highs, lows, closes, ATR_PERIOD)
    atr_m   = sma(np.nan_to_num(atr), 20)
    vol_m   = sma(vols, VOL_SMA)
    rsi     = calc_rsi(closes, RSI_PERIOD)

    # Novos indicadores
    vwap                    = calc_vwap(highs, lows, closes, vols)
    ema_cross, ema_f, ema_s = calc_ema_cross(closes)
    basis, spot_p, perp_p   = fetch_funding_rate()

    price    = closes[-1]
    vwap_dev = (price - vwap) / vwap * 100

    # ── 8 Sinais ──
    sig_sq   = not np.isnan(bbw[-1])   and bbw[-1]  < BB_SQUEEZE_PCT
    sig_atr  = not np.isnan(atr[-1])   and not np.isnan(atr_m[-1]) and atr[-1] > atr_m[-1]
    sig_vol  = not np.isnan(vol_m[-1]) and vols[-1] > vol_m[-1] * VOL_SPIKE_X
    sig_div  = rsi_divergence(closes, rsi)
    sig_sr, sr_lvl = near_sr(price)

    # Funding: basis (perp - spot) extremo
    sig_funding = False
    funding_dir = None
    if basis is not None:
        if basis >= FUNDING_BEAR:
            sig_funding = True; funding_dir = "bearish (longs sobrecarregados)"
        elif basis <= FUNDING_BULL:
            sig_funding = True; funding_dir = "bullish (shorts a ser espremidos)"

    sig_vwap = abs(vwap_dev) >= VWAP_DEV_PCT
    vwap_dir = "acima" if vwap_dev > 0 else "abaixo"
    sig_ema  = ema_cross is not None

    score = sum([sig_sq, sig_atr, sig_vol, sig_div is not None,
                 sig_sr, sig_funding, sig_vwap, sig_ema])

    return dict(
        price=price, bbw=bbw[-1], atr=atr[-1], atr_m=atr_m[-1],
        vol=vols[-1], vol_m=vol_m[-1], rsi=rsi[-1],
        vwap=vwap, vwap_dev=vwap_dev, vwap_dir=vwap_dir,
        ema_fast=ema_f, ema_slow=ema_s, ema_cross=ema_cross,
        basis=basis, spot_p=spot_p, perp_p=perp_p, funding_dir=funding_dir,
        sig_sq=sig_sq, sig_atr=sig_atr, sig_vol=sig_vol,
        sig_div=sig_div, sig_sr=sig_sr, sr_lvl=sr_lvl,
        sig_funding=sig_funding, sig_vwap=sig_vwap, sig_ema=sig_ema,
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
        log.info("Alerta enviado.")
    except Exception as e:
        log.error(f"Erro Telegram: {e}")


def build_msg(d):
    score = d["score"]
    icon  = "🔴" if score >= 6 else "🟠" if score >= 3 else "🟢"
    lvl   = "ALTO" if score >= 6 else "MÉDIO" if score >= 3 else "BAIXO"

    def line(sig, label): return f"{'✅' if sig else '⬜'} {label}"

    basis_txt = (
        f"Basis perp/spot: {d['basis']:+.3f}% ({d['funding_dir']})"
        if d['basis'] is not None else "Basis perp/spot: N/A"
    )
    interp = (
        "⚠️ ALERTA MÁXIMO — múltiplos sinais. Movimento ≥10% muito provável. Fecha posições ou define stops apertados."
        if score >= 6
        else "👀 Vários sinais activos. Mantém atenção e prepara o plano."
        if score >= 3
        else "✅ Mercado calmo. Boa altura para planear entradas."
    )

    return "\n".join([
        f"<b>⚡ BTC ALERT v3 — {icon} {lvl}  ({score}/8)</b>",
        f"💰 <b>Preço:</b> ${d['price']:,.0f}   🕐 {d['ts']}",
        "",
        "<b>── Indicadores ──</b>",
        line(d['sig_sq'],      f"BB Squeeze      (BB Width: {d['bbw']:.2f}%)"),
        line(d['sig_atr'],     f"ATR elevado     (${d['atr']:.0f} vs média ${d['atr_m']:.0f})"),
        line(d['sig_vol'],     f"Volume spike    ({d['vol']:.0f} vs média {d['vol_m']:.0f})"),
        line(d['sig_div'],     f"RSI divergência {d['sig_div'] or '—'}  (RSI: {d['rsi']:.1f})"),
        line(d['sig_sr'],      f"Zona S/R        (nível: ${d['sr_lvl']:,})" if d['sig_sr'] else "Zona S/R        (longe)"),
        line(d['sig_funding'], basis_txt),
        line(d['sig_vwap'],    f"VWAP desvio     {d['vwap_dev']:+.1f}% {d['vwap_dir']} (${d['vwap']:,.0f})"),
        line(d['sig_ema'],     f"EMA Cross       {d['ema_cross'] or 'sem cruzamento'} (fast ${d['ema_fast']:,.0f} / slow ${d['ema_slow']:,.0f})"),
        "",
        interp,
        "",
        "⚠️ Não é conselho financeiro. Gere sempre o risco.",
    ])


# ─────────────────────────────────────────────
#  LOOP PRINCIPAL
# ─────────────────────────────────────────────
def main():
    log.info("BTC Alert Bot v3 iniciado.")
    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN não definido! Define a variável e reinicia.")
        return

    send_telegram(
        f"🤖 <b>BTC Alert Bot v3 online!</b>\n"
        f"8 indicadores — 100% Crypto.com API (sem bloqueios).\n"
        f"BB Squeeze · ATR · Volume · RSI · S/R · Basis Perp/Spot · VWAP · EMA Cross\n"
        f"Alertas quando score ≥ {MIN_SCORE}/8 — de {CHECK_INTERVAL // 60} em {CHECK_INTERVAL // 60} minutos."
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
                f"Preço: ${d['price']:,.0f} | Score: {d['score']}/8 | "
                f"BBW: {d['bbw']:.2f}% | RSI: {d['rsi']:.1f} | "
                f"VWAP dev: {d['vwap_dev']:+.1f}% | "
                f"Basis: {d['basis']:+.3f}%" if d['basis'] else
                f"Preço: ${d['price']:,.0f} | Score: {d['score']}/8"
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
            log.error(f"Erro: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
