"""
╔══════════════════════════════════════════════════════════╗
║         BTC ALERT BOT v2 — Railway.app                   ║
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
MIN_SCORE       = 3     # score mínimo para alertar (agora em 8)

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
VWAP_DEV_PCT    = 2.0    # desvio do VWAP em % para sinal
FUNDING_BULL    = -0.05  # funding negativo → sinal bullish
FUNDING_BEAR    = 0.10   # funding positivo alto → sinal bearish

SR_LEVELS       = [70000, 72000, 74000, 76000, 78000, 80000, 82000, 85000, 90000, 95000, 100000]
SR_TOLERANCE    = 0.015

API_CRYPTO      = "https://api.crypto.com/exchange/v1"
API_BINANCE     = "https://fapi.binance.com"         # funding rate (grátis)
API_COINGLASS   = "https://open-api.coinglass.com/public/v2"  # open interest (grátis)
INSTRUMENT      = "BTC_USD"
SYMBOL_BINANCE  = "BTCUSDT"

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
    url = f"{API_CRYPTO}/public/get-candlestick"
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
        log.error(f"Erro candles: {e}")
        return None, None, None, None


def fetch_funding_rate():
    """Funding rate actual dos futuros perpétuos BTC (Binance)."""
    try:
        url = f"{API_BINANCE}/fapi/v1/premiumIndex"
        r = requests.get(url, params={"symbol": SYMBOL_BINANCE}, timeout=10)
        r.raise_for_status()
        data = r.json()
        rate = float(data.get("lastFundingRate", 0)) * 100  # em percentagem
        return rate
    except Exception as e:
        log.warning(f"Funding rate indisponível: {e}")
        return None


def fetch_open_interest():
    """Open Interest BTC em USD (Binance Futures)."""
    try:
        url = f"{API_BINANCE}/fapi/v1/openInterest"
        r = requests.get(url, params={"symbol": SYMBOL_BINANCE}, timeout=10)
        r.raise_for_status()
        data = r.json()
        return float(data.get("openInterest", 0))
    except Exception as e:
        log.warning(f"Open Interest indisponível: {e}")
        return None


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
    atr[p] = tr[:p].mean()
    for i in range(p + 1, len(c)):
        atr[i] = (atr[i - 1] * (p - 1) + tr[i - 1]) / p
    return atr


def calc_vwap(highs, lows, closes, volumes):
    """VWAP diário simples com os últimos 24 períodos (1h)."""
    h = np.array(highs[-24:])
    l = np.array(lows[-24:])
    c = np.array(closes[-24:])
    v = np.array(volumes[-24:])
    typical = (h + l + c) / 3
    vwap = np.sum(typical * v) / np.sum(v)
    return vwap


def calc_ema_cross(closes):
    """Retorna sinal de cruzamento EMA rápida/lenta."""
    ema_fast = calc_ema(closes, EMA_FAST)
    ema_slow = calc_ema(closes, EMA_SLOW)
    prev_diff = ema_fast[-2] - ema_slow[-2]
    curr_diff = ema_fast[-1] - ema_slow[-1]
    if prev_diff < 0 and curr_diff >= 0:
        return "bullish", ema_fast[-1], ema_slow[-1]
    if prev_diff > 0 and curr_diff <= 0:
        return "bearish", ema_fast[-1], ema_slow[-1]
    direction = "bullish" if curr_diff > 0 else "bearish"
    return None, ema_fast[-1], ema_slow[-1]


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
#  ANÁLISE PRINCIPAL
# ─────────────────────────────────────────────
def analyse():
    closes, highs, lows, vols = fetch_candles(limit=100)
    if not closes or len(closes) < 30:
        return None

    # Indicadores base
    _, bbw    = calc_bb(closes, BB_PERIOD, BB_MULT)
    atr       = calc_atr(highs, lows, closes, ATR_PERIOD)
    atr_m     = sma(np.nan_to_num(atr), 20)
    vol_m     = sma(vols, VOL_SMA)
    rsi       = calc_rsi(closes, RSI_PERIOD)

    # Novos indicadores
    vwap           = calc_vwap(highs, lows, closes, vols)
    ema_cross, ema_fast_val, ema_slow_val = calc_ema_cross(closes)
    funding_rate   = fetch_funding_rate()
    open_interest  = fetch_open_interest()

    price          = closes[-1]
    vwap_dev       = (price - vwap) / vwap * 100  # % de desvio do VWAP

    # ── Sinais (8 no total) ──
    sig_sq    = not np.isnan(bbw[-1])  and bbw[-1]  < BB_SQUEEZE_PCT
    sig_atr   = not np.isnan(atr[-1])  and not np.isnan(atr_m[-1]) and atr[-1] > atr_m[-1]
    sig_vol   = not np.isnan(vol_m[-1]) and vols[-1] > vol_m[-1] * VOL_SPIKE_X
    sig_div   = rsi_divergence(closes, rsi)
    sig_sr, sr_lvl = near_sr(price)

    # Funding Rate: extremos sinalizam reversão
    sig_funding = False
    funding_dir = None
    if funding_rate is not None:
        if funding_rate >= FUNDING_BEAR:
            sig_funding = True; funding_dir = "bearish"
        elif funding_rate <= FUNDING_BULL:
            sig_funding = True; funding_dir = "bullish"

    # VWAP: preço muito afastado do VWAP
    sig_vwap = abs(vwap_dev) >= VWAP_DEV_PCT
    vwap_dir = "acima" if vwap_dev > 0 else "abaixo"

    # EMA Cross: cruzamento recente
    sig_ema = ema_cross is not None

    score = sum([sig_sq, sig_atr, sig_vol, sig_div is not None,
                 sig_sr, sig_funding, sig_vwap, sig_ema])

    return dict(
        price=price, bbw=bbw[-1], atr=atr[-1], atr_m=atr_m[-1],
        vol=vols[-1], vol_m=vol_m[-1], rsi=rsi[-1],
        vwap=vwap, vwap_dev=vwap_dev, vwap_dir=vwap_dir,
        ema_fast=ema_fast_val, ema_slow=ema_slow_val,
        ema_cross=ema_cross, open_interest=open_interest,
        funding_rate=funding_rate, funding_dir=funding_dir,
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

    # Linhas de cada indicador
    def line(sig, label): return f"{'✅' if sig else '⬜'} {label}"

    funding_txt = f"Funding: {d['funding_rate']:.3f}% ({d['funding_dir']})" if d['funding_rate'] is not None else "Funding: N/A"
    vwap_txt    = f"VWAP: {d['vwap_dev']:+.1f}% {d['vwap_dir']} (${d['vwap']:,.0f})"
    ema_txt     = f"EMA Cross: {d['ema_cross'] or 'sem cruzamento'}  (fast ${d['ema_fast']:,.0f} / slow ${d['ema_slow']:,.0f})"
    sr_txt      = f"Zona S/R: ${d['sr_lvl']:,}" if d['sig_sr'] else "Zona S/R: longe"
    oi_txt      = f"Open Interest: {d['open_interest']:,.0f} BTC" if d['open_interest'] else ""

    interp = (
        "⚠️ ALERTA MÁXIMO — múltiplos sinais. Movimento ≥10% muito provável. Fecha posições ou define stops apertados."
        if score >= 6
        else "👀 Vários sinais activos. Mantém atenção e prepara o plano."
        if score >= 3
        else "✅ Mercado calmo. Boa altura para planear entradas."
    )

    lines = [
        f"<b>⚡ BTC ALERT v2 — {icon} {lvl}  ({score}/8)</b>",
        f"💰 <b>Preço:</b> ${d['price']:,.0f}   🕐 {d['ts']}",
        "",
        "<b>── Indicadores ──</b>",
        line(d['sig_sq'],       f"BB Squeeze      (BB Width: {d['bbw']:.2f}%)"),
        line(d['sig_atr'],      f"ATR elevado     (${d['atr']:.0f} vs média ${d['atr_m']:.0f})"),
        line(d['sig_vol'],      f"Volume spike    ({d['vol']:.0f} vs média {d['vol_m']:.0f})"),
        line(d['sig_div'],      f"RSI divergência {d['sig_div'] or '—'}  (RSI: {d['rsi']:.1f})"),
        line(d['sig_sr'],       sr_txt),
        line(d['sig_funding'],  funding_txt),
        line(d['sig_vwap'],     vwap_txt),
        line(d['sig_ema'],      ema_txt),
    ]
    if oi_txt:
        lines.append(f"📊 {oi_txt}")
    lines += ["", interp, "", "⚠️ Não é conselho financeiro. Gere sempre o risco."]
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  LOOP PRINCIPAL
# ─────────────────────────────────────────────
def main():
    log.info("BTC Alert Bot v2 iniciado.")
    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN não definido! Define a variável e reinicia.")
        return

    send_telegram(
        f"🤖 <b>BTC Alert Bot v2 online!</b>\n"
        f"8 indicadores activos: BB, ATR, Volume, RSI, S/R, Funding Rate, VWAP, EMA Cross.\n"
        f"Alertas quando score ≥ {MIN_SCORE}/8 — a cada {CHECK_INTERVAL // 60} minutos."
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
                f"Funding: {d['funding_rate']:.3f}%" if d['funding_rate'] else
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
