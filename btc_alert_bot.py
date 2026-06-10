"""
╔══════════════════════════════════════════════════════════╗
║         BTC ALERT BOT v4 — Railway                      ║
║   100% Crypto.com API — sem Binance, sem bloqueios       ║
║   Indicadores melhorados com deteção de quebra S/R       ║
║   Scores separados LONG / SHORT                          ║
╚══════════════════════════════════════════════════════════╝
"""

import os
import time
import logging
import requests
import numpy as np
from datetime import datetime, timezone

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO
# ─────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

CHECK_INTERVAL  = 900    # 15 minutos
ALERT_COOLDOWN  = 3600   # não repete o mesmo alerta em 1h
MIN_SCORE       = 3      # score mínimo para alertar

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
FUNDING_BULL    = -0.005
FUNDING_BEAR    = 0.010

# S/R dinâmico
SR_SWING_BARS   = 3      # nº de barras para confirmar swing high/low
SR_TOLERANCE    = 0.008  # 0.8% de tolerância para considerar nível ativo
SR_MIN_TOUCHES  = 1      # mínimo de toques para confirmar nível
BREAK_CONFIRM_PCT = 0.005 # quebra confirmada se fechar 0.5% abaixo/acima do nível

# Crypto.com API
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
#  FETCH DE DADOS
# ─────────────────────────────────────────────
def fetch_candles(instrument=INSTRUMENT_SPOT, timeframe="15m", limit=100):
    """Candlesticks 15min para deteção mais rápida."""
    url = f"{API_BASE}/public/get-candlestick"
    params = {"instrument_name": instrument, "timeframe": timeframe, "count": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("result", {}).get("data", [])
        if not data:
            return None, None, None, None, None
        return (
            [float(c["c"]) for c in data],   # closes
            [float(c["h"]) for c in data],   # highs
            [float(c["l"]) for c in data],   # lows
            [float(c["v"]) for c in data],   # volumes
            [float(c["t"]) for c in data],   # timestamps
        )
    except Exception as e:
        log.error(f"Erro candles ({instrument}): {e}")
        return None, None, None, None, None


def fetch_funding_rate():
    try:
        r_spot = requests.get(
            f"{API_BASE}/public/get-candlestick",
            params={"instrument_name": INSTRUMENT_SPOT, "timeframe": "1h", "count": 1},
            timeout=10
        )
        spot_data  = r_spot.json().get("result", {}).get("data", [])
        spot_price = float(spot_data[-1]["c"]) if spot_data else None

        r_perp = requests.get(
            f"{API_BASE}/public/get-candlestick",
            params={"instrument_name": INSTRUMENT_PERP, "timeframe": "1h", "count": 1},
            timeout=10
        )
        perp_data  = r_perp.json().get("result", {}).get("data", [])
        perp_price = float(perp_data[-1]["c"]) if perp_data else None

        if spot_price and perp_price and spot_price > 0:
            basis = (perp_price - spot_price) / spot_price * 100
            return basis, spot_price, perp_price
        return None, None, None
    except Exception as e:
        log.warning(f"Funding indisponível: {e}")
        return None, None, None


# ─────────────────────────────────────────────
#  INDICADORES BASE
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
        width[i] = (2 * m * sd) / mid[i] * 100 if mid[i] else np.nan
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


def calc_vwap(highs, lows, closes, volumes, periods=96):
    # 96 candles de 15min = 24h
    n = min(periods, len(closes))
    h = np.array(highs[-n:])
    l = np.array(lows[-n:])
    c = np.array(closes[-n:])
    v = np.array(volumes[-n:])
    typical = (h + l + c) / 3
    return np.sum(typical * v) / np.sum(v) if np.sum(v) > 0 else c[-1]


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
    if len(valid) < 4:
        return None
    p  = [v[0] for v in valid]
    rs = [v[1] for v in valid]
    if p[-1] > max(p[:-1]) and rs[-1] <= max(rs[:-1]): return "bearish"
    if p[-1] < min(p[:-1]) and rs[-1] >= min(rs[:-1]): return "bullish"
    return None


# ─────────────────────────────────────────────
#  S/R DINÂMICO — NOVO
# ─────────────────────────────────────────────
def find_swing_levels(highs, lows, closes, n=SR_SWING_BARS, lookback=60):
    """
    Encontra suportes e resistências dinâmicos a partir de swing highs/lows.
    Retorna listas de (nivel, tipo) dos últimos `lookback` candles.
    """
    h = highs[-lookback:]
    l = lows[-lookback:]

    supports    = []
    resistances = []

    for i in range(n, len(l) - n):
        # Swing low = suporte
        if all(l[i] <= l[i - j] for j in range(1, n + 1)) and \
           all(l[i] <= l[i + j] for j in range(1, n + 1)):
            supports.append(l[i])
        # Swing high = resistência
        if all(h[i] >= h[i - j] for j in range(1, n + 1)) and \
           all(h[i] >= h[i + j] for j in range(1, n + 1)):
            resistances.append(h[i])

    # Agrupa níveis próximos (dentro de 0.5%)
    def cluster(levels, tol=0.005):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for lvl in levels[1:]:
            if abs(lvl - clusters[-1][-1]) / clusters[-1][-1] <= tol:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        return [np.mean(c) for c in clusters]

    return cluster(supports), cluster(resistances)


def detect_support_break(closes, lows, volumes, supports):
    """
    Deteta quebra de suporte:
    - Preço fechou abaixo de um nível de suporte em mais de BREAK_CONFIRM_PCT
    - Volume acima da média no momento da quebra
    Retorna (True, nivel_quebrado) ou (False, None)
    """
    if not supports or len(closes) < 3:
        return False, None, False

    price    = closes[-1]
    vol_avg  = np.mean(volumes[-VOL_SMA:])
    high_vol = volumes[-1] > vol_avg * 1.5

    for lvl in sorted(supports, reverse=True):
        # Estava acima do suporte nas últimas 3 velas
        was_above = all(closes[-(i+2)] >= lvl * (1 - SR_TOLERANCE) for i in range(1, 4))
        # Agora fechou abaixo com confirmação
        broke_below = price < lvl * (1 - BREAK_CONFIRM_PCT)

        if was_above and broke_below:
            return True, lvl, high_vol

    return False, None, False


def detect_resistance_break(closes, highs, volumes, resistances):
    """Deteta quebra de resistência para cima (sinal bullish)."""
    if not resistances or len(closes) < 3:
        return False, None, False

    price    = closes[-1]
    vol_avg  = np.mean(volumes[-VOL_SMA:])
    high_vol = volumes[-1] > vol_avg * 1.5

    for lvl in sorted(resistances):
        was_below   = all(closes[-(i+2)] <= lvl * (1 + SR_TOLERANCE) for i in range(1, 4))
        broke_above = price > lvl * (1 + BREAK_CONFIRM_PCT)

        if was_below and broke_above:
            return True, lvl, high_vol

    return False, None, False


def bearish_momentum(closes, n=3):
    """Verifica se há N candles consecutivos bearish (close < open)."""
    if len(closes) < n + 1:
        return False
    return all(closes[-(i+1)] < closes[-(i+2)] for i in range(n))


def near_sr_dynamic(price, supports, resistances):
    """Verifica se está próximo de algum nível dinâmico."""
    for lvl in supports + resistances:
        if abs(price - lvl) / price <= SR_TOLERANCE:
            return True, lvl
    return False, None


# ─────────────────────────────────────────────
#  ANÁLISE PRINCIPAL
# ─────────────────────────────────────────────
def analyse():
    closes, highs, lows, vols, _ = fetch_candles(INSTRUMENT_SPOT, "15m", 100)
    if not closes or len(closes) < 30:
        return None

    # Indicadores base
    _, bbw  = calc_bb(closes, BB_PERIOD, BB_MULT)
    atr     = calc_atr(highs, lows, closes, ATR_PERIOD)
    atr_m   = sma(np.nan_to_num(atr), 20)
    vol_m   = sma(vols, VOL_SMA)
    rsi     = calc_rsi(closes, RSI_PERIOD)

    vwap                    = calc_vwap(highs, lows, closes, vols)
    ema_cross, ema_f, ema_s = calc_ema_cross(closes)
    basis, spot_p, perp_p   = fetch_funding_rate()

    price    = closes[-1]
    vwap_dev = (price - vwap) / vwap * 100

    # S/R dinâmico
    supports, resistances = find_swing_levels(highs, lows, closes)

    # Quebra de suporte (NOVO — sinal short principal)
    support_broke, broke_level, broke_high_vol = detect_support_break(
        closes, lows, vols, supports
    )
    # Quebra de resistência (sinal long)
    resist_broke, broke_resist_level, resist_high_vol = detect_resistance_break(
        closes, highs, vols, resistances
    )
    # Momentum bearish
    bearish_mom = bearish_momentum(closes, n=3)

    # Próximo de nível
    sig_sr, sr_lvl = near_sr_dynamic(price, supports, resistances)

    # Indicadores existentes
    sig_sq   = not np.isnan(bbw[-1]) and bbw[-1] < BB_SQUEEZE_PCT
    sig_atr  = not np.isnan(atr[-1]) and not np.isnan(atr_m[-1]) and atr[-1] > atr_m[-1]
    sig_vol  = not np.isnan(vol_m[-1]) and vols[-1] > vol_m[-1] * VOL_SPIKE_X
    sig_div  = rsi_divergence(closes, rsi)
    sig_vwap = abs(vwap_dev) >= VWAP_DEV_PCT
    vwap_dir = "acima" if vwap_dev > 0 else "abaixo"
    sig_ema  = ema_cross is not None

    sig_funding = False
    funding_dir = None
    if basis is not None:
        if basis >= FUNDING_BEAR:
            sig_funding = True; funding_dir = "bearish (longs sobrecarregados)"
        elif basis <= FUNDING_BULL:
            sig_funding = True; funding_dir = "bullish (shorts a ser espremidos)"

    # ── Score SHORT (sinais bearish) ──
    short_score = sum([
        support_broke,                          # quebra de suporte
        bearish_mom,                            # momentum bearish
        sig_div == "bearish",                   # divergência RSI bearish
        sig_ema and ema_cross == "bearish",     # EMA cross bearish
        sig_funding and "bearish" in (funding_dir or ""),  # funding bearish
        sig_vwap and vwap_dev < 0,              # abaixo do VWAP
        sig_vol and closes[-1] < closes[-2],    # volume com queda
        sig_atr,                                # volatilidade a subir
    ])

    # ── Score LONG (sinais bullish) ──
    long_score = sum([
        resist_broke,                           # quebra de resistência
        sig_div == "bullish",                   # divergência RSI bullish
        sig_ema and ema_cross == "bullish",     # EMA cross bullish
        sig_funding and "bullish" in (funding_dir or ""),  # funding bullish
        sig_vwap and vwap_dev > 0,              # acima do VWAP
        sig_vol and closes[-1] > closes[-2],    # volume com subida
        sig_sq,                                 # squeeze (neutro mas inclui)
        sig_sr,                                 # próximo de suporte
    ])

    score = max(short_score, long_score)
    bias  = "SHORT" if short_score > long_score else "LONG" if long_score > short_score else "NEUTRO"

    return dict(
        price=price,
        bbw=bbw[-1], atr=atr[-1], atr_m=atr_m[-1],
        vol=vols[-1], vol_m=vol_m[-1], rsi=rsi[-1],
        vwap=vwap, vwap_dev=vwap_dev, vwap_dir=vwap_dir,
        ema_fast=ema_f, ema_slow=ema_s, ema_cross=ema_cross,
        basis=basis, funding_dir=funding_dir,
        supports=supports, resistances=resistances,

        # Sinais novos
        support_broke=support_broke, broke_level=broke_level, broke_high_vol=broke_high_vol,
        resist_broke=resist_broke, broke_resist_level=broke_resist_level,
        bearish_mom=bearish_mom,

        # Sinais existentes
        sig_sq=sig_sq, sig_atr=sig_atr, sig_vol=sig_vol,
        sig_div=sig_div, sig_sr=sig_sr, sr_lvl=sr_lvl,
        sig_funding=sig_funding, sig_vwap=sig_vwap, sig_ema=sig_ema,

        short_score=short_score, long_score=long_score,
        score=score, bias=bias,
        ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


# ─────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────
def send_telegram(msg):
    if not TELEGRAM_TOKEN:
        log.warning("TELEGRAM_TOKEN nao definido.")
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
    bias  = d["bias"]
    score = d["score"]
    short_s = d["short_score"]
    long_s  = d["long_score"]

    if bias == "SHORT":
        icon = "🔴"; lvl = "SHORT"
    elif bias == "LONG":
        icon = "🟢"; lvl = "LONG"
    else:
        icon = "🟡"; lvl = "NEUTRO"

    urgency = "ALERTA FORTE" if score >= 6 else "ALERTA" if score >= 3 else "AVISO"

    def line(sig, label):
        return f"{'[X]' if sig else '[ ]'} {label}"

    # Secao S/R dinamico
    sup_txt = "  ".join([f"${s:,.0f}" for s in sorted(d["supports"])[-3:]]) or "nenhum detetado"
    res_txt = "  ".join([f"${r:,.0f}" for r in sorted(d["resistances"])[:3]]) or "nenhum detetado"

    basis_txt = (
        f"Basis perp/spot: {d['basis']:+.3f}% ({d['funding_dir']})"
        if d['basis'] is not None else "Basis perp/spot: N/A"
    )

    if score >= 6:
        interp = f"ALERTA MAXIMO {icon} — Posiciona stops. Movimento forte iminente."
    elif score >= 3:
        interp = f"Varios sinais {bias} ativos. Prepara o plano."
    else:
        interp = "Mercado calmo. Boa altura para planear entradas."

    lines = [
        f"<b>BTC ALERT v4 — {icon} {urgency} {lvl}</b>",
        f"Score: SHORT {short_s}/8  |  LONG {long_s}/8",
        f"Preco: ${d['price']:,.0f}   {d['ts']}",
        "",
        "<b>Niveis dinamicos (15min):</b>",
        f"  Suporte:    {sup_txt}",
        f"  Resistencia: {res_txt}",
        "",
        "<b>Sinais SHORT:</b>",
        line(d["support_broke"],
             f"QUEBRA SUPORTE ${d['broke_level']:,.0f}" + (" + volume alto" if d["broke_high_vol"] else "") if d["support_broke"] else "Quebra de suporte: nao"),
        line(d["bearish_mom"],   "Momentum bearish (3 velas consecutivas)"),
        line(d["sig_div"] == "bearish", f"Divergencia RSI bearish (RSI: {d['rsi']:.1f})"),
        line(d["sig_ema"] and d["ema_cross"] == "bearish", "EMA Cross bearish"),
        line(d["sig_vwap"] and d["vwap_dev"] < 0, f"Abaixo do VWAP ({d['vwap_dev']:+.1f}%)"),
        "",
        "<b>Sinais LONG:</b>",
        line(d["resist_broke"],
             f"QUEBRA RESISTENCIA ${d['broke_resist_level']:,.0f}" + (" + volume alto" if d["resist_high_vol"] else "") if d["resist_broke"] else "Quebra de resistencia: nao"),
        line(d["sig_div"] == "bullish", f"Divergencia RSI bullish (RSI: {d['rsi']:.1f})"),
        line(d["sig_ema"] and d["ema_cross"] == "bullish", "EMA Cross bullish"),
        line(d["sig_vwap"] and d["vwap_dev"] > 0, f"Acima do VWAP ({d['vwap_dev']:+.1f}%)"),
        "",
        "<b>Indicadores gerais:</b>",
        line(d["sig_sq"],  f"BB Squeeze (BBW: {d['bbw']:.2f}%)"),
        line(d["sig_atr"], f"ATR elevado (${d['atr']:.0f} vs media ${d['atr_m']:.0f})"),
        line(d["sig_vol"], f"Volume spike ({d['vol']:.0f} vs media {d['vol_m']:.0f})"),
        line(d["sig_funding"], basis_txt),
        "",
        interp,
        "",
        "Nao e conselho financeiro. Gere sempre o risco.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  LOOP PRINCIPAL
# ─────────────────────────────────────────────
def main():
    log.info("BTC Alert Bot v4 iniciado.")
    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN nao definido!")
        return

    send_telegram(
        "<b>BTC Alert Bot v4 online!</b>\n"
        "Novidades:\n"
        "- Candles 15min (mais rapido)\n"
        "- S/R dinamico automatico\n"
        "- Detetor de quebra de suporte/resistencia\n"
        "- Score separado SHORT / LONG\n"
        f"Alertas quando score >= {MIN_SCORE}/8"
    )

    last_score = 0
    last_bias  = ""
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
                f"${d['price']:,.0f} | SHORT: {d['short_score']}/8 LONG: {d['long_score']}/8 "
                f"| Bias: {d['bias']} | RSI: {d['rsi']:.1f} | "
                f"SupBreak: {d['support_broke']}"
            )

            now = time.time()
            score_changed = (d["score"] != last_score or d["bias"] != last_bias)
            cooldown_ok   = (now - last_alert > ALERT_COOLDOWN)

            # Alerta imediato se houver quebra de suporte com volume
            urgent = d["support_broke"] and d["broke_high_vol"]

            if d["score"] >= MIN_SCORE and (score_changed or cooldown_ok or urgent):
                send_telegram(build_msg(d))
                last_score = d["score"]
                last_bias  = d["bias"]
                last_alert = now

        except KeyboardInterrupt:
            log.info("Bot parado.")
            break
        except Exception as e:
            log.error(f"Erro: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
