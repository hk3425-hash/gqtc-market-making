"""
QB Market Making Dashboard  —  Analytics Engine
=================================================

Real-time dashboard computing ALL performance metrics from raw event stream.

Architecture:
  quant_bot (ZMQ PUB) --> dashboard (ZMQ SUB) --> analytics --> Dash web UI
  OR:  dashboard --file logs/events.jsonl (tail mode)

Metrics computed HERE (not in bot):
  Sharpe ratio, Sortino ratio, Calmar ratio, profit factor, expectancy,
  win rate, avg win/loss, max consecutive W/L, avg trade duration,
  PnL distribution, fee analysis, return per trade, and more.

Run:
  pip install pyzmq dash plotly pandas numpy
  python3 dashboard.py                          # live ZMQ
  python3 dashboard.py --file logs/events.jsonl  # replay/tail
  Open http://127.0.0.1:8050
"""

import argparse, json, time, math, threading
from collections import deque
from datetime import datetime, timezone
from typing import List, Dict

import numpy as np
import zmq
from dash import Dash, html, dcc, dash_table, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# CONFIG
# ============================================================

WINDOW_SEC  = 3600     # 1-hour rolling for time-series
MAX_PTS     = 60_000   # ring buffer capacity
MAX_ORDERS  = 3_000
REFRESH_MS  = 600      # UI refresh interval

# Annualization: assume ~1 tick per 0.5s → 7200 ticks/hr → ~63M ticks/yr
# We'll compute from actual elapsed time instead.


# ============================================================
# ANALYTICS ENGINE
# ============================================================

class Analytics:
    """
    Computes all performance metrics from raw fill + tick data.
    Fed by DataStore, queried by Dash callback.
    """

    @staticmethod
    def sharpe(equity_returns: np.ndarray, periods_per_year: float) -> float:
        """Annualized Sharpe ratio from a return series."""
        if len(equity_returns) < 2: return 0.0
        mu = np.mean(equity_returns)
        sigma = np.std(equity_returns, ddof=1)
        if sigma < 1e-15: return 0.0
        return (mu / sigma) * math.sqrt(periods_per_year)

    @staticmethod
    def sortino(equity_returns: np.ndarray, periods_per_year: float) -> float:
        """Annualized Sortino ratio (downside deviation only)."""
        if len(equity_returns) < 2: return 0.0
        mu = np.mean(equity_returns)
        downside = equity_returns[equity_returns < 0]
        if len(downside) < 1: return float('inf') if mu > 0 else 0.0
        dd = np.sqrt(np.mean(downside ** 2))
        if dd < 1e-15: return 0.0
        return (mu / dd) * math.sqrt(periods_per_year)

    @staticmethod
    def calmar(total_return: float, max_drawdown: float) -> float:
        """Calmar ratio = annualized return / max drawdown."""
        if max_drawdown < 1e-15: return 0.0
        return total_return / max_drawdown

    @staticmethod
    def trade_metrics(fills: List[dict]) -> dict:
        """Compute trade-level analytics from fill events."""
        if not fills:
            return {
                "num_trades": 0, "num_round_trips": 0,
                "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "profit_factor": 0.0, "expectancy": 0.0,
                "max_consec_wins": 0, "max_consec_losses": 0,
                "avg_trade_duration_s": 0.0, "median_trade_duration_s": 0.0,
                "avg_pnl_per_trade": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
                "pnl_values": [], "trade_durations": [],
                "buy_fills": 0, "sell_fills": 0,
                "total_fees": 0.0, "fee_pct_of_volume": 0.0,
            }

        # Separate by side
        buys = [f for f in fills if f["side"] == "buy"]
        sells = [f for f in fills if f["side"] == "sell"]
        closing = [f for f in fills if f.get("is_closing", False)]
        pnl_vals = [f["trade_pnl"] for f in closing]
        fees = [f.get("fee", 0) for f in fills]
        total_fees = sum(fees)
        total_vol = sum(f.get("cum_volume", 0) for f in fills[-1:])  # last fill has cum

        wins = [p for p in pnl_vals if p > 0]
        losses = [p for p in pnl_vals if p <= 0]

        # Consecutive W/L
        max_cw, max_cl, cw, cl = 0, 0, 0, 0
        for p in pnl_vals:
            if p > 0:
                cw += 1; cl = 0; max_cw = max(max_cw, cw)
            else:
                cl += 1; cw = 0; max_cl = max(max_cl, cl)

        # Trade durations (pair buys → sells)
        durations = []
        buy_times = deque()
        for f in fills:
            if f["side"] == "buy":
                buy_times.append(f["_ts"])
            elif f["side"] == "sell" and buy_times:
                bt = buy_times.popleft()
                durations.append(f["_ts"] - bt)

        gp = sum(wins) if wins else 0.0
        gl = abs(sum(losses)) if losses else 0.0
        wr = len(wins) / len(pnl_vals) * 100 if pnl_vals else 0.0
        avg_w = np.mean(wins) if wins else 0.0
        avg_l = np.mean(losses) if losses else 0.0
        pf = gp / gl if gl > 1e-15 else 0.0
        exp = (wr/100 * avg_w) + ((1 - wr/100) * avg_l) if pnl_vals else 0.0

        return {
            "num_trades": len(fills),
            "num_round_trips": len(closing),
            "win_rate": wr,
            "avg_win": avg_w,
            "avg_loss": avg_l,
            "profit_factor": pf,
            "expectancy": exp,
            "gross_profit": gp,
            "gross_loss": gl,
            "max_consec_wins": max_cw,
            "max_consec_losses": max_cl,
            "avg_trade_duration_s": float(np.mean(durations)) if durations else 0.0,
            "median_trade_duration_s": float(np.median(durations)) if durations else 0.0,
            "avg_pnl_per_trade": float(np.mean(pnl_vals)) if pnl_vals else 0.0,
            "best_trade": max(pnl_vals) if pnl_vals else 0.0,
            "worst_trade": min(pnl_vals) if pnl_vals else 0.0,
            "pnl_values": pnl_vals,
            "trade_durations": durations,
            "buy_fills": len(buys),
            "sell_fills": len(sells),
            "total_fees": total_fees,
            "fee_pct_of_volume": (total_fees / total_vol * 100) if total_vol > 1e-15 else 0.0,
        }


# ============================================================
# DATA STORE  (thread-safe ring buffers)
# ============================================================

class DataStore:

    def __init__(self):
        self.lock = threading.Lock()

        # Tick time-series
        self.tick_ts       = deque(maxlen=MAX_PTS)
        self.mid_price     = deque(maxlen=MAX_PTS)
        self.micro_price   = deque(maxlen=MAX_PTS)
        self.fair_price    = deque(maxlen=MAX_PTS)
        self.spread        = deque(maxlen=MAX_PTS)
        self.ofi           = deque(maxlen=MAX_PTS)
        self.volatility    = deque(maxlen=MAX_PTS)
        self.trade_impulse = deque(maxlen=MAX_PTS)
        self.best_bid_qty  = deque(maxlen=MAX_PTS)
        self.best_ask_qty  = deque(maxlen=MAX_PTS)
        self.sniper_mode   = deque(maxlen=MAX_PTS)

        # Stats time-series
        self.equity        = deque(maxlen=MAX_PTS)
        self.realized_pnl  = deque(maxlen=MAX_PTS)
        self.unrealized_pnl = deque(maxlen=MAX_PTS)
        self.total_pnl     = deque(maxlen=MAX_PTS)
        self.max_drawdown  = deque(maxlen=MAX_PTS)
        self.inventory     = deque(maxlen=MAX_PTS)
        self.volume        = deque(maxlen=MAX_PTS)
        self.qty_traded    = deque(maxlen=MAX_PTS)
        self.total_fees    = deque(maxlen=MAX_PTS)

        # Fill-level data (for analytics)
        self.fills: List[dict] = []

        # Append-only order log
        self.order_log = deque(maxlen=MAX_ORDERS)

        # System messages
        self.system_log = deque(maxlen=200)

        # Snapshots
        self.latest_stats: dict = {}
        self.latest_signal: dict = {}
        self.latest_active_bids: list = []
        self.latest_active_asks: list = []
        self.connected = False
        self.msg_count = 0

    def ingest(self, e: dict):
        with self.lock:
            self.msg_count += 1
            et = e.get("type")
            ts = e.get("_ts", time.time())
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)

            if et == "tick":
                self.tick_ts.append(dt)
                for attr in ("mid_price","micro_price","fair_price","spread","ofi",
                             "volatility","trade_impulse","best_bid_qty","best_ask_qty","sniper_mode"):
                    getattr(self, attr).append(e.get(attr, 0))
                for attr in ("equity","realized_pnl","unrealized_pnl","total_pnl",
                             "max_drawdown","inventory","volume","qty_traded","total_fees"):
                    getattr(self, attr).append(e.get(attr, 0))
                # Cache latest
                self.latest_stats = {k: e.get(k, 0) for k in (
                    "equity","cash","inventory","avg_entry","unrealized_pnl","realized_pnl",
                    "total_pnl","max_drawdown","hwm","volume","total_fees","qty_traded",
                    "win_rate","fills","buy_fills","sell_fills","wins","losses",
                    "avg_win","avg_loss","gross_profit","gross_loss","profit_factor",
                    "expectancy","max_consec_wins","max_consec_losses")}
                self.latest_signal = {k: e.get(k, 0) for k in (
                    "mid_price","spread","volatility","ofi","trade_impulse","sniper_mode",
                    "best_bid_qty","best_ask_qty","active_bids","active_asks")}

            elif et == "fill":
                e["_dt"] = dt
                self.fills.append(e)
                self.order_log.append({
                    "time": dt.strftime("%H:%M:%S.%f")[:-3],
                    "event": "FILL",
                    "side": e.get("side",""),
                    "price": f"{e.get('price',0):.2f}",
                    "size": f"{e.get('size',0):.8f}",
                    "pnl": f"{e.get('trade_pnl',0):.8f}",
                    "order_id": str(e.get("order_id",""))[:12],
                })

            elif et == "order":
                self.order_log.append({
                    "time": dt.strftime("%H:%M:%S.%f")[:-3],
                    "event": e.get("event",""),
                    "side": e.get("side",""),
                    "price": f"{e.get('price',0):.2f}",
                    "size": f"{e.get('size',0):.8f}",
                    "pnl": "",
                    "order_id": str(e.get("order_id",""))[:12],
                })

            elif et == "status":
                self.latest_stats.update({k: e.get(k,0) for k in e
                                          if k not in ("type","_seq","_ts","_uptime","active_bids_list","active_asks_list")})
                # Catch the new arrays
                self.latest_active_bids = e.get("active_bids_list", [])
                self.latest_active_asks = e.get("active_asks_list", [])

            elif et == "system":
                self.system_log.append({
                    "time": dt.strftime("%H:%M:%S"),
                    "level": e.get("level","info"),
                    "message": e.get("message",""),
                })

    def get_window(self, seconds=WINDOW_SEC):
        with self.lock:
            if not self.tick_ts: return {}
            cutoff = datetime.now(timezone.utc).timestamp() - seconds
            start = 0
            for i, dt in enumerate(self.tick_ts):
                if dt.timestamp() >= cutoff: start = i; break
            def sl(d): return list(d)[start:]
            return {k: sl(getattr(self, k)) for k in (
                "tick_ts","mid_price","micro_price","fair_price","spread","ofi",
                "volatility","trade_impulse","equity","realized_pnl","total_pnl",
                "inventory","volume","qty_traded","total_fees","max_drawdown")}

    def compute_analytics(self) -> dict:
        """Compute all derived metrics from raw data."""
        with self.lock:
            fills_copy = list(self.fills)
            eq_list = list(self.equity)
            ts_list = list(self.tick_ts)

        # Trade-level metrics
        tm = Analytics.trade_metrics(fills_copy)

        # Equity return series for Sharpe/Sortino
        eq = np.array(eq_list, dtype=np.float64) if eq_list else np.array([])
        if len(eq) > 1:
            returns = np.diff(eq)
            # Estimate periods/year from actual tick timestamps
            if len(ts_list) >= 2:
                elapsed = (ts_list[-1] - ts_list[0]).total_seconds()
                ticks = len(ts_list)
                if elapsed > 0:
                    ticks_per_sec = ticks / elapsed
                    ticks_per_year = ticks_per_sec * 365.25 * 86400
                else:
                    ticks_per_year = 63e6  # fallback
            else:
                ticks_per_year = 63e6

            sharpe = Analytics.sharpe(returns, ticks_per_year)
            sortino = Analytics.sortino(returns, ticks_per_year)
        else:
            returns = np.array([])
            sharpe = 0.0
            sortino = 0.0

        total_pnl = eq[-1] if len(eq) > 0 else 0.0
        max_dd = float(np.max(np.array(list(self.max_drawdown)))) if self.max_drawdown else 0.0
        calmar = Analytics.calmar(total_pnl, max_dd)

        # Uptime
        uptime = (ts_list[-1] - ts_list[0]).total_seconds() if len(ts_list) >= 2 else 0

        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "uptime_s": uptime,
            "equity_returns": returns,
            **tm,
        }


# ============================================================
# ZMQ / JSONL FEED THREADS
# ============================================================

def zmq_sub(store: DataStore, host: str, port: int):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.connect(f"tcp://{host}:{port}")
    print(f"[ZMQ] Connected tcp://{host}:{port}")
    store.connected = True
    while True:
        try:
            store.ingest(json.loads(sock.recv_string()))
        except zmq.Again: continue
        except Exception as e: print(f"[ZMQ] {e}"); time.sleep(0.1)

def jsonl_tail(store: DataStore, path: str):
    import os
    print(f"[JSONL] Tailing {path}")
    store.connected = True
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    try: store.ingest(json.loads(line))
                    except: pass
    with open(path) as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if line and line.strip():
                try: store.ingest(json.loads(line))
                except: pass
            else: time.sleep(0.05)


# ============================================================
# DASH APP
# ============================================================

# Styles
DARK = "#0d1117"
PANEL = "#161b22"
CARD = "#1a1a2e"
BORDER = "#30363d"
TEXT = "#c9d1d9"
DIM = "#8b949e"
GREEN = "#4caf50"
RED = "#f44336"
AMBER = "#ffa726"
BLUE = "#42a5f5"


def _card(label, value, fmt=".8f", prefix="", color=TEXT):
    return html.Div([
        html.Div(label, style={"fontSize":"10px","color":DIM,"textTransform":"uppercase","letterSpacing":"0.5px"}),
        html.Div(f"{prefix}{value:{fmt}}", style={
            "fontSize":"14px","color":color,"fontFamily":"JetBrains Mono, monospace","fontWeight":"bold"}),
    ], style={"background":CARD,"padding":"6px 12px","borderRadius":"5px",
              "minWidth":"110px","border":f"1px solid {BORDER}"})


def create_app(store: DataStore) -> Dash:
    app = Dash(__name__, title="QB MM Dashboard")
    app.layout = html.Div([
        # Header
        html.Div([
            html.H2("QB Market Making Dashboard", style={"margin":"0","color":"#e0e0e0","fontSize":"18px"}),
            html.Div(id="hdr", style={"fontSize":"12px","color":DIM}),
        ], style={"padding":"8px 16px","background":CARD,"borderBottom":f"1px solid {BORDER}",
                  "display":"flex","justifyContent":"space-between","alignItems":"center"}),

        # KPI row 1: Risk-adjusted returns
        html.Div(id="kpi-risk", style={
            "display":"flex","gap":"8px","padding":"8px 16px","flexWrap":"wrap","background":PANEL}),

        # KPI row 2: Trade stats
        html.Div(id="kpi-trade", style={
            "display":"flex","gap":"8px","padding":"4px 16px 8px","flexWrap":"wrap","background":PANEL}),

        # Charts
        dcc.Graph(id="charts", style={"height":"62vh"}, config={"displayModeBar": False}),

        # PnL distribution
        dcc.Graph(id="pnl-hist", style={"height":"20vh"}, config={"displayModeBar": False}),

        # Order log
        html.Div([
            html.H4("Order History (append-only, newest first)",
                     style={"color":"#e0e0e0","margin":"8px 16px 4px","fontSize":"14px"}),
            dash_table.DataTable(
                id="orders",
                columns=[{"name":n,"id":i} for n,i in [
                    ("Time","time"),("Event","event"),("Side","side"),
                    ("Price","price"),("Size","size"),("PnL","pnl"),("OrderID","order_id")]],
                data=[], page_size=30, sort_action="native",
                style_table={"overflowX":"auto"},
                style_header={"backgroundColor":CARD,"color":"#e0e0e0","fontWeight":"bold","fontSize":"11px"},
                style_cell={"backgroundColor":PANEL,"color":"#c0c0c0","fontSize":"11px",
                            "padding":"3px 6px","border":f"1px solid {BORDER}","textAlign":"left",
                            "fontFamily":"JetBrains Mono, monospace"},
                style_data_conditional=[
                    {"if":{"filter_query":'{event} = "FILL"'},"backgroundColor":"#1b4332","color":"#52b788"},
                    {"if":{"filter_query":'{event} = "FILLED"'},"backgroundColor":"#1b4332","color":"#52b788"},
                    {"if":{"filter_query":'{event} = "PLACED"'},"backgroundColor":CARD,"color":"#90caf9"},
                    {"if":{"filter_query":'{event} = "CANCELLED"'},"backgroundColor":"#2d1b1b","color":"#ef9a9a"},
                    {"if":{"filter_query":'{event} = "STALE_CANCEL"'},"backgroundColor":"#2d2b1b","color":"#ffcc80"},
                    {"if":{"filter_query":'{event} = "REJECTED"'},"backgroundColor":"#4a1010","color":"#ef5350"},
                    {"if":{"filter_query":'{side} = "buy"',"column_id":"side"},"color":GREEN},
                    {"if":{"filter_query":'{side} = "sell"',"column_id":"side"},"color":RED},
                ],
            ),
        ]),

        # Live Active Orders
        html.Div([
            html.H4("Live Active Orders (Current Snapshot)",
                     style={"color":"#e0e0e0","margin":"8px 16px 4px","fontSize":"14px"}),
            dash_table.DataTable(
                id="active-orders",
                columns=[{"name":n,"id":i} for n,i in [
                    ("Side","side"),("Price","price"),("Size","size"),("OrderID","id")]],
                data=[],
                style_table={"overflowX":"auto"},
                style_header={"backgroundColor":CARD,"color":"#e0e0e0","fontWeight":"bold","fontSize":"11px"},
                style_cell={"backgroundColor":PANEL,"color":"#c0c0c0","fontSize":"11px",
                            "padding":"3px 6px","border":f"1px solid {BORDER}","textAlign":"left",
                            "fontFamily":"JetBrains Mono, monospace"},
                style_data_conditional=[
                    {"if":{"filter_query":'{side} = "BID"',"column_id":"side"},"color":GREEN},
                    {"if":{"filter_query":'{side} = "ASK"',"column_id":"side"},"color":RED},
                ],
            ),
        ]),

        # System log
        html.Div([
            html.H4("System Log", style={"color":"#e0e0e0","margin":"8px 16px 4px","fontSize":"14px"}),
            html.Div(id="syslog", style={
                "maxHeight":"120px","overflow":"auto","padding":"4px 16px",
                "fontSize":"11px","fontFamily":"monospace","background":DARK,"color":DIM}),
        ]),

        dcc.Interval(id="tick", interval=REFRESH_MS, n_intervals=0),
    ], style={"backgroundColor":DARK,"minHeight":"100vh","fontFamily":"Inter, sans-serif"})


    @app.callback(
        [Output("hdr","children"), Output("kpi-risk","children"), Output("kpi-trade","children"),
         Output("charts","figure"), Output("pnl-hist","figure"),
         Output("orders","data"), Output("syslog","children"),
         Output("active-orders","data")],
        [Input("tick","n_intervals")]
    )
    def update(n):
        data = store.get_window(WINDOW_SEC)
        an = store.compute_analytics()
        st = store.latest_stats
        sig = store.latest_signal

        # Header
        hdr = (f"Msgs: {store.msg_count:,}  |  "
               f"{'Connected' if store.connected else 'Disconnected'}  |  "
               f"Pts: {len(store.tick_ts):,}  |  "
               f"Uptime: {an['uptime_s']:.0f}s")

        # KPI Row 1: Risk-adjusted
        pnl = st.get("total_pnl", 0)
        pnl_c = GREEN if pnl >= 0 else RED
        sharpe_c = GREEN if an["sharpe"] > 0 else RED if an["sharpe"] < 0 else TEXT
        sortino_c = GREEN if an["sortino"] > 0 else RED if an["sortino"] < 0 else TEXT

        kpi1 = [
            _card("Total PnL", pnl, prefix="$", color=pnl_c),
            _card("Equity", st.get("equity",0), prefix="$"),
            _card("Sharpe", an["sharpe"], fmt=".4f", color=sharpe_c),
            _card("Sortino", an["sortino"], fmt=".4f", color=sortino_c),
            _card("Calmar", an["calmar"], fmt=".4f"),
            _card("Max DD", st.get("max_drawdown",0), prefix="$", color=RED),
            _card("Inventory", st.get("inventory",0)),
            _card("Avg Entry", st.get("avg_entry",0), fmt=".2f"),
            _card("Volatility", sig.get("volatility",0)),
            _card("Mid", sig.get("mid_price",0), fmt=".2f"),
        ]

        # KPI Row 2: Trade stats
        kpi2 = [
            _card("Trades", an["num_trades"], fmt=".0f"),
            _card("Round Trips", an["num_round_trips"], fmt=".0f"),
            _card("Win Rate", an["win_rate"], fmt=".1f", color=GREEN if an["win_rate"]>50 else AMBER),
            _card("Profit Factor", an["profit_factor"], fmt=".3f"),
            _card("Expectancy", an["expectancy"], prefix="$"),
            _card("Avg Win", an["avg_win"], prefix="$", color=GREEN),
            _card("Avg Loss", an["avg_loss"], prefix="$", color=RED),
            _card("Best Trade", an["best_trade"], prefix="$", color=GREEN),
            _card("Worst Trade", an["worst_trade"], prefix="$", color=RED),
            _card("Avg Duration", an["avg_trade_duration_s"], fmt=".1f"),
            _card("Max W Streak", an["max_consec_wins"], fmt=".0f"),
            _card("Max L Streak", an["max_consec_losses"], fmt=".0f"),
            _card("Fees", an["total_fees"], prefix="$"),
            _card("Fee %Vol", an["fee_pct_of_volume"], fmt=".3f"),
            _card("Volume", st.get("volume",0), fmt=".2f", prefix="$"),
            _card("Qty Traded", st.get("qty_traded",0)),
        ]

        # ---- Time-series charts (4x2) ----
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=("Mid / Fair Price","Spread",
                            "OFI","Volatility",
                            "PnL (Realized + Total)","Inventory",
                            "Cumulative Volume ($)","Max Drawdown ($)"),
            vertical_spacing=0.06, horizontal_spacing=0.06)

        ts = data.get("tick_ts", [])
        if ts:
            fig.add_trace(go.Scattergl(x=ts,y=data["mid_price"],name="Mid",
                          line=dict(width=1.5,color=BLUE)),row=1,col=1)
            fig.add_trace(go.Scattergl(x=ts,y=data["fair_price"],name="Fair",
                          line=dict(width=1,color=AMBER,dash="dot")),row=1,col=1)

            fig.add_trace(go.Scattergl(x=ts,y=data["spread"],name="Spread",
                          line=dict(width=1,color="#ab47bc")),row=1,col=2)

            fig.add_trace(go.Scattergl(x=ts,y=data["ofi"],name="OFI",
                          line=dict(width=1,color="#66bb6a")),row=2,col=1)

            fig.add_trace(go.Scattergl(x=ts,y=data["volatility"],name="Vol",
                          line=dict(width=1,color=RED)),row=2,col=2)

            fig.add_trace(go.Scattergl(x=ts,y=data["total_pnl"],name="TotalPnL",
                          line=dict(width=1.5,color=GREEN)),row=3,col=1)
            fig.add_trace(go.Scattergl(x=ts,y=data["realized_pnl"],name="RPnL",
                          line=dict(width=1,color="#81c784",dash="dot")),row=3,col=1)
            fig.add_hline(y=0,row=3,col=1,line_dash="dash",line_color="#555",line_width=0.5)

            fig.add_trace(go.Scattergl(x=ts,y=data["inventory"],name="Inv",
                          line=dict(width=1.5,color="#29b6f6")),row=3,col=2)
            fig.add_hline(y=0,row=3,col=2,line_dash="dash",line_color="#555",line_width=0.5)

            fig.add_trace(go.Scattergl(x=ts,y=data["volume"],name="CumVol$",
                          line=dict(width=1,color=AMBER)),row=4,col=1)

            fig.add_trace(go.Scattergl(x=ts,y=data["max_drawdown"],name="MaxDD$",
                          line=dict(width=1.5,color=RED)),row=4,col=2)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor=DARK, plot_bgcolor=PANEL,
            font=dict(size=10,color=TEXT), showlegend=True,
            legend=dict(orientation="h",y=1.02,x=0.5,xanchor="center",font=dict(size=9)),
            margin=dict(l=50,r=20,t=35,b=25), height=650,
            uirevision="constant")
        fig.update_xaxes(gridcolor="#21262d",zerolinecolor="#333")
        fig.update_yaxes(gridcolor="#21262d",zerolinecolor="#333")

        # ---- PnL Distribution ----
        pnl_fig = go.Figure()
        pnl_vals = an.get("pnl_values", [])
        if pnl_vals:
            colors = [GREEN if v > 0 else RED for v in pnl_vals]
            pnl_fig.add_trace(go.Histogram(
                x=pnl_vals, nbinsx=min(50, max(10, len(pnl_vals)//2)),
                marker_color=BLUE, opacity=0.8, name="PnL/Trade"))
            pnl_fig.add_vline(x=0, line_dash="dash", line_color="#888")
            if len(pnl_vals) > 1:
                pnl_fig.add_vline(x=float(np.mean(pnl_vals)), line_color=AMBER,
                                  annotation_text=f"μ={np.mean(pnl_vals):.8f}")
        pnl_fig.update_layout(
            title="PnL Distribution (per closing trade)",
            template="plotly_dark", paper_bgcolor=DARK, plot_bgcolor=PANEL,
            font=dict(size=10,color=TEXT),
            margin=dict(l=50,r=20,t=35,b=25), height=180,
            xaxis_title="PnL ($)", yaxis_title="Count",
            uirevision="pnl_hist")

        # ---- Order log (newest first) ----
        with store.lock:
            order_data = list(reversed(store.order_log))

        # ---- System log ----
        with store.lock:
            sys_entries = list(reversed(store.system_log))
        syslog = [html.Div(
            f"[{e['time']}] [{e['level'].upper()}] {e['message']}",
            style={"color": {"error":RED,"warn":AMBER}.get(e.get("level"), DIM)}
        ) for e in sys_entries[:50]]

        active_data = []
        # Asks: reversed so the lowest ask is immediately above the highest bid
        for ask in reversed(store.latest_active_asks):
            active_data.append({
                "side": "ASK",
                "price": f"{float(ask.get('price', 0)):.2f}",
                "size": f"{float(ask.get('size', 0)):.8f}",
                "id": str(ask.get('id', ''))[:12]
            })

        for bid in store.latest_active_bids:
            active_data.append({
                "side": "BID",
                "price": f"{float(bid.get('price', 0)):.2f}",
                "size": f"{float(bid.get('size', 0)):.8f}",
                "id": str(bid.get('id', ''))[:12]
            })

        return hdr, kpi1, kpi2, fig, pnl_fig, order_data, syslog, active_data

    return app


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="QB MM Dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--file", default=None, help="JSONL file instead of ZMQ")
    ap.add_argument("--dash-port", type=int, default=8050)
    ap.add_argument("--window", type=int, default=3600)
    args = ap.parse_args()

    global WINDOW_SEC
    WINDOW_SEC = args.window

    store = DataStore()
    t = threading.Thread(
        target=jsonl_tail if args.file else zmq_sub,
        args=(store, args.file) if args.file else (store, args.host, args.port),
        daemon=True)
    t.start()

    app = create_app(store)
    print(f"\n  Dashboard: http://127.0.0.1:{args.dash_port}\n")
    app.run(host="0.0.0.0", port=args.dash_port, debug=False)

if __name__ == "__main__":
    main()
