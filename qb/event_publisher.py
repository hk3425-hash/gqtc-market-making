from pathlib import Path
import json
import time
import zmq
import zmq.asyncio

class EventPublisher:
    """
    Publishes structured JSON events to ZMQ + JSONL.
    Event types: tick, fill, order, status, system.
    All heavy analytics (Sharpe, Sortino, etc.) are computed
    in dashboard.py from these raw events.
    """

    def __init__(self, port: int, path: str, logger):
        self.logger = logger
        self._t0 = time.time()
        self._seq = 0
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(path, "a", buffering=1)
        self._sock = None
        ctx = zmq.asyncio.Context.instance()
        self._sock = ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 10000)
        self._sock.bind(f"tcp://127.0.0.1:{port}")
        logger.info(f"ZMQ PUB on tcp://127.0.0.1:{port}")

    def close(self):
        if self._jsonl and not self._jsonl.closed:
            self._jsonl.close()
        if self._sock:
            self._sock.close()

    def _emit(self, evt: dict):
        if not self._jsonl or self._jsonl.closed:
            return

        self._seq += 1
        evt["_seq"] = self._seq
        evt["_ts"] = time.time()
        evt["_uptime"] = time.time() - self._t0
        line = json.dumps(evt, default=str)
        self._jsonl.write(line + "\n")
        if self._sock:
            try:
                self._sock.send_string(line, zmq.NOBLOCK)
            except zmq.Again:
                pass

    # -- tick: full signal + stats snapshot --
    def tick(self, sig_dict: dict, stats_dict: dict, n_bids: int, n_asks: int, n_actions: int):
        self._emit({
            "type": "tick",
            **sig_dict,
            **stats_dict,
            "active_bids": n_bids,
            "active_asks": n_asks,
            "actions_count": n_actions,
        })

    # -- fill: per-trade PnL from C++ FillResult --
    def fill(self, side: str, price: float, size: float,
             order_id: str, fill_result_dict: dict):
        self._emit({
            "type": "fill",
            "side": side,
            "price": price,
            "size": size,
            "order_id": order_id,
            **fill_result_dict
        })

    # -- order: append-only (PLACED, FILLED, CANCELLED, etc.) --
    def order(self, event: str, oid: str, side: str,
              price: float, size: float, **extra):
        self._emit({
            "type": "order", "event": event,
            "order_id": oid, "side": side,
            "price": price, "size": size, **extra,
        })

    # -- status: periodic summary --
    def status(self, stats: dict, sig, active_bids: dict, active_asks: dict):
        # Convert order objects into dictionaries for JSON serialization
        bids_list = [{"price": o.price, "size": o.size, "id": o.id} for o in active_bids.values()]
        asks_list = [{"price": o.price, "size": o.size, "id": o.id} for o in active_asks.values()]

        self._emit({
            "type": "status", **stats,
            "mid_price": sig.mid_price if sig else 0,
            "spread": sig.spread if sig else 0,
            "active_bids": len(bids_list), "active_asks": len(asks_list),
            "active_bids_list": bids_list, "active_asks_list": asks_list, # Send to dashboard
        })

    # -- system: lifecycle events --
    def system(self, msg: str, level: str = "info"):
        self._emit({"type": "system", "message": msg, "level": level})
