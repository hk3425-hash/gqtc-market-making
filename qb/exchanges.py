import aiohttp
import asyncio
import base64
import hashlib
import hmac
import json
import time


class GeminiExchange:
    def __init__(self, api_key, api_secret, logger, sandbox=True):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.logger = logger
        self._last_nonce = 0.0
        self.session = None
        base = "api.sandbox.gemini.com" if sandbox else "api.gemini.com"
        self.base_url = f"https://{base}"
        self.ws_md = f"wss://{base}/v2/marketdata"
        self.ws_ord = f"wss://{base}/v1/order/events"
        self.ws_ready = asyncio.Event()

        # Populated by fetch_symbol_details()
        self.quote_increment = None   # price tick (e.g. 0.01)
        self.qty_increment = None     # lot size (e.g. 1e-8)
        self.min_order_size = None

    async def connect(self):
        if not self.session: self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session: await self.session.close(); self.session = None

    async def fetch_symbol_details(self, symbol):
        """Fetch quote_increment (price tick) and tick_size (qty increment)
        from Gemini REST API. Must be called after connect()."""
        url = f"{self.base_url}/v1/symbols/details/{symbol}"
        try:
            async with self.session.get(url) as resp:
                data = await resp.json()

            # Gemini naming (confusing):
            #   "quote_increment" = price precision (e.g. 0.01 for USD)
            #   "tick_size"       = quantity precision (e.g. 1e-8 for BTC)
            self.quote_increment = float(data["quote_increment"])
            self.qty_increment = float(data["tick_size"])
            self.min_order_size = float(data.get("min_order_size", 0))

            self.logger.info(
                f"Symbol details | quote_increment={self.quote_increment} "
                f"(qty_increment={self.qty_increment} "
                f"(min_order_size={self.min_order_size}")

            return {
                "quote_increment": self.quote_increment,
                "qty_increment": self.qty_increment,
                "min_order_size": self.min_order_size,
            }
        except Exception as e:
            self.logger.error(f"fetch_symbol_details: {e}")
            return None

    def _nonce(self):
        """Monotonic nonce in seconds (float) — compatible with Gemini's
        time-based nonce option which requires ±30s of current epoch."""
        n = time.time()
        if n <= self._last_nonce:
            n = self._last_nonce + 0.000001
        self._last_nonce = n
        return n

    def _sign(self, payload, *, rest=False):
        payload["nonce"] = self._nonce()
        b64 = base64.b64encode(json.dumps(payload).encode())
        sig = hmac.new(self.api_secret, b64, hashlib.sha384).hexdigest()
        headers = {
            "X-GEMINI-APIKEY": self.api_key,
            "X-GEMINI-PAYLOAD": b64.decode(),
            "X-GEMINI-SIGNATURE": sig,
        }
        if rest:
            headers["Content-Type"] = "text/plain"
            headers["Content-Length"] = "0"
            headers["Cache-Control"] = "no-cache"
        return headers

    async def market_stream(self, symbol):
        while True:
            try:
                async with self.session.ws_connect(self.ws_md) as ws:
                    self.logger.info(f"WS MarketData: {self.ws_md}")
                    await ws.send_json({"type": "subscribe", "subscriptions": [
                        {"name": "l2", "symbols": [symbol]},
                        {"name": "trade", "symbols": [symbol]},
                    ]})
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT: yield msg.json()
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED): break
            except Exception as e:
                self.logger.warn(f"MarketData WS: {e}. Reconnect 3s..."); await asyncio.sleep(3)

    async def order_stream(self):
        while True:
            try:
                headers = self._sign({"request": "/v1/order/events"})
                async with self.session.ws_connect(self.ws_ord, headers=headers) as ws:
                    self.logger.info("WS OrderEvents connected")
                    self.ws_ready.set()

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT: yield msg.json()
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED): break
            except Exception as e:
                self.ws_ready.clear()
                self.logger.warn(f"OrderEvents WS: {e}. Reconnect 3s..."); await asyncio.sleep(3)

    async def place_order(self, symbol, side, price, size, options=None):
        await self.ws_ready.wait()

        payload = {
            "request": "/v1/order/new",
            "symbol": symbol,
            "amount": size,
            "price": price,
            "side": side,
            "type": "exchange limit",
            "options": options or ["maker-or-cancel"],
        }
        try:
            async with self.session.post(f"{self.base_url}/v1/order/new",
                                         headers=self._sign(payload, rest=True)) as resp:
                data = await resp.json()
                if "order_id" in data:
                    return str(data["order_id"])

                # Improved error logging to see exactly why an order is rejected
                self.logger.warn(f"Rejected: {data.get('reason', data.get('message', data))}")
                return None
        except Exception as e:
            self.logger.error(f"place_order: {e}")
            return None

    async def cancel_order(self, order_id):
        await self.ws_ready.wait()

        payload = {"request": "/v1/order/cancel", "order_id": order_id}
        try:
            async with self.session.post(f"{self.base_url}/v1/order/cancel",
                                         headers=self._sign(payload, rest=True)) as resp:
                return (await resp.json()).get("is_cancelled", False)
        except Exception as e:
            self.logger.error(f"cancel_order: {e}"); return False


class MockExchange:
    def __init__(self, logger, start_price=100000.0):
        self.logger = logger; self.price = start_price; self._ctr = 0
        self.quote_increment = 0.01
        self.qty_increment = 0.00000001
        self.min_order_size = 0.00001
    async def connect(self): pass
    async def close(self): pass

    async def fetch_symbol_details(self, symbol):
        """Mock: return sensible defaults."""
        return {
            "quote_increment": self.quote_increment,
            "qty_increment": self.qty_increment,
            "min_order_size": self.min_order_size,
        }

    async def market_stream(self, symbol):
        import random
        while True:
            await asyncio.sleep(0.1)
            self.price += random.gauss(0, 0.5)
            sp = 0.01; b = round(self.price - sp/2, 2); a = round(self.price + sp/2, 2)
            yield {"type": "l2_updates", "changes": [
                ["buy", str(b), str(round(random.uniform(0.1,5),4))],
                ["buy", str(b-0.01), str(round(random.uniform(0.1,3),4))],
                ["buy", str(b-0.02), str(round(random.uniform(0.1,2),4))],
                ["sell", str(a), str(round(random.uniform(0.1,5),4))],
                ["sell", str(a+0.01), str(round(random.uniform(0.1,3),4))],
                ["sell", str(a+0.02), str(round(random.uniform(0.1,2),4))],
            ]}
            if random.random() < 0.3:
                yield {"type": "trade", "events": [
                    {"side": random.choice(["buy","sell"]),
                     "amount": str(round(random.uniform(0.001,0.1),4))}
                ]}

    async def order_stream(self):
        while True: await asyncio.sleep(60); yield {}

    async def place_order(self, symbol, side, price, size, options=None):
        self._ctr += 1; oid = f"mock_{self._ctr}"
        self.logger.info(f"MOCK FILL {side} {size:.8f} @ {price:.6f} ({oid})")
        return oid

    async def cancel_order(self, oid): return True
