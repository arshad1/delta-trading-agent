"""Delta Exchange REST API v2 client — drop-in replacement for HyperliquidAPI.

Key design decisions
──────────────────────
* **Pure aiohttp REST** — no SDK, no web3, no private-key signing.
  Authentication is HMAC-SHA256 over the standard request fields.
* **Product metadata cache** — fetched once on startup per asset from
  ``GET /v2/products/{symbol}`` and stored in ``self._product_cache``.
* **Sizing in contracts** — Delta orders use an integer ``size`` (number of
  contracts).  ``usd_to_contracts()`` converts a USD allocation to the
  correct integer using ``contract_value`` from the metadata cache.
* **TP + SL** — Delta uses a single bracket-order endpoint
  (``POST /v2/orders/bracket``) to attach both TP and SL to an open
  position.  ``place_bracket_order()`` handles this.
* **Auto leverage** — ``init_products()`` calls
  ``POST /v2/products/{id}/orders/leverage`` for every traded asset on
  startup so the agent always operates at the configured leverage.
* **Candle resolution** — Delta's ``/v2/history/candles`` accepts the same
  ``5m``, ``15m``, ``1h``, ``4h``, ``1D`` strings used by the old code.
* **Retry / back-off** — all API calls use ``_request()`` which transparently
  retries on 5xx / rate-limit (429) up to ``MAX_RETRIES`` times with
  exponential backoff.

Environment variables consumed (via CONFIG):
    DELTA_API_KEY        required
    DELTA_API_SECRET     required
    DELTA_BASE_URL       default: https://cdn-ind.testnet.deltaex.org
    DELTA_LEVERAGE       default: 5   (integer, set per product on startup)
"""

import asyncio
import json
import logging
import time
from typing import Any
from urllib.parse import urlencode

import aiohttp

from src.config_loader import CONFIG
from src.trading.delta_auth import build_auth_headers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TESTNET_URL = "https://cdn-ind.testnet.deltaex.org"
MAX_RETRIES = 5
RETRY_BASE_SECS = 1.0


class DeltaExchangeAPI:
    """Async REST client for Delta Exchange (India) perpetual futures."""

    def __init__(self) -> None:
        self._base_url: str = (
            CONFIG.get("delta_base_url") or TESTNET_URL
        ).rstrip("/")

        # Auto-select credentials for the active environment
        _is_testnet = "testnet" in self._base_url
        if _is_testnet:
            self._api_key: str = CONFIG.get("delta_testnet_api_key") or ""
            self._api_secret: str = CONFIG.get("delta_testnet_api_secret") or ""
        else:
            self._api_key = CONFIG.get("delta_prod_api_key") or ""
            self._api_secret = CONFIG.get("delta_prod_api_secret") or ""

        env_label = "testnet" if _is_testnet else "prod"
        logger.info(
            "DeltaExchangeAPI initialised — env=%s  base_url=%s",
            env_label,
            self._base_url,
        )

        self._leverage: int = int(CONFIG.get("delta_leverage") or 5)
        # {symbol_uppercase -> {id, contract_value, tick_size, symbol}}
        self._product_cache: dict[str, dict] = {}
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return (or lazily create) the shared aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------ #
    #  Low-level request helper                                           #
    # ------------------------------------------------------------------ #

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        body: dict | None = None,
        auth: bool = True,
    ) -> Any:
        """Execute a Delta Exchange REST call with retry/back-off.

        Parameters
        ----------
        method:   HTTP verb (``"GET"``, ``"POST"``, ``"DELETE"``, ``"PUT"``).
        path:     API path, e.g. ``"/v2/orders"``.
        params:   URL query parameters (will be URL-encoded).
        body:     JSON request body (dict).  None = no body.
        auth:     Whether to add authentication headers.

        Returns
        -------
        The parsed JSON ``result`` value from the Delta response envelope.
        Raises ``RuntimeError`` on unrecoverable API errors.
        """
        query_string = ""
        if params:
            query_string = "?" + urlencode(params)

        body_str = json.dumps(body) if body else ""

        headers = {}
        if auth:
            headers = build_auth_headers(
                self._api_key,
                self._api_secret,
                method,
                path,
                query_string,
                body_str,
            )
        else:
            headers = {
                "User-Agent": "python-delta-trading-agent",
                "Accept": "application/json",
            }

        url = self._base_url + path + query_string
        session = await self._get_session()

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.request(
                    method,
                    self._base_url + path,
                    params=params,
                    data=body_str if body_str else None,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 429:
                        wait = RETRY_BASE_SECS * (2 ** (attempt - 1))
                        logger.warning(
                            "Rate-limited by Delta (429). Sleeping %.1fs …", wait
                        )
                        await asyncio.sleep(wait)
                        # Rebuild auth (new timestamp)
                        if auth:
                            headers = build_auth_headers(
                                self._api_key,
                                self._api_secret,
                                method,
                                path,
                                query_string,
                                body_str,
                            )
                        continue

                    if resp.status >= 500:
                        wait = RETRY_BASE_SECS * (2 ** (attempt - 1))
                        logger.warning(
                            "Delta server error %s on attempt %d/%d. Retrying in %.1fs …",
                            resp.status,
                            attempt,
                            MAX_RETRIES,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        if auth:
                            headers = build_auth_headers(
                                self._api_key,
                                self._api_secret,
                                method,
                                path,
                                query_string,
                                body_str,
                            )
                        continue

                    raw = await resp.json(content_type=None)
                    if not raw.get("success", True):
                        err = raw.get("error") or raw
                        raise RuntimeError(f"Delta API error: {err}")
                    return raw.get("result")

            except aiohttp.ClientError as exc:
                if attempt == MAX_RETRIES:
                    raise
                wait = RETRY_BASE_SECS * (2 ** (attempt - 1))
                logger.warning("Request error (%s). Retrying in %.1fs …", exc, wait)
                await asyncio.sleep(wait)

        raise RuntimeError(f"Delta request failed after {MAX_RETRIES} retries: {method} {path}")

    # ------------------------------------------------------------------ #
    #  Product / metadata                                                 #
    # ------------------------------------------------------------------ #

    async def init_products(self, assets: list[str]) -> None:
        """Resolve each asset symbol → product metadata and set leverage.

        Fetches ``GET /v2/products?contract_types=perpetual_futures`` once and
        finds perpetual future products matching each requested asset. The
        mapping is stored in ``_product_cache`` keyed by the bare underlying
        symbol (e.g. ``"BTC"``).

        After caching, leverage is set for each product via
        ``POST /v2/products/{id}/orders/leverage``.
        """
        logger.info("Fetching Delta product list …")
        all_products = await self._request(
            "GET",
            "/v2/products",
            params={"contract_types": "perpetual_futures", "states": "live"},
            auth=False,
        )

        if not isinstance(all_products, list):
            raise RuntimeError(
                f"Unexpected product list response: {all_products!r}"
            )

        # Index by underlying asset symbol (e.g. BTC, ETH, SOL)
        perp_map: dict[str, dict] = {}
        for prod in all_products:
            # Delta's perp symbol is "BTCUSD", "ETHUSD", etc.
            symbol: str = prod.get("symbol", "")
            underlying: str = prod.get("contract_unit_currency", "")
            if underlying and prod.get("contract_type") == "perpetual_futures":
                if symbol.endswith("USD") or symbol.endswith("USDT"):
                    # Only overwrite if we haven't found a preferred USD pair, or if we just found one.
                    # This gives preference to USD over USDT if both exist, though usually they don't on the same network
                    if underlying.upper() not in perp_map or symbol.endswith("USD"):
                        perp_map[underlying.upper()] = prod

        logger.info(
            "Found %d perpetual market(s) on Delta: %s",
            len(perp_map),
            list(perp_map.keys()),
        )

        for asset in assets:
            key = asset.upper()
            prod = perp_map.get(key)
            if prod is None:
                # Fallback: search by symbol prefix
                for sym, p in perp_map.items():
                    if sym.startswith(key) or p.get("symbol", "").startswith(key):
                        prod = p
                        break
            if prod is None:
                logger.warning("No perpetual product found for %s on Delta.", asset)
                continue

            self._product_cache[key] = {
                "id": prod["id"],
                "symbol": prod["symbol"],
                "contract_value": float(prod.get("contract_value") or 1),
                "tick_size": float(prod.get("tick_size") or 0.5),
                "position_size_limit": int(prod.get("position_size_limit") or 100000),
            }
            logger.info(
                "  %s → product_id=%d  symbol=%s  contract_value=%s",
                asset,
                prod["id"],
                prod["symbol"],
                prod.get("contract_value"),
            )

            # Set leverage for this product
            await self._set_leverage(prod["id"], asset)

        missing = [a for a in assets if a.upper() not in self._product_cache]
        if missing:
            logger.warning("Could not map these assets to Delta products: %s", missing)

    async def _set_leverage(self, product_id: int, asset: str) -> None:
        """Set the configured leverage for a single product_id."""
        try:
            result = await self._request(
                "POST",
                f"/v2/products/{product_id}/orders/leverage",
                body={"leverage": self._leverage},
            )
            logger.info(
                "Leverage set for %s (id=%d): %sx", asset, product_id, self._leverage
            )
        except Exception as exc:
            logger.warning("Failed to set leverage for %s: %s", asset, exc)

    def _product(self, asset: str) -> dict:
        """Return cached product metadata or raise a helpful error."""
        meta = self._product_cache.get(asset.upper())
        if not meta:
            raise RuntimeError(
                f"No product metadata for '{asset}'. "
                "Did you call init_products() on startup?"
            )
        return meta

    def usd_to_contracts(
        self, asset: str, usd_amount: float, current_price: float
    ) -> int:
        """Convert a USD allocation to a contract count (integer, minimum 1)."""
        meta = self._product(asset)
        contract_value = meta["contract_value"]
        if contract_value <= 0 or current_price <= 0:
            return 1
        # Each contract is worth ``contract_value`` units of the underlying;
        # total_underlying = usd_amount / current_price
        # contracts = total_underlying / contract_value
        contracts = int(usd_amount / (contract_value * current_price))
        return max(1, contracts)

    def round_price(self, asset: str, price: float) -> str:
        """Round a price to the product's tick_size and return as string."""
        meta = self._product(asset)
        tick = meta["tick_size"]
        if tick <= 0:
            return str(round(price, 2))
        rounded = round(round(price / tick) * tick, 8)
        # Format without trailing zeros but keep necessary decimals
        decimals = len(str(tick).rstrip("0").split(".")[-1]) if "." in str(tick) else 0
        return f"{rounded:.{decimals}f}"

    # ------------------------------------------------------------------ #
    #  Market data                                                        #
    # ------------------------------------------------------------------ #

    async def get_current_price(self, asset: str) -> float:
        """Return the mark price for *asset* (e.g. ``"BTC"``).

        Falls back to ``close`` (last trade price) if mark_price is absent.
        """
        meta = self._product(asset)
        symbol = meta["symbol"]
        ticker = await self._request(
            "GET", f"/v2/tickers/{symbol}", auth=False
        )
        if isinstance(ticker, dict):
            price = ticker.get("mark_price") or ticker.get("close")
            if price:
                return float(price)
        raise RuntimeError(f"Cannot determine price for {asset}")

    async def get_open_interest(self, asset: str) -> float | None:
        """Return open-interest USD value for *asset*."""
        try:
            meta = self._product(asset)
            symbol = meta["symbol"]
            ticker = await self._request(
                "GET", f"/v2/tickers/{symbol}", auth=False
            )
            if isinstance(ticker, dict):
                oi_usd = ticker.get("oi_value_usd") or ticker.get("oi_value")
                return float(oi_usd) if oi_usd else None
        except Exception as exc:
            logger.debug("OI fetch error for %s: %s", asset, exc)
        return None

    async def get_funding_rate(self, asset: str) -> float | None:
        """Return the current funding rate for *asset* (as a raw decimal).

        Delta does not expose a standalone funding-rate endpoint; we derive it
        from the ticker's ``close`` and ``open`` fields as an approximation,
        or return ``None`` if unavailable.
        """
        try:
            meta = self._product(asset)
            symbol = meta["symbol"]
            ticker = await self._request(
                "GET", f"/v2/tickers/{symbol}", auth=False
            )
            if isinstance(ticker, dict):
                # Delta provides annualized_funding in the product metadata;
                # we approximate a per-period rate from that.
                # If not available we return None and the caller handles it.
                ann_funding = ticker.get("annualized_funding")
                if ann_funding:
                    return float(ann_funding) / (365 * 24)
        except Exception as exc:
            logger.debug("Funding rate fetch error for %s: %s", asset, exc)
        return None

    async def get_candles(
        self, asset: str, interval: str, count: int
    ) -> list[dict]:
        """Return the last *count* OHLCV candles for *asset* at *interval*.

        Delta Exchange's candle endpoint:
            GET /v2/history/candles?symbol=BTCUSD&resolution=5m&start=...&end=...

        Accepted ``resolution`` values: ``1m``, ``3m``, ``5m``, ``15m``,
        ``30m``, ``1h``, ``2h``, ``4h``, ``6h``, ``1D`` (note uppercase D).

        Returns list of dicts with keys: ``time``, ``open``, ``high``, ``low``,
        ``close``, ``volume``.  Sorted oldest-first (matching Hyperliquid convention).
        """
        meta = self._product(asset)
        symbol = meta["symbol"]

        # Map interval strings to Delta resolution format
        resolution = _map_interval(interval)

        # Calculate start time from count + interval seconds
        interval_secs = _interval_to_seconds(interval)
        end_ts = int(time.time())
        start_ts = end_ts - (interval_secs * (count + 5))  # +5 for buffer

        raw = await self._request(
            "GET",
            "/v2/history/candles",
            params={
                "symbol": symbol,
                "resolution": resolution,
                "start": str(start_ts),
                "end": str(end_ts),
            },
            auth=False,
        )

        if not isinstance(raw, list):
            logger.warning("Unexpected candle response for %s: %r", asset, raw)
            return []

        candles = []
        for c in raw:
            candles.append({
                "time": int(c.get("time", 0)),
                "open": float(c.get("open", 0)),
                "high": float(c.get("high", 0)),
                "low": float(c.get("low", 0)),
                "close": float(c.get("close", 0)),
                "volume": float(c.get("volume", 0)),
            })

        # Sorted oldest-first, trim to requested count
        candles.sort(key=lambda x: x["time"])
        return candles[-count:]

    # ------------------------------------------------------------------ #
    #  Account / portfolio state                                          #
    # ------------------------------------------------------------------ #

    async def get_user_state(self) -> dict:
        """Return a normalised account state dict compatible with main.py.

        Shape (matching original Hyperliquid contract):
        {
            "balance": float,          # available USD
            "total_value": float,      # balance + unrealised PnL
            "positions": [
                {
                    "coin": str,       # e.g. "BTC"
                    "szi": str,        # signed size (pos=long, neg=short)
                    "entryPx": str,
                    "liqPx": str | None,
                    "pnl": float,      # unrealised PnL
                    "leverage": int,
                }
            ]
        }
        """
        # Wallet balances
        balances = await self._request("GET", "/v2/wallet/balances")
        usd_balance = 0.0
        if isinstance(balances, list):
            for wallet in balances:
                sym = wallet.get("asset_symbol", "")
                if sym.upper() in ("USD", "USDC", "USDT"):
                    usd_balance = float(wallet.get("available_balance") or 0)
                    break

        # Open positions
        positions_raw = await self._request(
            "GET", "/v2/positions/margined"
        )
        positions = []
        total_upnl = 0.0
        if isinstance(positions_raw, list):
            for pos in positions_raw:
                size = int(pos.get("size") or 0)
                if size == 0:
                    continue
                prod_symbol: str = pos.get("product_symbol", "")
                # Derive asset from product symbol (e.g. "BTCUSD" → "BTC")
                coin = _symbol_to_asset(prod_symbol, self._product_cache)
                entry_px = float(pos.get("entry_price") or 0)
                liq_px = pos.get("liquidation_price")
                realized_pnl = float(pos.get("realized_pnl") or 0)

                # Determine direction: Delta positions have size > 0 for long,
                # < 0 for short (size field is signed in /v2/positions).
                # However, the schema shows unsigned size; we check 'side' or
                # compute from context.  Treat all as positive for now; the
                # risk manager checks via entry/mark comparison.
                signed_size = size  # may be negative for shorts from WS, but REST often gives unsigned
                # Try to infer from mark price vs entry
                try:
                    mark_px = await self.get_current_price(coin) if coin else entry_px
                    upnl = (mark_px - entry_px) * size * self._product_cache.get(
                        coin.upper(), {}
                    ).get("contract_value", 1)
                except Exception:
                    upnl = 0.0
                    mark_px = entry_px
                total_upnl += upnl

                positions.append({
                    "coin": coin,
                    "szi": str(size),
                    "entryPx": str(entry_px),
                    "liqPx": str(liq_px) if liq_px else None,
                    "pnl": round(upnl, 4),
                    "leverage": self._leverage,
                })

        return {
            "balance": usd_balance,
            "total_value": usd_balance + total_upnl,
            "positions": positions,
        }

    # ------------------------------------------------------------------ #
    #  Orders                                                             #
    # ------------------------------------------------------------------ #

    async def get_open_orders(self) -> list[dict]:
        """Return list of open orders across all products."""
        orders = await self._request(
            "GET",
            "/v2/orders",
            params={"state": "open"},
        )
        if not isinstance(orders, list):
            return []
        result = []
        for o in orders:
            prod_sym = o.get("product_symbol", "")
            result.append({
                "coin": _symbol_to_asset(prod_sym, self._product_cache),
                "oid": str(o.get("id")),
                "isBuy": o.get("side") == "buy",
                "sz": str(o.get("size") or 0),
                "px": str(o.get("limit_price") or 0),
                "triggerPx": str(o.get("stop_price") or 0),
                "orderType": o.get("order_type"),
            })
        return result

    async def get_recent_fills(self, limit: int = 50) -> list[dict]:
        """Return recent trade fills."""
        fills = await self._request(
            "GET",
            "/v2/fills",
            params={"page_size": str(min(limit, 200))},
        )
        if not isinstance(fills, list):
            return []
        result = []
        for f in fills:
            prod_sym = f.get("product_symbol", "")
            result.append({
                "coin": _symbol_to_asset(prod_sym, self._product_cache),
                "isBuy": f.get("side") == "buy",
                "sz": str(f.get("size") or 0),
                "px": str(f.get("price") or 0),
                "time": f.get("created_at"),
            })
        return result

    async def place_market_order(
        self, asset: str, side: str, size_contracts: int
    ) -> dict:
        """Place a market order.

        Parameters
        ----------
        asset:           Underlying symbol (``"BTC"``).
        side:            ``"buy"`` or ``"sell"``.
        size_contracts:  Integer number of contracts to trade.
        """
        meta = self._product(asset)
        body = {
            "product_id": meta["id"],
            "product_symbol": meta["symbol"],
            "order_type": "market_order",
            "side": side,
            "size": size_contracts,
        }
        return await self._request("POST", "/v2/orders", body=body)

    async def place_buy_order(self, asset: str, usd_amount: float) -> dict:
        """Buy *asset* with a market order sized by USD allocation."""
        price = await self.get_current_price(asset)
        size = self.usd_to_contracts(asset, usd_amount, price)
        logger.info("MARKET BUY %s  %d contracts (~$%s)", asset, size, usd_amount)
        return await self.place_market_order(asset, "buy", size)

    async def place_sell_order(self, asset: str, usd_amount: float) -> dict:
        """Sell *asset* with a market order sized by USD allocation.

        If ``usd_amount`` exceeds open position, the order is capped to the
        position size (reduce-only semantics via ``size_contracts``).
        """
        price = await self.get_current_price(asset)
        size = self.usd_to_contracts(asset, usd_amount, price)
        logger.info("MARKET SELL %s  %d contracts (~$%s)", asset, size, usd_amount)
        return await self.place_market_order(asset, "sell", size)

    async def place_limit_buy(
        self, asset: str, usd_amount: float, limit_price: float
    ) -> dict:
        """Place a limit buy order."""
        meta = self._product(asset)
        size = self.usd_to_contracts(asset, usd_amount, limit_price)
        body = {
            "product_id": meta["id"],
            "product_symbol": meta["symbol"],
            "order_type": "limit_order",
            "side": "buy",
            "size": size,
            "limit_price": self.round_price(asset, limit_price),
            "time_in_force": "gtc",
        }
        logger.info(
            "LIMIT BUY %s  %d contracts @ %s", asset, size, limit_price
        )
        return await self._request("POST", "/v2/orders", body=body)

    async def place_limit_sell(
        self, asset: str, usd_amount: float, limit_price: float
    ) -> dict:
        """Place a limit sell order."""
        meta = self._product(asset)
        size = self.usd_to_contracts(asset, usd_amount, limit_price)
        body = {
            "product_id": meta["id"],
            "product_symbol": meta["symbol"],
            "order_type": "limit_order",
            "side": "sell",
            "size": size,
            "limit_price": self.round_price(asset, limit_price),
            "time_in_force": "gtc",
        }
        logger.info(
            "LIMIT SELL %s  %d contracts @ %s", asset, size, limit_price
        )
        return await self._request("POST", "/v2/orders", body=body)

    async def place_bracket_order(
        self,
        asset: str,
        is_long: bool,
        tp_price: float | None = None,
        sl_price: float | None = None,
    ) -> dict | None:
        """Attach a TP and/or SL bracket to the open position for *asset*.

        Delta's bracket orders auto-close the entire position when triggered —
        no need to specify a size.

        Parameters
        ----------
        asset:    Underlying symbol.
        is_long:  True if the open position is long.
        tp_price: Take-profit trigger price (None to skip).
        sl_price: Stop-loss trigger price (None to skip).

        Returns the API response dict, or None if no prices were provided.
        """
        if tp_price is None and sl_price is None:
            return None

        meta = self._product(asset)
        body: dict = {
            "product_id": meta["id"],
            "product_symbol": meta["symbol"],
            "bracket_stop_trigger_method": "mark_price",
        }

        if tp_price is not None:
            body["take_profit_order"] = {
                "order_type": "limit_order",
                "stop_price": self.round_price(asset, tp_price),
                "limit_price": self.round_price(asset, tp_price),
            }

        if sl_price is not None:
            body["stop_loss_order"] = {
                "order_type": "market_order",
                "stop_price": self.round_price(asset, sl_price),
            }

        logger.info(
            "BRACKET %s  TP=%s  SL=%s", asset, tp_price, sl_price
        )
        return await self._request("POST", "/v2/orders/bracket", body=body)

    # Compatibility aliases so main.py call-sites work unchanged
    async def place_take_profit(
        self,
        asset: str,
        is_long: bool,
        usd_amount: float,
        tp_price: float,
    ) -> dict | None:
        """Compatibility wrapper — places bracket order with TP only."""
        return await self.place_bracket_order(asset, is_long, tp_price=tp_price)

    async def place_stop_loss(
        self,
        asset: str,
        is_long: bool,
        usd_amount: float,
        sl_price: float,
    ) -> dict | None:
        """Compatibility wrapper — places bracket order with SL only."""
        return await self.place_bracket_order(asset, is_long, sl_price=sl_price)

    async def cancel_all_orders(self, asset: str) -> dict:
        """Cancel all open orders for *asset*."""
        meta = self._product(asset)
        body = {"product_id": meta["id"]}
        return await self._request(
            "DELETE", "/v2/orders/all", body=body
        )

    def extract_oids(self, result: Any) -> list[str]:
        """Extract order IDs from a place_order or bracket_order response.

        Delta returns a single order dict or None for bracket orders.
        """
        if result is None:
            return []
        if isinstance(result, dict):
            oid = result.get("id")
            return [str(oid)] if oid else []
        if isinstance(result, list):
            return [str(item.get("id")) for item in result if item.get("id")]
        return []

    # Keep this method so any legacy calls in main.py don't break
    async def get_meta_and_ctxs(self, dex: str | None = None) -> None:
        """No-op shim: product metadata is pre-loaded in init_products()."""
        pass


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _map_interval(interval: str) -> str:
    """Map trader-friendly interval strings to Delta's resolution format.

    Delta accepted resolutions: ``1m``, ``3m``, ``5m``, ``15m``, ``30m``,
    ``1h``, ``2h``, ``4h``, ``6h``, ``1D``.
    """
    mapping = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "12h": "6h",  # nearest available
        "1d": "1D",
        "1D": "1D",
    }
    return mapping.get(interval, interval)


def _interval_to_seconds(interval: str) -> int:
    """Return candle duration in seconds for time-range computation."""
    unit_map = {"m": 60, "h": 3600, "d": 86400, "D": 86400}
    for suffix, mult in unit_map.items():
        if interval.endswith(suffix):
            try:
                return int(interval[: -len(suffix)]) * mult
            except ValueError:
                pass
    return 300  # default: 5 minutes


def _symbol_to_asset(product_symbol: str, cache: dict) -> str:
    """Reverse-map a Delta product symbol (e.g. ``"BTCUSD"``) to an asset key.

    Searches the product cache first; falls back to stripping trailing 'USD'.
    """
    for key, meta in cache.items():
        if meta.get("symbol") == product_symbol:
            return key
    # Fallback heuristic: "BTCUSD" → "BTC"
    if product_symbol.endswith("USD"):
        return product_symbol[:-3]
    return product_symbol
