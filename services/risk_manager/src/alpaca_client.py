"""
Alpaca Integration for Risk Management

This module provides integration with Alpaca API for real-time portfolio data,
position tracking, and account information needed for risk management.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import aiohttp

from shared.config import get_config
from shared.models import PortfolioState, Position

logger = logging.getLogger(__name__)


class AlpacaRiskClient:
    """Alpaca API client for risk management operations."""

    def __init__(self) -> None:
        """Initialize Alpaca client."""
        self.config = get_config()
        self.alpaca_config = self.config.alpaca

        # API endpoints
        self.base_url = self.alpaca_config.base_url
        self.data_url = self.alpaca_config.data_url

        # Authentication headers
        self.headers = {
            "APCA-API-KEY-ID": self.alpaca_config.api_key,
            "APCA-API-SECRET-KEY": self.alpaca_config.secret_key,
            "Content-Type": "application/json",
        }

        # Rate limiting
        self.last_request_time = datetime.now(timezone.utc)
        self.min_request_interval = timedelta(
            milliseconds=200
        )  # 5 requests per second max

        # Cache for recent data
        self.portfolio_cache: Optional[PortfolioState] = None
        self.cache_timestamp: Optional[datetime] = None
        self.cache_ttl = timedelta(seconds=30)

    async def get_portfolio_state(
        self, force_refresh: bool = False
    ) -> Optional[PortfolioState]:
        """
        Get current portfolio state from Alpaca.

        Args:
            force_refresh: Force refresh even if cached data is available

        Returns:
            Current portfolio state or None if error
        """
        try:
            # Check cache
            if (
                not force_refresh
                and self.portfolio_cache
                and self.cache_timestamp
                and datetime.now(timezone.utc) - self.cache_timestamp < self.cache_ttl
            ):
                return self.portfolio_cache

            # Get account information
            account_info = await self._get_account_info()
            if not account_info:
                return None

            # Get positions
            positions = await self._get_positions()
            if positions is None:
                return None

            # Create portfolio state
            portfolio = PortfolioState(
                account_id=account_info["id"],
                timestamp=datetime.now(timezone.utc),
                cash=Decimal(account_info["cash"]),
                buying_power=Decimal(account_info["buying_power"]),
                total_equity=Decimal(account_info["equity"]),
                positions=positions,
                day_trades_count=int(account_info.get("daytrade_count", 0)),
                pattern_day_trader=account_info.get("pattern_day_trader", False),
            )

            # Update cache
            self.portfolio_cache = portfolio
            self.cache_timestamp = datetime.now(timezone.utc)

            logger.debug(
                f"Portfolio state updated: ${portfolio.total_equity} equity, {len(positions)} positions"
            )

            return portfolio

        except Exception as e:
            logger.error(f"Error getting portfolio state: {e}")
            return None

    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Decimal]:
        """
        Get current market prices for symbols.

        Args:
            symbols: List of trading symbols

        Returns:
            Dictionary mapping symbols to current prices
        """
        try:
            if not symbols:
                return {}

            # Rate limiting
            await self._rate_limit()

            # Build quote request
            symbol_list = ",".join(symbols)
            url = f"{self.data_url}/v2/stocks/quotes/latest"
            params = {"symbols": symbol_list}

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=self.headers, params=params, timeout=timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        prices = {}
                        quotes = data.get("quotes", {})

                        for symbol in symbols:
                            if symbol in quotes:
                                quote = quotes[symbol]
                                # Use bid-ask midpoint for better accuracy
                                bid = Decimal(str(quote.get("bid", 0)))
                                ask = Decimal(str(quote.get("ask", 0)))

                                if bid > 0 and ask > 0:
                                    prices[symbol] = (bid + ask) / Decimal("2")
                                else:
                                    # Fallback to last trade price
                                    prices[symbol] = Decimal(str(quote.get("price", 0)))

                        logger.debug(f"Retrieved prices for {len(prices)} symbols")
                        return prices
                    else:
                        logger.error(f"Failed to get quotes: HTTP {response.status}")
                        return {}

        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {}

    async def get_account_buying_power(self) -> Decimal:
        """Get current account buying power."""
        try:
            account_info = await self._get_account_info()
            if account_info:
                return Decimal(account_info["buying_power"])
            return Decimal("0")
        except Exception as e:
            logger.error(f"Error getting buying power: {e}")
            return Decimal("0")

    async def get_position_pnl(self, symbol: str) -> Optional[Tuple[Decimal, Decimal]]:
        """
        Get unrealized and realized P&L for a position.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (unrealized_pnl, realized_pnl) or None if error
        """
        try:
            positions = await self._get_positions()
            if not positions:
                return None

            for position in positions:
                if position.symbol == symbol:
                    # Realized P&L would need to be calculated from trades
                    # For now, return unrealized P&L and 0 for realized
                    return position.unrealized_pnl, Decimal("0")

            return None

        except Exception as e:
            logger.error(f"Error getting P&L for {symbol}: {e}")
            return None

    async def get_day_trade_count(self) -> int:
        """Get current day trade count."""
        try:
            account_info = await self._get_account_info()
            if account_info:
                return int(account_info.get("daytrade_count", 0))
            return 0
        except Exception as e:
            logger.error(f"Error getting day trade count: {e}")
            return 0

    async def check_pattern_day_trader_status(self) -> bool:
        """Check if account is marked as pattern day trader."""
        try:
            account_info = await self._get_account_info()
            if account_info:
                return account_info.get("pattern_day_trader", False)
            return False
        except Exception as e:
            logger.error(f"Error checking PDT status: {e}")
            return False

    async def get_market_hours_info(self) -> Dict:
        """Get market hours and trading status."""
        try:
            await self._rate_limit()

            url = f"{self.base_url}/v2/clock"

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=self.headers, timeout=timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "is_open": data.get("is_open", False),
                            "next_open": data.get("next_open"),
                            "next_close": data.get("next_close"),
                            "timestamp": data.get("timestamp"),
                        }
                    else:
                        logger.error(
                            f"Failed to get market hours: HTTP {response.status}"
                        )
                        return {"is_open": False}

        except Exception as e:
            logger.error(f"Error getting market hours: {e}")
            return {"is_open": False}

    async def get_historical_portfolio_values(
        self, days: int = 30
    ) -> List[Tuple[datetime, Decimal]]:
        """
        Get historical portfolio values for performance calculation.

        Args:
            days: Number of days of history to retrieve

        Returns:
            List of (timestamp, portfolio_value) tuples
        """
        try:
            await self._rate_limit()

            # Get portfolio history
            end_date = datetime.now(timezone.utc)

            url = f"{self.base_url}/v2/account/portfolio/history"
            params = {
                "period": f"{days}D",
                "timeframe": "1D",
                "end": end_date.strftime("%Y-%m-%d"),
                "extended_hours": "false",
            }

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=self.headers, params=params, timeout=timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        timestamps = data.get("timestamp", [])
                        equity_values = data.get("equity", [])

                        history = []
                        for ts, equity in zip(timestamps, equity_values):
                            timestamp = datetime.fromtimestamp(ts)
                            portfolio_value = Decimal(str(equity))
                            history.append((timestamp, portfolio_value))

                        logger.debug(
                            f"Retrieved {len(history)} days of portfolio history"
                        )
                        return history
                    else:
                        logger.error(
                            f"Failed to get portfolio history: HTTP {response.status}"
                        )
                        return []

        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            return []

    async def _get_account_info(self) -> Optional[Dict]:
        """Get account information from Alpaca."""
        try:
            await self._rate_limit()

            url = f"{self.base_url}/v2/account"

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=self.headers, timeout=timeout
                ) as response:
                    if response.status == 200:
                        account_data = await response.json()
                        logger.debug("Account information retrieved successfully")
                        return account_data
                    else:
                        logger.error(
                            f"Failed to get account info: HTTP {response.status}"
                        )
                        return None

        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

    async def _get_positions(self) -> Optional[List[Position]]:
        """Get current positions from Alpaca."""
        try:
            await self._rate_limit()

            url = f"{self.base_url}/v2/positions"

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=self.headers, timeout=timeout
                ) as response:
                    if response.status == 200:
                        positions_data = await response.json()

                        positions = []
                        for pos_data in positions_data:
                            try:
                                position = Position(
                                    symbol=pos_data["symbol"],
                                    quantity=int(pos_data["qty"]),
                                    entry_price=Decimal(pos_data["avg_entry_price"]),
                                    current_price=(
                                        Decimal(pos_data["market_value"])
                                        / Decimal(pos_data["qty"])
                                        if int(pos_data["qty"]) != 0
                                        else Decimal("0")
                                    ),
                                    unrealized_pnl=Decimal(pos_data["unrealized_pl"]),
                                    market_value=Decimal(pos_data["market_value"]),
                                    cost_basis=Decimal(pos_data["cost_basis"]),
                                    last_updated=datetime.now(timezone.utc),
                                )
                                positions.append(position)
                            except (ValueError, KeyError) as e:
                                logger.warning(
                                    f"Error parsing position data for {pos_data.get('symbol', 'unknown')}: {e}"
                                )
                                continue

                        logger.debug(f"Retrieved {len(positions)} positions")
                        return positions
                    else:
                        logger.error(f"Failed to get positions: HTTP {response.status}")
                        return []

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return None

    async def get_historical_prices(
        self, symbol: str, days: int = 30
    ) -> List[Tuple[datetime, Decimal]]:
        """
        Get historical prices for volatility and correlation calculations.

        Args:
            symbol: Trading symbol
            days: Number of days of history

        Returns:
            List of (timestamp, close_price) tuples
        """
        try:
            await self._rate_limit()

            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            url = f"{self.data_url}/v2/stocks/{symbol}/bars"
            params = {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "timeframe": "1Day",
                "limit": "500",
                "adjustment": "raw",
            }

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=self.headers, params=params, timeout=timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        bars = data.get("bars", [])
                        prices = []

                        for bar in bars:
                            timestamp = datetime.fromisoformat(
                                bar["t"].replace("Z", "+00:00")
                            )
                            close_price = Decimal(str(bar["c"]))
                            prices.append((timestamp, close_price))

                        logger.debug(
                            f"Retrieved {len(prices)} price points for {symbol}"
                        )
                        return prices
                    else:
                        logger.error(
                            f"Failed to get historical prices for {symbol}: HTTP {response.status}"
                        )
                        return []

        except Exception as e:
            logger.error(f"Error getting historical prices for {symbol}: {e}")
            return []

    async def get_real_time_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote for a symbol."""
        try:
            await self._rate_limit()

            url = f"{self.data_url}/v2/stocks/{symbol}/quotes/latest"

            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=self.headers, timeout=timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        quote = data.get("quote", {})

                        return {
                            "symbol": symbol,
                            "bid": Decimal(str(quote.get("bp", 0))),
                            "ask": Decimal(str(quote.get("ap", 0))),
                            "bid_size": int(quote.get("bs", 0)),
                            "ask_size": int(quote.get("as", 0)),
                            "timestamp": quote.get("t"),
                        }
                    else:
                        logger.warning(
                            f"Failed to get quote for {symbol}: HTTP {response.status}"
                        )
                        return None

        except Exception as e:
            logger.error(f"Error getting real-time quote for {symbol}: {e}")
            return None

    async def get_account_activities(
        self, activity_type: str = "FILL", limit: int = 100
    ) -> List[Dict]:
        """
        Get account activities for trade tracking.

        Args:
            activity_type: Type of activity to retrieve
            limit: Maximum number of activities to retrieve

        Returns:
            List of activity records
        """
        try:
            await self._rate_limit()

            url = f"{self.base_url}/v2/account/activities"
            params = {"activity_type": activity_type, "page_size": str(limit)}

            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=self.headers, params=params, timeout=timeout
                ) as response:
                    if response.status == 200:
                        activities = await response.json()
                        logger.debug(f"Retrieved {len(activities)} account activities")
                        return activities
                    else:
                        logger.error(
                            f"Failed to get account activities: HTTP {response.status}"
                        )
                        return []

        except Exception as e:
            logger.error(f"Error getting account activities: {e}")
            return []

    async def get_orders(
        self, status: Optional[str] = None, limit: int = 100
    ) -> List[Dict]:
        """
        Get order history.

        Args:
            status: Filter by order status
            limit: Maximum number of orders to retrieve

        Returns:
            List of order records
        """
        try:
            await self._rate_limit()

            url = f"{self.base_url}/v2/orders"
            params = {"limit": str(limit)}

            if status:
                params["status"] = status

            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=self.headers, params=params, timeout=timeout
                ) as response:
                    if response.status == 200:
                        orders = await response.json()
                        logger.debug(f"Retrieved {len(orders)} orders")
                        return orders
                    else:
                        logger.error(f"Failed to get orders: HTTP {response.status}")
                        return []

        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []

    async def get_daily_pnl(self) -> Decimal:
        """Get today's realized + unrealized P&L."""
        try:
            # Get portfolio history for today

            url = f"{self.base_url}/v2/account/portfolio/history"
            params = {"period": "1D", "timeframe": "5Min"}

            await self._rate_limit()

            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=self.headers, params=params, timeout=timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        equity_values = data.get("equity", [])
                        if len(equity_values) >= 2:
                            # P&L is difference from start of day to now
                            start_value = Decimal(str(equity_values[0]))
                            current_value = Decimal(str(equity_values[-1]))
                            daily_pnl = current_value - start_value

                            logger.debug(f"Daily P&L calculated: ${daily_pnl}")
                            return daily_pnl
                        else:
                            return Decimal("0")
                    else:
                        logger.error(
                            f"Failed to get portfolio history: HTTP {response.status}"
                        )
                        return Decimal("0")

        except Exception as e:
            logger.error(f"Error getting daily P&L: {e}")
            return Decimal("0")

    async def calculate_portfolio_volatility(self, days: int = 30) -> float:
        """Calculate portfolio volatility from historical data."""
        try:
            # Get portfolio history
            history = await self.get_historical_portfolio_values(days)

            if len(history) < 10:  # Need minimum data points
                return 0.15  # Default volatility

            # Calculate daily returns
            returns = []
            for i in range(1, len(history)):
                prev_value = history[i - 1][1]
                curr_value = history[i][1]

                if prev_value > 0:
                    daily_return = float((curr_value - prev_value) / prev_value)
                    returns.append(daily_return)

            if not returns:
                return 0.15

            # Calculate volatility (standard deviation of returns)
            import numpy as np

            volatility = np.std(returns) * np.sqrt(252)  # Annualize

            return float(volatility)

        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.15

    async def calculate_symbol_correlation(
        self, symbol1: str, symbol2: str, days: int = 60
    ) -> float:
        """Calculate correlation between two symbols."""
        try:
            # Get historical prices for both symbols
            prices1 = await self.get_historical_prices(symbol1, days)
            prices2 = await self.get_historical_prices(symbol2, days)

            if len(prices1) < 10 or len(prices2) < 10:
                return 0.0  # Insufficient data

            # Align dates and calculate returns
            price_dict1 = {date.date(): price for date, price in prices1}
            price_dict2 = {date.date(): price for date, price in prices2}

            common_dates = set(price_dict1.keys()) & set(price_dict2.keys())
            common_dates_list = sorted(common_dates)

            if len(common_dates_list) < 10:
                return 0.0

            returns1 = []
            returns2 = []

            for i in range(1, len(common_dates_list)):
                prev_date = common_dates_list[i - 1]
                curr_date = common_dates_list[i]

                prev_price1 = price_dict1[prev_date]
                curr_price1 = price_dict1[curr_date]
                prev_price2 = price_dict2[prev_date]
                curr_price2 = price_dict2[curr_date]

                if prev_price1 > 0 and prev_price2 > 0:
                    return1 = float((curr_price1 - prev_price1) / prev_price1)
                    return2 = float((curr_price2 - prev_price2) / prev_price2)

                    returns1.append(return1)
                    returns2.append(return2)

            if len(returns1) < 10:
                return 0.0

            # Calculate correlation
            import numpy as np

            correlation = np.corrcoef(returns1, returns2)[0, 1]

            # Handle NaN values
            if np.isnan(correlation):
                correlation = 0.0

            return float(correlation)

        except Exception as e:
            logger.error(
                f"Error calculating correlation between {symbol1} and {symbol2}: {e}"
            )
            return 0.0

    async def get_symbol_beta(
        self, symbol: str, benchmark: str = "SPY", days: int = 60
    ) -> float:
        """Calculate beta of symbol relative to benchmark."""
        try:
            # Get historical prices for symbol and benchmark
            symbol_prices = await self.get_historical_prices(symbol, days)
            benchmark_prices = await self.get_historical_prices(benchmark, days)

            if len(symbol_prices) < 10 or len(benchmark_prices) < 10:
                return 1.0  # Default beta

            # Calculate returns
            symbol_returns = self._calculate_returns_from_prices(symbol_prices)
            benchmark_returns = self._calculate_returns_from_prices(benchmark_prices)

            if len(symbol_returns) < 10 or len(benchmark_returns) < 10:
                return 1.0

            # Align returns by date
            aligned_returns = self._align_returns(symbol_returns, benchmark_returns)

            if len(aligned_returns) < 10:
                return 1.0

            # Calculate beta using linear regression
            import numpy as np

            symbol_ret = [ret[1] for ret in aligned_returns]
            benchmark_ret = [ret[2] for ret in aligned_returns]

            covariance = np.cov(symbol_ret, benchmark_ret)[0, 1]
            benchmark_variance = np.var(benchmark_ret)

            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
                return float(beta)
            else:
                return 1.0

        except Exception as e:
            logger.error(f"Error calculating beta for {symbol}: {e}")
            return 1.0

    def _calculate_returns_from_prices(
        self, prices: List[Tuple[datetime, Decimal]]
    ) -> List[Tuple[datetime, float]]:
        """Calculate returns from price series."""
        returns = []

        for i in range(1, len(prices)):
            date = prices[i][0]
            prev_price = prices[i - 1][1]
            curr_price = prices[i][1]

            if prev_price > 0:
                daily_return = float((curr_price - prev_price) / prev_price)
                returns.append((date, daily_return))

        return returns

    def _align_returns(
        self,
        returns1: List[Tuple[datetime, float]],
        returns2: List[Tuple[datetime, float]],
    ) -> List[Tuple[datetime, float, float]]:
        """Align two return series by date."""
        dict1 = {date.date(): ret for date, ret in returns1}
        dict2 = {date.date(): ret for date, ret in returns2}

        common_dates = set(dict1.keys()) & set(dict2.keys())
        common_dates_list = sorted(common_dates)

        aligned = []
        for date in common_dates_list:
            # Convert date back to datetime for return type consistency
            dt = datetime.combine(date, datetime.min.time())
            aligned.append((dt, dict1[date], dict2[date]))

        return aligned

    async def _rate_limit(self) -> None:
        """Implement rate limiting for API requests."""
        time_since_last = datetime.now(timezone.utc) - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = (self.min_request_interval - time_since_last).total_seconds()
            await asyncio.sleep(sleep_time)

        self.last_request_time = datetime.now(timezone.utc)

    async def validate_connection(self) -> bool:
        """Validate connection to Alpaca API."""
        try:
            account_info = await self._get_account_info()
            return account_info is not None
        except Exception as e:
            logger.error(f"Alpaca connection validation failed: {e}")
            return False

    def get_connection_info(self) -> Dict:
        """Get connection information."""
        return {
            "base_url": self.base_url,
            "data_url": self.data_url,
            "paper_trading": self.alpaca_config.paper_trading,
            "api_key": (
                self.alpaca_config.api_key[:8] + "..."
                if self.alpaca_config.api_key
                else None
            ),
            "cache_ttl_seconds": self.cache_ttl.total_seconds(),
            "min_request_interval_ms": self.min_request_interval.total_seconds() * 1000,
        }

    async def get_market_data_for_risk_calc(
        self, symbols: List[str], days: int = 60
    ) -> Dict[str, Dict]:
        """
        Get comprehensive market data for risk calculations.

        Args:
            symbols: List of symbols to get data for
            days: Number of days of historical data

        Returns:
            Dictionary with market data for each symbol
        """
        try:
            market_data = {}

            # Get current prices
            current_prices = await self.get_current_prices(symbols)

            # Get historical data for each symbol
            for symbol in symbols:
                try:
                    historical_prices = await self.get_historical_prices(symbol, days)

                    if historical_prices:
                        # Calculate basic statistics
                        prices = [float(price) for _, price in historical_prices]

                        # Calculate returns
                        returns = []
                        for i in range(1, len(prices)):
                            if prices[i - 1] > 0:
                                ret = (prices[i] - prices[i - 1]) / prices[i - 1]
                                returns.append(ret)

                        if returns:
                            import numpy as np

                            volatility = np.std(returns) * np.sqrt(252)  # Annualized
                            avg_return = np.mean(returns) * 252  # Annualized

                            market_data[symbol] = {
                                "current_price": current_prices.get(
                                    symbol, Decimal("0")
                                ),
                                "volatility": volatility,
                                "avg_return": avg_return,
                                "price_history": historical_prices[
                                    -30:
                                ],  # Last 30 days
                                "returns": returns[-30:],  # Last 30 returns
                                "data_points": len(historical_prices),
                            }
                        else:
                            market_data[symbol] = {
                                "current_price": current_prices.get(
                                    symbol, Decimal("0")
                                ),
                                "volatility": 0.20,  # Default
                                "avg_return": 0.0,
                                "price_history": [],
                                "returns": [],
                                "data_points": 0,
                            }
                    else:
                        market_data[symbol] = {
                            "current_price": current_prices.get(symbol, Decimal("0")),
                            "volatility": 0.20,  # Default
                            "avg_return": 0.0,
                            "price_history": [],
                            "returns": [],
                            "data_points": 0,
                        }

                except Exception as e:
                    logger.warning(f"Error getting data for {symbol}: {e}")
                    continue

            return market_data

        except Exception as e:
            logger.error(f"Error getting market data for risk calculations: {e}")
            return {}

    async def check_market_conditions(self) -> Dict:
        """Check current market conditions for risk assessment."""
        try:
            # Get market status
            market_info = await self.get_market_hours_info()

            # Get major index prices for market context
            index_symbols = ["SPY", "QQQ", "IWM"]
            index_prices = await self.get_current_prices(index_symbols)

            # Get VIX if available (volatility index)
            vix_data = await self.get_real_time_quote("VIX")

            conditions = {
                "market_open": market_info.get("is_open", False),
                "next_open": market_info.get("next_open"),
                "next_close": market_info.get("next_close"),
                "major_indices": {
                    symbol: str(price) for symbol, price in index_prices.items()
                },
                "vix": (
                    str(vix_data["bid"]) if vix_data and vix_data["bid"] > 0 else None
                ),
                "market_volatility": "normal",  # Would calculate from market data
                "trading_session": self._get_trading_session(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            return conditions

        except Exception as e:
            logger.error(f"Error checking market conditions: {e}")
            return {
                "market_open": False,
                "trading_session": "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def _get_trading_session(self) -> str:
        """Determine current trading session."""
        try:
            now = datetime.now(timezone.utc)
            hour = now.hour
            minute = now.minute

            # Convert to EST/EDT (market timezone)
            # This is a simplified conversion - in production use proper timezone handling
            est_hour = hour - 5  # Assuming UTC-5 for EST

            if est_hour < 0:
                est_hour += 24

            # Pre-market: 4:00 AM - 9:30 AM EST
            if 4 <= est_hour < 9 or (est_hour == 9 and minute < 30):
                return "pre_market"
            # Regular session: 9:30 AM - 4:00 PM EST
            elif 9 < est_hour < 16 or (est_hour == 9 and minute >= 30):
                return "regular"
            # After-hours: 4:00 PM - 8:00 PM EST
            elif 16 <= est_hour < 20:
                return "after_hours"
            else:
                return "closed"

        except Exception as e:
            logger.error(f"Error determining trading session: {e}")
            return "unknown"

    async def get_position_details(self, symbol: str) -> Optional[Dict]:
        """Get detailed information for a specific position."""
        try:
            await self._rate_limit()

            url = f"{self.base_url}/v2/positions/{symbol}"

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=self.headers, timeout=timeout
                ) as response:
                    if response.status == 200:
                        position_data = await response.json()
                        return position_data
                    elif response.status == 404:
                        # No position for this symbol
                        return None
                    else:
                        logger.error(
                            f"Failed to get position details for {symbol}: HTTP {response.status}"
                        )
                        return None

        except Exception as e:
            logger.error(f"Error getting position details for {symbol}: {e}")
            return None

    async def get_account_configuration(self) -> Dict:
        """Get account configuration and trading permissions."""
        try:
            await self._rate_limit()

            url = f"{self.base_url}/v2/account/configurations"

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=self.headers, timeout=timeout
                ) as response:
                    if response.status == 200:
                        config_data = await response.json()
                        return config_data
                    else:
                        logger.error(
                            f"Failed to get account configuration: HTTP {response.status}"
                        )
                        return {}

        except Exception as e:
            logger.error(f"Error getting account configuration: {e}")
            return {}

    async def is_shortable(self, symbol: str) -> bool:
        """Check if a symbol is available for short selling."""
        try:
            await self._rate_limit()

            url = f"{self.base_url}/v2/stocks/{symbol}/quotes/latest"

            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=self.headers, timeout=timeout
                ) as response:
                    if response.status == 200:
                        # In a real implementation, you'd check the shortable attribute
                        # For now, assume most stocks are shortable
                        return True
                    else:
                        return False

        except Exception as e:
            logger.error(f"Error checking if {symbol} is shortable: {e}")
            return False

    async def get_trade_history(
        self, symbol: Optional[str] = None, days: int = 30
    ) -> List[Dict]:
        """Get trade history for performance analysis."""
        try:
            # Get account activities (fills)
            activities = await self.get_account_activities("FILL", limit=1000)

            # Filter and process trades
            trades = []
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

            for activity in activities:
                try:
                    activity_date = datetime.fromisoformat(
                        activity["date"].replace("Z", "+00:00")
                    )

                    if activity_date < cutoff_date:
                        continue

                    if symbol and activity.get("symbol") != symbol:
                        continue

                    trade_info = {
                        "symbol": activity.get("symbol"),
                        "side": activity.get("side"),
                        "quantity": int(activity.get("qty", 0)),
                        "price": Decimal(str(activity.get("price", 0))),
                        "timestamp": activity_date,
                        "order_id": activity.get("order_id"),
                        "trade_id": activity.get("transaction_id"),
                    }
                    trades.append(trade_info)

                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing trade activity: {e}")
                    continue

            logger.debug(f"Retrieved {len(trades)} trades for analysis")
            return trades

        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []

    async def calculate_realized_pnl(
        self, symbol: Optional[str] = None, days: int = 1
    ) -> Decimal:
        """Calculate realized P&L for a symbol or entire portfolio."""
        try:
            trades = await self.get_trade_history(symbol, days)

            if not trades:
                return Decimal("0")

            # Simple P&L calculation (buy price vs sell price)
            # In production, you'd use FIFO/LIFO accounting
            total_pnl = Decimal("0")

            buy_positions = {}  # symbol -> [quantity, avg_price]

            for trade in sorted(trades, key=lambda x: x["timestamp"]):
                symbol_trade = trade["symbol"]
                quantity = trade["quantity"]
                price = trade["price"]
                side = trade["side"]

                if side.upper() == "BUY":
                    # Add to buy positions
                    if symbol_trade not in buy_positions:
                        buy_positions[symbol_trade] = [0, Decimal("0")]

                    # Update average price
                    current_qty, current_avg = buy_positions[symbol_trade]
                    current_qty = (
                        int(current_qty)
                        if isinstance(current_qty, (int, float))
                        else current_qty
                    )
                    current_avg = (
                        Decimal(str(current_avg))
                        if not isinstance(current_avg, Decimal)
                        else current_avg
                    )
                    total_qty = current_qty + quantity
                    total_cost = (Decimal(str(current_qty)) * current_avg) + (
                        quantity * price
                    )
                    new_avg = total_cost / total_qty if total_qty > 0 else Decimal("0")

                    buy_positions[symbol_trade] = [total_qty, new_avg]

                elif side.upper() == "SELL":
                    # Calculate P&L from sale
                    if symbol_trade in buy_positions:
                        buy_qty, buy_avg = buy_positions[symbol_trade]

                        if buy_qty >= quantity:
                            # Full or partial sale
                            pnl = (price - buy_avg) * quantity
                            total_pnl += pnl

                            # Update remaining position
                            remaining_qty = buy_qty - quantity
                            buy_positions[symbol_trade] = [remaining_qty, buy_avg]

                            if remaining_qty <= 0:
                                del buy_positions[symbol_trade]

            return total_pnl

        except Exception as e:
            logger.error(f"Error calculating realized P&L: {e}")
            return Decimal("0")

    async def get_risk_metrics_data(self, portfolio: PortfolioState) -> Dict:
        """Get all data needed for comprehensive risk metrics calculation."""
        try:
            symbols = [p.symbol for p in portfolio.positions if p.quantity != 0]

            if not symbols:
                return {}

            # Get comprehensive market data
            market_data = await self.get_market_data_for_risk_calc(symbols, days=60)

            # Get account configuration
            account_config = await self.get_account_configuration()

            # Get recent trade history
            recent_trades = await self.get_trade_history(days=7)

            # Get market conditions
            market_conditions = await self.check_market_conditions()

            risk_data = {
                "market_data": market_data,
                "account_config": account_config,
                "recent_trades": recent_trades,
                "market_conditions": market_conditions,
                "symbols": symbols,
                "data_timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.debug(f"Compiled risk metrics data for {len(symbols)} symbols")
            return risk_data

        except Exception as e:
            logger.error(f"Error getting risk metrics data: {e}")
            return {}

    def clear_cache(self) -> None:
        """Clear cached data."""
        self.portfolio_cache = None
        self.cache_timestamp = None
        logger.debug("Alpaca client cache cleared")
