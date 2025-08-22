"""
FinViz Elite screener integration for fetching market data.

This module provides a comprehensive interface to FinViz Elite's screener
functionality with rate limiting, data parsing, and error handling.
"""

import asyncio
import csv
import io
import logging
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Dict, Any

import aiohttp
from pydantic import BaseModel, Field

from shared.models import FinVizData
from shared.config import get_config


logger = logging.getLogger(__name__)


class FinVizScreenerParams(BaseModel):
    """FinViz screener parameters configuration."""

    # Market cap filters
    market_cap_min: Optional[str] = Field(None, description="Minimum market cap")
    market_cap_max: Optional[str] = Field(None, description="Maximum market cap")

    # Volume filters
    avg_volume_min: Optional[str] = Field(None, description="Minimum average volume")
    current_volume_min: Optional[str] = Field(None, description="Minimum current volume")

    # Price filters
    price_min: Optional[float] = Field(None, description="Minimum price")
    price_max: Optional[float] = Field(None, description="Maximum price")

    # Technical filters
    above_sma20: bool = Field(False, description="Above SMA 20")
    weekly_volatility_min: Optional[float] = Field(None, description="Minimum weekly volatility")

    # Custom filters
    custom_filters: Dict[str, str] = Field(default_factory=dict, description="Additional custom filters")

    def to_finviz_params(self) -> Dict[str, str]:
        """Convert parameters to FinViz screener format."""
        params = {
            'v': '111',  # Export format
            'f': '',     # Filters will be built
            'c': '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70'
        }

        filters = []

        # Market cap filter
        if self.market_cap_min or self.market_cap_max:
            if self.market_cap_min == "2M" and self.market_cap_max == "2B":
                filters.append("cap_smallover")
            elif self.market_cap_min == "2M":
                filters.append("cap_microover")
            elif self.market_cap_max == "2B":
                filters.append("cap_smallunder")

        # Volume filters
        if self.avg_volume_min:
            if self.avg_volume_min == "1000K":
                filters.append("sh_avgvol_o1000")
            elif self.avg_volume_min == "500K":
                filters.append("sh_avgvol_o500")

        if self.current_volume_min:
            if self.current_volume_min == "400K":
                filters.append("sh_curvol_o400")
            elif self.current_volume_min == "200K":
                filters.append("sh_curvol_o200")

        # Price filters
        if self.price_min and self.price_max:
            if self.price_min == 5 and self.price_max == 35:
                filters.append("sh_price_5to35")
            elif self.price_min == 1 and self.price_max == 50:
                filters.append("sh_price_1to50")
        elif self.price_min:
            if self.price_min == 5:
                filters.append("sh_price_o5")
        elif self.price_max:
            if self.price_max == 35:
                filters.append("sh_price_u35")

        # Technical filters
        if self.above_sma20:
            filters.append("ta_sma20_pa")

        if self.weekly_volatility_min:
            if self.weekly_volatility_min >= 6:
                filters.append("ta_volatility_wo6")
            elif self.weekly_volatility_min >= 4:
                filters.append("ta_volatility_wo4")

        # Add custom filters
        filters.extend(self.custom_filters.values())

        params['f'] = ','.join(filters)
        return params


class FinVizScreenerResult(BaseModel):
    """Result from FinViz screener."""

    data: List[FinVizData]
    timestamp: datetime
    total_count: int
    screener_params: FinVizScreenerParams
    execution_time: float


class RateLimiter:
    """Rate limiter for API calls."""

    def __init__(self, min_interval: float = 300.0):  # 5 minutes default
        self.min_interval = min_interval
        self.last_call = 0.0
        self._lock = asyncio.Lock()

    async def wait_if_needed(self):
        """Wait if necessary to respect rate limiting."""
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_call

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)

            self.last_call = time.time()


class FinVizScreener:
    """
    FinViz Elite screener client for fetching market data.

    Provides rate-limited access to FinViz screener with customizable parameters
    and automatic data parsing.
    """

    def __init__(
        self,
        base_url: str = "https://elite.finviz.com",
        timeout: int = 30,
        max_retries: int = 3,
        rate_limit_interval: float = 300.0,
        session: Optional[aiohttp.ClientSession] = None,
        api_key: Optional[str] = None
    ):
        """Initialize FinViz screener."""
        self.base_url = base_url.rstrip('/')
        self.export_url = f"{self.base_url}/export.ashx"
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self._session = session
        self._rate_limiter = RateLimiter(rate_limit_interval)

        # Get API key from config or environment
        if api_key:
            self.api_key = api_key
        else:
            try:
                config = get_config()
                self.api_key = config.finviz.api_key
            except:
                # Fallback to environment variable
                self.api_key = os.getenv('FINVIZ_API_KEY')

        if not self.api_key:
            logger.warning("No FinViz API key configured - screener will likely return no results")

        # Default screener parameters for momentum trading (relaxed criteria)
        self.default_params = FinVizScreenerParams(
            market_cap_min="2M",
            market_cap_max="10B",  # Allow larger caps
            avg_volume_min="500K",  # Reduced from 1000K
            current_volume_min="200K",  # Reduced from 400K
            price_min=3.0,  # Reduced from 5.0
            price_max=50.0,  # Increased from 35.0
            above_sma20=True,
            weekly_volatility_min=3.0  # Reduced from 6.0 to 3.0
        )

    async def __aenter__(self):
        """Async context manager entry."""
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def fetch_screener_data(
        self,
        params: Optional[FinVizScreenerParams] = None,
        limit: Optional[int] = 20
    ) -> FinVizScreenerResult:
        """
        Fetch data from FinViz screener.

        Args:
            params: Screener parameters. Uses default if None.
            limit: Maximum number of results to return

        Returns:
            FinVizScreenerResult with parsed data

        Raises:
            aiohttp.ClientError: On HTTP request failures
            ValueError: On data parsing errors
        """
        start_time = time.time()

        if params is None:
            params = self.default_params

        # Wait for rate limiting
        await self._rate_limiter.wait_if_needed()

        # Build request parameters
        request_params = params.to_finviz_params()

        # Add API authentication
        if self.api_key:
            request_params['auth'] = self.api_key
            request_params['ft'] = '4'  # Filter type for authenticated requests

        logger.info(f"Fetching FinViz screener data with filters: {request_params['f']}")
        logger.info(f"Request URL: {self.export_url}")
        logger.info(f"Request params: {request_params}")

        session = await self._get_session()

        for attempt in range(self.max_retries):
            try:
                async with session.get(
                    self.export_url,
                    params=request_params,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'text/csv,application/csv,*/*',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                    }
                ) as response:
                    response.raise_for_status()
                    content = await response.text()

                    logger.info(f"Response status: {response.status}")
                    logger.info(f"Response content length: {len(content)} chars")
                    logger.info(f"Response first 500 chars: {content[:500]}")

                    # Parse CSV data
                    parsed_data = self._parse_csv_response(content, limit)

                    execution_time = time.time() - start_time

                    result = FinVizScreenerResult(
                        data=parsed_data,
                        timestamp=datetime.now(timezone.utc),
                        total_count=len(parsed_data),
                        screener_params=params,
                        execution_time=execution_time
                    )

                    logger.info(f"Successfully fetched {len(parsed_data)} tickers in {execution_time:.2f}s")
                    return result

            except aiohttp.ClientError as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        raise Exception("Max retries exceeded")

    def _parse_csv_response(self, content: str, limit: Optional[int] = None) -> List[FinVizData]:
        """
        Parse CSV response from FinViz into structured data.

        Args:
            content: Raw CSV content from FinViz
            limit: Maximum number of records to return

        Returns:
            List of FinVizData objects
        """
        try:
            # Read CSV content
            csv_reader = csv.DictReader(io.StringIO(content))

            # Check if we have any rows
            rows_list = list(csv_reader)
            logger.info(f"Total rows in CSV: {len(rows_list)}")
            if rows_list:
                logger.info(f"CSV headers: {rows_list[0].keys()}")
                logger.info(f"First row data: {rows_list[0]}")

            parsed_data = []
            timestamp = datetime.now(timezone.utc)

            for i, row in enumerate(rows_list):
                if limit and i >= limit:
                    break

                try:
                    # Clean and validate data
                    ticker_data = self._clean_row_data(row, timestamp)
                    if ticker_data:
                        parsed_data.append(ticker_data)
                    else:
                        logger.info(f"Row {i} returned None from _clean_row_data")

                except Exception as e:
                    logger.warning(f"Failed to parse row {i}: {e}")
                    logger.info(f"Row {i} data: {row}")
                    continue

            # Sort by volume (descending) to get most active stocks first
            parsed_data.sort(key=lambda x: x.volume or 0, reverse=True)

            logger.info(f"Successfully parsed {len(parsed_data)} tickers from {len(rows_list)} rows")
            return parsed_data

        except Exception as e:
            logger.error(f"Failed to parse CSV response: {e}")
            logger.info(f"Content preview: {content[:1000]}")
            raise ValueError(f"CSV parsing failed: {e}")

    def _clean_row_data(self, row: Dict[str, str], timestamp: datetime) -> Optional[FinVizData]:
        """
        Clean and validate a single row of data.

        Args:
            row: Raw CSV row data
            timestamp: Timestamp to assign to the data

        Returns:
            FinVizData object or None if data is invalid
        """
        try:
            # Map common column names (FinViz column names may vary)
            symbol_keys = ['Ticker', 'Symbol', 'ticker', 'symbol']
            company_keys = ['Company', 'company', 'Company Name']
            sector_keys = ['Sector', 'sector']
            industry_keys = ['Industry', 'industry']
            country_keys = ['Country', 'country']
            market_cap_keys = ['Market Cap', 'Market_Cap', 'market_cap', 'Mkt Cap']
            pe_keys = ['P/E', 'PE', 'pe', 'PE Ratio']
            price_keys = ['Price', 'price', 'Current Price']
            change_keys = ['Change', 'change', '% Change', 'Change %']
            volume_keys = ['Volume', 'volume', 'Vol']

            # Extract data using flexible key matching
            symbol = self._get_value_by_keys(row, symbol_keys)
            if not symbol:
                return None

            company = self._get_value_by_keys(row, company_keys) or "Unknown"
            sector = self._get_value_by_keys(row, sector_keys) or "Unknown"
            industry = self._get_value_by_keys(row, industry_keys) or "Unknown"
            country = self._get_value_by_keys(row, country_keys) or "USA"

            # Parse numeric fields with error handling
            market_cap = self._parse_market_cap(self._get_value_by_keys(row, market_cap_keys))
            pe_ratio = self._parse_float(self._get_value_by_keys(row, pe_keys))
            price = self._parse_decimal(self._get_value_by_keys(row, price_keys))
            change = self._parse_float(self._get_value_by_keys(row, change_keys))
            volume = self._parse_volume(self._get_value_by_keys(row, volume_keys))

            return FinVizData(
                ticker=symbol.upper(),
                symbol=symbol.upper(),
                company=company,
                sector=sector,
                industry=industry,
                country=country,
                market_cap=market_cap,
                pe_ratio=pe_ratio,
                price=price,
                change=change,
                volume=volume,
                timestamp=timestamp
            )

        except Exception as e:
            logger.warning(f"Failed to clean row data for {row.get('Ticker', 'unknown')}: {e}")
            return None

    def _get_value_by_keys(self, row: Dict[str, str], keys: List[str]) -> Optional[str]:
        """Get value from row using multiple possible keys."""
        for key in keys:
            if key in row and row[key]:
                return row[key].strip()
        return None

    def _parse_float(self, value: Optional[str]) -> Optional[float]:
        """Parse string to float with error handling."""
        if not value or value in ['-', 'N/A', '']:
            return None
        try:
            # Remove percentage signs and commas
            cleaned = value.replace('%', '').replace(',', '')
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def _parse_decimal(self, value: Optional[str]) -> Optional[Decimal]:
        """Parse string to Decimal with error handling."""
        if not value or value in ['-', 'N/A', '']:
            return None
        try:
            # Remove currency symbols and commas
            cleaned = value.replace('$', '').replace(',', '')
            return Decimal(cleaned)
        except (ValueError, TypeError, Exception):
            return None

    def _parse_volume(self, value: Optional[str]) -> Optional[int]:
        """Parse volume string to integer."""
        if not value or value in ['-', 'N/A', '']:
            return None
        try:
            # Handle volume suffixes (K, M, B)
            cleaned = value.replace(',', '').upper()

            if cleaned.endswith('K'):
                return int(float(cleaned[:-1]) * 1000)
            elif cleaned.endswith('M'):
                return int(float(cleaned[:-1]) * 1000000)
            elif cleaned.endswith('B'):
                return int(float(cleaned[:-1]) * 1000000000)
            else:
                return int(float(cleaned))

        except (ValueError, TypeError):
            return None

    def _parse_market_cap(self, value: Optional[str]) -> Optional[Decimal]:
        """Parse market cap string to Decimal with error handling."""
        if not value or value in ['-', 'N/A', '']:
            return None
        try:
            # Remove currency symbols and commas
            cleaned = value.replace('$', '').replace(',', '').upper()

            # Handle market cap suffixes (K, M, B, T)
            if cleaned.endswith('K'):
                return Decimal(str(float(cleaned[:-1]) * 1000))
            elif cleaned.endswith('M'):
                return Decimal(str(float(cleaned[:-1]) * 1000000))
            elif cleaned.endswith('B'):
                return Decimal(str(float(cleaned[:-1]) * 1000000000))
            elif cleaned.endswith('T'):
                return Decimal(str(float(cleaned[:-1]) * 1000000000000))
            else:
                return Decimal(cleaned)

        except (ValueError, TypeError, Exception):
            return None

    async def get_top_momentum_stocks(
        self,
        limit: int = 20,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> FinVizScreenerResult:
        """
        Get top momentum stocks using default parameters.

        Args:
            limit: Maximum number of stocks to return
            custom_params: Custom parameters to override defaults

        Returns:
            FinVizScreenerResult with top momentum stocks
        """
        params = self.default_params.copy()

        if custom_params:
            for key, value in custom_params.items():
                if hasattr(params, key):
                    setattr(params, key, value)

        return await self.fetch_screener_data(params, limit)

    async def get_high_volume_breakouts(
        self,
        limit: int = 15,
        min_volume_ratio: float = 2.0
    ) -> FinVizScreenerResult:
        """
        Get stocks with high volume breakouts.

        Args:
            limit: Maximum number of stocks to return
            min_volume_ratio: Minimum volume ratio vs average

        Returns:
            FinVizScreenerResult with breakout candidates
        """
        params = FinVizScreenerParams(
            market_cap_min="50M",
            market_cap_max=None,
            avg_volume_min="500K",
            current_volume_min="1000K",
            price_min=3.0,
            price_max=100.0,
            above_sma20=True,
            weekly_volatility_min=8.0,
            custom_filters={
                "volume_ratio": f"sh_relvol_o{min_volume_ratio}"
            }
        )

        return await self.fetch_screener_data(params, limit)

    async def get_gappers(
        self,
        limit: int = 15,
        min_gap_percent: float = 5.0
    ) -> FinVizScreenerResult:
        """
        Get stocks with significant gaps.

        Args:
            limit: Maximum number of stocks to return
            min_gap_percent: Minimum gap percentage

        Returns:
            FinVizScreenerResult with gap stocks
        """
        params = FinVizScreenerParams(
            market_cap_min="10M",
            market_cap_max=None,
            avg_volume_min="200K",
            current_volume_min="500K",
            price_min=2.0,
            price_max=50.0,
            above_sma20=False,
            weekly_volatility_min=None,
            custom_filters={
                "gap_up": f"ta_gap_u{min_gap_percent}",
                "premarket_high": "sh_price_hi52w"
            }
        )

        return await self.fetch_screener_data(params, limit)

    async def get_stable_growth_stocks(
        self,
        limit: int = 15
    ) -> FinVizScreenerResult:
        """
        Get stable growth stocks with consistent performance.
        Conservative approach for steady returns.

        Args:
            limit: Maximum number of stocks to return

        Returns:
            FinVizScreenerResult with stable growth stocks
        """
        params = FinVizScreenerParams(
            market_cap_min="100M",
            market_cap_max=None,
            avg_volume_min="200K",
            current_volume_min="100K",
            price_min=5.0,
            price_max=200.0,
            above_sma20=True,
            weekly_volatility_min=1.0,  # Very low volatility requirement
            custom_filters={
                "earnings_growth": "fa_epsqoq_pos",  # Positive earnings growth
                "revenue_growth": "fa_salesqoq_pos"  # Positive revenue growth
            }
        )

        return await self.fetch_screener_data(params, limit)

    async def get_value_stocks(
        self,
        limit: int = 15
    ) -> FinVizScreenerResult:
        """
        Get undervalued stocks with good fundamentals.
        Value investing approach.

        Args:
            limit: Maximum number of stocks to return

        Returns:
            FinVizScreenerResult with value stocks
        """
        params = FinVizScreenerParams(
            market_cap_min="50M",
            market_cap_max=None,
            avg_volume_min="100K",
            current_volume_min="50K",
            price_min=2.0,
            price_max=100.0,
            above_sma20=False,  # Don't require above SMA for value plays
            weekly_volatility_min=None,  # No volatility requirement
            custom_filters={
                "pe_low": "fa_pe_low",  # Low P/E ratio
                "pb_low": "fa_pb_low",  # Low P/B ratio
                "debt_low": "fa_debteq_low"  # Low debt-to-equity
            }
        )

        return await self.fetch_screener_data(params, limit)

    async def get_dividend_stocks(
        self,
        limit: int = 10,
        min_yield: float = 2.0
    ) -> FinVizScreenerResult:
        """
        Get dividend-paying stocks for income strategy.

        Args:
            limit: Maximum number of stocks to return
            min_yield: Minimum dividend yield percentage

        Returns:
            FinVizScreenerResult with dividend stocks
        """
        params = FinVizScreenerParams(
            market_cap_min="500M",  # Larger, more stable companies
            market_cap_max=None,
            avg_volume_min="100K",
            current_volume_min="50K",
            price_min=10.0,
            price_max=500.0,
            above_sma20=False,
            weekly_volatility_min=None,
            custom_filters={
                "dividend_yield": f"fa_div_o{min_yield}",  # Dividend yield over min_yield%
                "payout_ratio": "fa_payoutratio_low"  # Sustainable payout ratio
            }
        )

        return await self.fetch_screener_data(params, limit)

    def get_last_scan_time(self) -> datetime:
        """Get timestamp of last successful scan."""
        if hasattr(self._rate_limiter, 'last_call') and self._rate_limiter.last_call > 0:
            return datetime.fromtimestamp(self._rate_limiter.last_call)
        return None

    def can_scan_now(self) -> bool:
        """Check if we can perform a scan now without rate limiting."""
        if self._rate_limiter.last_call == 0:
            return True

        time_since_last = time.time() - self._rate_limiter.last_call
        return time_since_last >= self._rate_limiter.min_interval

    def time_until_next_scan(self) -> float:
        """Get seconds until next scan is allowed."""
        if self.can_scan_now():
            return 0.0

        time_since_last = time.time() - self._rate_limiter.last_call
        return self._rate_limiter.min_interval - time_since_last

    async def validate_connection(self) -> bool:
        """
        Validate connection to FinViz by making a simple request.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Use minimal parameters for validation
            test_params = FinVizScreenerParams(
                market_cap_min="1B",
                market_cap_max=None,
                avg_volume_min=None,
                current_volume_min=None,
                price_min=None,
                price_max=None,
                above_sma20=False,
                weekly_volatility_min=None,
                custom_filters={"sector": "sec_technology"}
            )

            session = await self._get_session()
            request_params = test_params.to_finviz_params()

            async with session.get(
                self.export_url,
                params=request_params,
                timeout=aiohttp.ClientTimeout(total=10.0)
            ) as response:
                if response.status == 200:
                    content = await response.text()
                    # Check if we got valid CSV content
                    return content.startswith('No.,Ticker') or 'Ticker' in content[:100]
                return False

        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    async def get_sector_leaders(
        self,
        sector: str,
        limit: int = 10
    ) -> FinVizScreenerResult:
        """
        Get leading stocks from a specific sector.

        Args:
            sector: Sector name
            limit: Maximum number of stocks to return

        Returns:
            FinVizScreenerResult with sector leaders
        """
        # Map common sector names to FinViz codes
        sector_map = {
            "technology": "sec_technology",
            "healthcare": "sec_healthcare",
            "financial": "sec_financial",
            "energy": "sec_energy",
            "consumer": "sec_consumer",
            "industrial": "sec_industrial",
            "materials": "sec_basicmaterials",
            "utilities": "sec_utilities",
            "real_estate": "sec_realestate",
            "communication": "sec_communicationservices"
        }

        sector_code = sector_map.get(sector.lower(), f"sec_{sector.lower()}")

        params = FinVizScreenerParams(
            market_cap_min="100M",
            market_cap_max=None,
            avg_volume_min="500K",
            current_volume_min=None,
            price_min=5.0,
            price_max=None,
            above_sma20=True,
            weekly_volatility_min=None,
            custom_filters={
                "sector": sector_code,
                "performance": "ta_perf_1w_o5"  # 1-week performance > 5%
            }
        )

        return await self.fetch_screener_data(params, limit)

    def create_custom_screener(
        self,
        **kwargs
    ) -> FinVizScreenerParams:
        """
        Create custom screener parameters.

        Args:
            **kwargs: Parameter overrides

        Returns:
            FinVizScreenerParams with custom configuration
        """
        params_dict = self.default_params.dict()
        params_dict.update(kwargs)
        return FinVizScreenerParams(**params_dict)


# Convenience functions for common screening strategies
async def scan_momentum_stocks(
    screener: FinVizScreener,
    limit: int = 20
) -> List[str]:
    """
    Quick scan for momentum stocks returning just ticker symbols.

    Args:
        screener: FinVizScreener instance
        limit: Maximum number of tickers to return

    Returns:
        List of ticker symbols
    """
    result = await screener.get_top_momentum_stocks(limit)
    return [stock.symbol for stock in result.data]


async def scan_volume_leaders(
    screener: FinVizScreener,
    limit: int = 15
) -> List[Dict[str, Any]]:
    """
    Scan for volume leaders with essential data.

    Args:
        screener: FinVizScreener instance
        limit: Maximum number of stocks to return

    Returns:
        List of dictionaries with ticker data
    """
    result = await screener.get_high_volume_breakouts(limit)

    return [
        {
            'symbol': stock.symbol,
            'price': float(stock.price) if stock.price else 0.0,
            'change': stock.change or 0.0,
            'volume': stock.volume or 0,
            'sector': stock.sector,
            'market_cap': stock.market_cap
        }
        for stock in result.data
    ]


# Example usage patterns
if __name__ == "__main__":
    async def main():
        async with FinVizScreener() as screener:
            # Test connection
            if not await screener.validate_connection():
                print("Failed to connect to FinViz")
                return

            # Get momentum stocks
            result = await screener.get_top_momentum_stocks(limit=10)
            print(f"Found {result.total_count} momentum stocks:")

            for stock in result.data[:5]:  # Show top 5
                print(f"  {stock.symbol}: ${stock.price} ({stock.change:+.2f}%) Vol: {stock.volume:,}")

            # Get technology sector leaders
            tech_result = await screener.get_sector_leaders("technology", limit=5)
            print("\nTop 5 technology stocks:")

            for stock in tech_result.data:
                print(f"  {stock.symbol}: ${stock.price} in {stock.industry}")

    asyncio.run(main())
