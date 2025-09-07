"""
Ollama client for local AI testing.

This module provides an interface to Ollama for running local LLM models
as an alternative to expensive cloud-based AI services for testing purposes.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class OllamaResponse:
    """Response from Ollama API."""

    content: str
    model: str
    tokens_used: int
    response_time: float
    cost: float = 0.0  # Local models are free


class OllamaClient:
    """Client for interacting with local Ollama instance."""

    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "llama3.1:latest"
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            model: Model name to use
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def query(
        self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7
    ) -> OllamaResponse:
        """
        Query Ollama with a prompt.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            OllamaResponse object
        """
        await self._ensure_session()

        start_time = time.time()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            await self._ensure_session()
            if self.session is None:
                raise RuntimeError("Failed to initialize session")
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                response.raise_for_status()
                result = await response.json()

                response_time = time.time() - start_time

                return OllamaResponse(
                    content=result.get("response", ""),
                    model=self.model,
                    tokens_used=result.get("eval_count", 0),
                    response_time=response_time,
                    cost=0.0,  # Local models are free
                )

        except asyncio.TimeoutError:
            logger.error("Ollama request timed out after 120 seconds")
            raise Exception("Ollama request timed out")
        except aiohttp.ClientError as e:
            logger.error(f"Ollama client error: {e}")
            raise Exception(f"Failed to connect to Ollama: {e}")
        except Exception as e:
            logger.error(f"Unexpected error querying Ollama: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if Ollama server is healthy and model is available.

        Returns:
            True if healthy, False otherwise
        """
        await self._ensure_session()

        try:
            # Check server health
            await self._ensure_session()
            if self.session is None:
                raise RuntimeError("Failed to initialize session")
            async with self.session.get(
                f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                models_data = await response.json()

                # Check if our model is available
                available_models = [
                    model["name"] for model in models_data.get("models", [])
                ]
                if self.model not in available_models:
                    logger.warning(
                        f"Model {self.model} not found. Available models: {available_models}"
                    )
                    return False

                return True

        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def list_models(self) -> List[str]:
        """
        List available models on Ollama server.

        Returns:
            List of model names
        """
        await self._ensure_session()

        try:
            await self._ensure_session()
            if self.session is None:
                raise RuntimeError("Failed to initialize session")
            async with self.session.get(
                f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                models_data = await response.json()
                return [model["name"] for model in models_data.get("models", [])]

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model to Ollama server.

        Args:
            model_name: Name of model to pull

        Returns:
            True if successful, False otherwise
        """
        await self._ensure_session()

        try:
            payload = {"name": model_name}

            await self._ensure_session()
            if self.session is None:
                raise RuntimeError("Failed to initialize session")
            async with self.session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=600
                ),  # 10 minutes for model download
            ) as response:
                response.raise_for_status()

                # Stream the response to track progress
                async for line in response.content:
                    if line:
                        try:
                            status = json.loads(line)
                            if "status" in status:
                                logger.info(f"Model pull status: {status['status']}")
                        except json.JSONDecodeError:
                            continue

                return True

        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class OllamaAIStrategy:
    """AI Strategy implementation using Ollama for local testing."""

    def __init__(
        self,
        ollama_url: str = "http://192.168.1.133:11434",
        model: str = "llama3.1:latest",
    ):
        """
        Initialize Ollama AI Strategy.

        Args:
            ollama_url: Ollama server URL
            model: Model name to use
        """
        self.client = OllamaClient(ollama_url, model)
        self.model = model

    async def analyze_market_data(
        self, market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze market data using Ollama.

        Args:
            market_context: Market data context

        Returns:
            Trading decision dictionary
        """
        # Health check first
        if not await self.client.health_check():
            raise Exception("Ollama server is not healthy or model is not available")

        # Build prompt for trading analysis
        prompt = self._build_trading_prompt(market_context)

        try:
            response = await self.client.query(prompt, max_tokens=1000, temperature=0.3)

            # Parse the response to extract trading decision
            decision = self._parse_trading_response(response.content)

            # Add metadata
            decision["model_used"] = self.model
            decision["response_time"] = response.response_time
            decision["tokens_used"] = response.tokens_used
            decision["cost"] = 0.0  # Free for local models
            decision["timestamp"] = datetime.now().isoformat()

            return decision

        except Exception as e:
            logger.error(f"Failed to analyze market data with Ollama: {e}")
            raise

    def _build_trading_prompt(self, market_context: Dict[str, Any]) -> str:
        """Build trading analysis prompt from market context."""
        ticker = market_context.get("ticker", "UNKNOWN")
        current_price = market_context.get("current_price", 0)
        daily_change = market_context.get("daily_change", 0)
        volume = market_context.get("volume", 0)
        rsi = market_context.get("technical_indicators", {}).get("rsi", 0)

        prompt = f"""You are an expert trading analyst. Analyze the following market data and provide a trading recommendation.

TICKER: {ticker}
Current Price: ${current_price}
Daily Change: {daily_change}%
Volume: {volume}
RSI: {rsi}

Technical Analysis:
- RSI of {rsi} indicates {'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'} conditions
- Daily change of {daily_change}% shows {'strong bullish' if daily_change > 2 else 'strong bearish' if daily_change < -2 else 'sideways'} momentum

Provide your recommendation in this JSON format:
{{
    "decision": "BUY" or "SELL" or "HOLD",
    "confidence": number between 0-100,
    "entry_price": price or null,
    "stop_loss": price or null,
    "take_profit": price or null,
    "reasoning": "brief explanation of your decision",
    "risk_level": "LOW" or "MEDIUM" or "HIGH"
}}

Respond only with the JSON, no additional text."""

        return prompt

    def _parse_trading_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Ollama response to extract trading decision."""
        try:
            # Try to extract JSON from response
            response_text = response_text.strip()

            # Find JSON in the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                decision = json.loads(json_str)

                # Validate required fields
                required_fields = ["decision", "confidence", "reasoning"]
                for field in required_fields:
                    if field not in decision:
                        decision[field] = None

                # Ensure decision is valid
                if decision["decision"] not in ["BUY", "SELL", "HOLD"]:
                    decision["decision"] = "HOLD"

                # Ensure confidence is valid
                if (
                    not isinstance(decision["confidence"], (int, float))
                    or decision["confidence"] < 0
                    or decision["confidence"] > 100
                ):
                    decision["confidence"] = 50

                return decision

            else:
                # Fallback: parse from text
                return self._parse_text_response(response_text)

        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, attempting text parsing")
            return self._parse_text_response(response_text)
        except Exception as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            return {
                "decision": "HOLD",
                "confidence": 0,
                "reasoning": "Failed to parse AI response",
                "error": str(e),
            }

    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """Parse text response when JSON parsing fails."""
        response_lower = response_text.lower()

        # Determine decision
        if "buy" in response_lower and "buy" in response_lower[:200]:
            decision = "BUY"
        elif "sell" in response_lower and "sell" in response_lower[:200]:
            decision = "SELL"
        else:
            decision = "HOLD"

        # Try to extract confidence
        confidence = 50
        confidence_indicators = ["confidence", "certain", "sure"]
        for indicator in confidence_indicators:
            if indicator in response_lower:
                # Try to find numbers near confidence indicators
                import re

                pattern = rf"{indicator}[^\d]*(\d+)"
                match = re.search(pattern, response_lower)
                if match:
                    confidence = min(100, max(0, int(match.group(1))))
                    break

        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": (
                response_text[:200] + "..."
                if len(response_text) > 200
                else response_text
            ),
            "parsed_from_text": True,
        }

    async def close(self):
        """Close the Ollama client."""
        await self.client.close()
