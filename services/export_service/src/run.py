#!/usr/bin/env python
"""
Export Service Runner
Simple startup script for the export service
"""

import logging
import os
import sys

import uvicorn

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the export service"""
    try:
        # Add app directory to path
        sys.path.insert(0, "/app")

        # Import the FastAPI app
        from src.main import app

        # Get configuration from environment
        host = "0.0.0.0"
        port = int(os.getenv("SERVICE_PORT", "9106"))
        log_level = os.getenv("LOG_LEVEL", "info").lower()

        logger.info(f"Starting Export Service on {host}:{port}")
        logger.info(f"Log level: {log_level}")

        # Run the service
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level,
            access_log=True,
            log_config={
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "default": {
                        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    },
                },
                "handlers": {
                    "default": {
                        "formatter": "default",
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stdout",
                    },
                },
                "loggers": {
                    "uvicorn": {"handlers": ["default"], "level": log_level.upper()},
                    "uvicorn.error": {"level": log_level.upper()},
                    "uvicorn.access": {
                        "handlers": ["default"],
                        "level": log_level.upper(),
                        "propagate": False,
                    },
                },
            },
        )

    except ImportError as e:
        logger.error(f"Failed to import app: {e}")
        logger.error("Make sure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start Export Service: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
