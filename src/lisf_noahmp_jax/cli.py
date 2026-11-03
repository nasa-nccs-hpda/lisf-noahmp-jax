"""Command line entry point: `lisf-noahmp-jax [-C CASE_DIR] lis.config` runs the JAX port from a LIS configuration file."""
from .driver.lis import main

__all__ = ["main"]
