"""Enable 64-bit precision for iSCAT tests."""

import jax

jax.config.update("jax_enable_x64", True)
