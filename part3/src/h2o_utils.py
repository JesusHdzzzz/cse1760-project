"""Small helpers for explicit H2O cluster ownership."""

import h2o


def init_h2o(max_mem_size=None) -> bool:
    """Initialize or attach to H2O and return whether this process started it."""
    owned = h2o.connection() is None
    kwargs = {"max_mem_size": max_mem_size} if max_mem_size else {}
    h2o.init(**kwargs)
    return owned


def shutdown_h2o_if_owned(owned: bool, keep_cluster: bool) -> None:
    if owned and not keep_cluster and h2o.connection() is not None:
        h2o.cluster().shutdown(prompt=False)
