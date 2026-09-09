"""Small helpers for explicit H2O cluster ownership."""

from pathlib import Path

import h2o


def init_h2o(max_mem_size=None):
    """Initialize H2O and return its local-server handle only when started here."""
    kwargs = {"max_mem_size": max_mem_size} if max_mem_size else {}
    h2o.init(**kwargs)
    connection = h2o.connection()
    local_server = connection.local_server if connection is not None else None
    if local_server is not None and local_server.is_running():
        return local_server
    return None


def shutdown_h2o_if_owned(local_server, keep_cluster: bool) -> None:
    """Stop only the exact local H2O process launched by ``init_h2o``."""
    if local_server is None or keep_cluster:
        return
    connection = h2o.connection()
    if (
        connection is not None
        and connection.local_server is local_server
        and local_server.is_running()
    ):
        h2o.cluster().shutdown(prompt=False)
    else:
        local_server.shutdown()


def save_h2o_model(model, output_dir: Path, filename: str) -> Path:
    """Save a model under ``output_dir`` using a predictable local filename."""
    filename_path = Path(filename)
    if not filename or filename_path.name != filename or filename in {".", ".."}:
        raise ValueError("H2O model filename must be a single nonempty path component")

    resolved_output_dir = Path(output_dir).resolve()
    expected_path = resolved_output_dir / filename
    saved_path = Path(
        h2o.save_model(
            model,
            path=str(resolved_output_dir),
            filename=filename,
            force=True,
        )
    ).resolve()
    if saved_path != expected_path:
        raise RuntimeError(
            f"H2O saved the model outside its expected path: {saved_path} "
            f"(expected {expected_path})"
        )
    return saved_path
