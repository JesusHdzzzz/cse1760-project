from types import SimpleNamespace

import h2o_utils


class FakeLocalServer:
    def __init__(self):
        self.shutdown_calls = 0

    def is_running(self):
        return True

    def shutdown(self):
        self.shutdown_calls += 1


class FakeCluster:
    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self, prompt):
        assert prompt is False
        self.shutdown_calls += 1


def test_attached_cluster_is_not_owned_or_shutdown(monkeypatch):
    connection = SimpleNamespace(local_server=None)
    monkeypatch.setattr(h2o_utils.h2o, "init", lambda **kwargs: None)
    monkeypatch.setattr(h2o_utils.h2o, "connection", lambda: connection)
    monkeypatch.setattr(
        h2o_utils.h2o,
        "cluster",
        lambda: (_ for _ in ()).throw(AssertionError("must not shut down")),
    )

    local_server = h2o_utils.init_h2o("1G")
    h2o_utils.shutdown_h2o_if_owned(local_server, keep_cluster=False)

    assert local_server is None


def test_self_started_cluster_is_shutdown(monkeypatch):
    server = FakeLocalServer()
    cluster = FakeCluster()
    connection = SimpleNamespace(local_server=server)
    monkeypatch.setattr(h2o_utils.h2o, "init", lambda **kwargs: None)
    monkeypatch.setattr(h2o_utils.h2o, "connection", lambda: connection)
    monkeypatch.setattr(h2o_utils.h2o, "cluster", lambda: cluster)

    local_server = h2o_utils.init_h2o("1G")
    h2o_utils.shutdown_h2o_if_owned(local_server, keep_cluster=False)

    assert local_server is server
    assert cluster.shutdown_calls == 1
    assert server.shutdown_calls == 0


def test_keep_cluster_preserves_self_started_cluster(monkeypatch):
    server = FakeLocalServer()
    cluster = FakeCluster()
    connection = SimpleNamespace(local_server=server)
    monkeypatch.setattr(h2o_utils.h2o, "init", lambda **kwargs: None)
    monkeypatch.setattr(h2o_utils.h2o, "connection", lambda: connection)
    monkeypatch.setattr(h2o_utils.h2o, "cluster", lambda: cluster)

    local_server = h2o_utils.init_h2o()
    h2o_utils.shutdown_h2o_if_owned(local_server, keep_cluster=True)

    assert local_server is server
    assert cluster.shutdown_calls == 0
    assert server.shutdown_calls == 0
