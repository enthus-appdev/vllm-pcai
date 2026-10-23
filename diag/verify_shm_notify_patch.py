"""CPU-only behavioral tripwire for the vendored SHM notify-backlog fix."""

import ast
import pathlib
import types
from unittest.mock import Mock


def verify(source: str) -> None:
    tree = ast.parse(source)
    condition = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SpinCondition"
    )
    method = next(
        n
        for n in condition.body
        if isinstance(n, ast.FunctionDef) and n.name == "record_read"
    )
    interval = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "SHM_NOTIFY_POLL_INTERVAL_S"
            for t in n.targets
        )
    )
    clock = Mock(return_value=0.5)
    namespace = {
        "time": types.SimpleNamespace(monotonic=clock),
        "zmq": types.SimpleNamespace(NOBLOCK=1),
    }
    exec(
        compile(
            ast.Module(body=[interval, method], type_ignores=[]), "shm-tripwire", "exec"
        ),
        namespace,
    )
    assert namespace["SHM_NOTIFY_POLL_INTERVAL_S"] == 2.0
    socket = Mock()
    socket.poll.return_value = True
    reader = types.SimpleNamespace(
        last_read=0.0, last_notify_poll=0.0, local_notify_socket=socket
    )
    namespace["record_read"](reader)
    socket.poll.assert_not_called()
    clock.return_value = 2.0
    namespace["record_read"](reader)
    socket.poll.assert_called_once_with(timeout=0)
    socket.recv.assert_called_once_with(flags=1, copy=False)
    assert reader.last_notify_poll == reader.last_read == 2.0
    socket.reset_mock()
    clock.return_value = 2.5
    namespace["record_read"](reader)
    socket.poll.assert_not_called()
    clock.return_value = 4.0
    socket.poll.return_value = False
    namespace["record_read"](reader)
    socket.poll.assert_called_once_with(timeout=0)
    socket.recv.assert_not_called()
    assert reader.last_notify_poll == 4.0


if __name__ == "__main__":
    import sys

    source_root = pathlib.Path(sys.argv[1])
    verify(
        (source_root / "distributed/device_communicators/shm_broadcast.py").read_text()
    )
    print(
        "SHM busy-reader poll interval, drain and empty-socket behavior OK (CPU source isolation)"
    )
