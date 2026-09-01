import datetime
import os
import signal
import socket
import traceback
from contextlib import contextmanager

import pytest
import torch
import torch.multiprocessing as mp


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


def _kill_siblings():
    """SIGKILL every other child of our parent (the pytest process).

    NCCL collectives don't honor SIGTERM, so if one rank raises and we let
    mp.spawn's default cleanup run, it would block on join() until the NCCL
    timeout. Killing siblings here lets mp.spawn collect all exits and
    re-raise to the parent immediately.
    """
    my_pid = os.getpid()
    ppid = os.getppid()
    try:
        with open(f"/proc/{ppid}/task/{ppid}/children") as f:
            sibling_pids = [int(p) for p in f.read().split()]
    except OSError:
        return
    for pid in sibling_pids:
        if pid == my_pid:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _format_child_traceback(exc):
    """Format a child exception, dropping the entrypoint frame so the first
    line shown belongs to user code (not conftest's ``func(**func_args)``)."""
    tb = exc.__traceback__
    if tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    return "".join(traceback.format_exception(type(exc), exc, tb)).rstrip()


def _distributed_entrypoint(rank, world_size, port, backend, func, func_args, error_writer):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["LOCAL_RANK"] = str(rank)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=datetime.timedelta(seconds=600),
    )

    try:
        func(**func_args)
    except BaseException as e:
        # Forward the formatted traceback to the parent so it can surface the
        # *real* error rather than just ``ProcessRaisedException``. Only the
        # failing rank writes — siblings die without sending, so single-writer
        # Pipe semantics are safe.
        error_writer.send((rank, _format_child_traceback(e)))
        error_writer.close()
        _kill_siblings()
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@contextmanager
def _capture_child_tracebacks():
    """Collect ``(rank, traceback)`` tuples sent by failing children into the
    ``errors`` list yielded alongside the pipe writer.

    Re-raises the original ``mp.spawn`` exception only if no rank actually
    reported a traceback (e.g. the spawn machinery itself blew up).
    """
    reader, writer = mp.get_context("spawn").Pipe(duplex=False)
    errors = []
    try:
        try:
            yield writer, errors
        except (mp.ProcessRaisedException, mp.ProcessExitedException):
            # Close the parent's writer copy so ``recv`` raises ``EOFError``
            # once every (still-alive) child has closed its own writer end.
            writer.close()
            while True:
                try:
                    errors.append(reader.recv())
                except EOFError:
                    break
            if not errors:
                raise
    finally:
        reader.close()
        if not writer.closed:
            writer.close()


def _format_distributed_failure(errors, world_size):
    """Pretty-print failure sections, one per rank, framed by Unicode rules."""
    width = 72
    sep = "─" * width
    sections = []
    for rank, tb_text in sorted(errors):
        header = f" rank {rank} / {world_size} ".center(width, "─")
        sections.append(f"{header}\n{tb_text}\n{sep}")
    # First line becomes pytest's short-summary line; keep it self-contained.
    summary = f"Distributed test failed on {len(errors)}/{world_size} rank(s)"
    return "\n".join([summary, ""] + sections)


def pytest_configure(config):
    config.addinivalue_line("markers", "distributed(world_size, backend): mark test to run in distributed mode")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    dist_marker = pyfuncitem.get_closest_marker("distributed")

    if not dist_marker:
        return None

    world_size = dist_marker.kwargs.get("world_size", 2)
    backend = dist_marker.kwargs.get("backend", "nccl")
    assert not dist_marker.args

    test_func = pyfuncitem.obj
    func_args = pyfuncitem.funcargs

    if pyfuncitem.instance is not None:
        instance = pyfuncitem.instance

        def wrapper(**kwargs):
            return test_func(instance, **kwargs)
    else:
        wrapper = test_func

    with _capture_child_tracebacks() as (error_writer, errors):
        mp.spawn(
            _distributed_entrypoint,
            args=(world_size, _find_free_port(), backend, wrapper, func_args, error_writer),
            nprocs=world_size,
            join=True,
        )

    # Outside the except block — no ProcessRaisedException is active, so
    # pytest.fail won't chain ("During handling of the above exception ...").
    if errors:
        pytest.fail(_format_distributed_failure(errors, world_size), pytrace=False)

    return True
