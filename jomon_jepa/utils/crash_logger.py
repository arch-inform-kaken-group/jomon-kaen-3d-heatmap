import datetime
import faulthandler
import os
import platform
import socket
import sys
import threading
import traceback
from typing import Callable


def _get_rank_tag() -> str:
    rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0"
    return f"rank{rank}"


def write_crash_report(
    log_dir: str,
    script_name: str,
    exc_type,
    exc_value,
    exc_tb,
    extra_context: dict | None = None,
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rank_tag = _get_rank_tag()
    log_path = os.path.join(log_dir, f"crash_{script_name}_{rank_tag}_{ts}.txt")

    lines = [
        "=" * 80,
        f"CRASH REPORT | {script_name} | {ts} | {rank_tag}",
        "=" * 80,
        f"Host:       {socket.gethostname()}",
        f"Platform:   {platform.platform()}",
        f"Python:     {sys.version.splitlines()[0]}",
        f"CWD:        {os.getcwd()}",
        f"Command:    {' '.join(sys.argv)}",
    ]

    if extra_context:
        lines.append("-" * 80)
        lines.append("Context:")
        for k, v in extra_context.items():
            lines.append(f"  {k}: {v}")

    lines.append("-" * 80)
    lines.append("Traceback:")
    lines.extend(traceback.format_exception(exc_type, exc_value, exc_tb))
    lines.append("=" * 80)

    report = "\n".join(lines) + "\n"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.flush()
        os.fsync(f.fileno())

    sys.stderr.write(f"\n[CrashLogger] Fatal error dumped to: {log_path}\n")
    sys.stderr.flush()
    return log_path


def run_with_crash_logger(
    main_fn: Callable[[], None],
    log_dir: str = "./crash_logs",
    script_name: str = "run",
    extra_context: dict | None = None,
) -> None:
    os.makedirs(log_dir, exist_ok=True)
    rank_tag = _get_rank_tag()
    native_fault_path = os.path.join(log_dir, f"native_fault_{script_name}_{rank_tag}.txt")
    fault_file = open(native_fault_path, "a", encoding="utf-8")
    faulthandler.enable(file=fault_file, all_threads=True)

    def _sys_hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        write_crash_report(log_dir, script_name, exc_type, exc_value, exc_tb, extra_context)
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    def _thread_hook(args: threading.ExceptHookArgs):
        write_crash_report(
            log_dir,
            f"{script_name}_thread_{args.thread.name if args.thread else 'unknown'}",
            args.exc_type,
            args.exc_value,
            args.exc_traceback,
            extra_context,
        )

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook

    try:
        main_fn()
    except KeyboardInterrupt:
        raise
    except BaseException:
        exc_type, exc_value, exc_tb = sys.exc_info()
        write_crash_report(log_dir, script_name, exc_type, exc_value, exc_tb, extra_context)
        raise
    finally:
        fault_file.flush()
        fault_file.close()
        # Clean up empty native fault file if no C/CUDA segfault occurred
        if os.path.exists(native_fault_path) and os.path.getsize(native_fault_path) == 0:
            os.remove(native_fault_path)