#!/usr/bin/env python3
import argparse
import getpass
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List


DEFAULT_PATTERNS = (
    "VllmWorkerProcess",
    "vllm",
    "multiproc_worker_utils",
)

RESCORE_RUN_PATTERNS = (
    "source.run.inference",
    "source.run.train",
)


@dataclass
class ProcessInfo:
    pid: int
    ppid: int
    stat: str
    command: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "List or kill vLLM-related processes owned by the current user. "
            "Default mode is dry-run; pass --yes to send a signal."
        )
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually send the signal. Without this flag the script only lists targets.",
    )
    parser.add_argument(
        "--signal",
        choices=["TERM", "KILL", "INT"],
        default="TERM",
        help="Signal to send when --yes is set.",
    )
    parser.add_argument(
        "--include-rescore-runs",
        action="store_true",
        help="Also match parent python processes running source.run.inference/source.run.train.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Extra substring pattern to match in process command lines. Can be repeated.",
    )
    return parser.parse_args()


def list_current_user_processes() -> List[ProcessInfo]:
    user = getpass.getuser()
    result = subprocess.run(
        ["ps", "-u", user, "-o", "pid=", "-o", "ppid=", "-o", "stat=", "-o", "args="],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    processes = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=3)
        if len(parts) < 4:
            continue
        pid, ppid, stat, command = parts
        processes.append(
            ProcessInfo(
                pid=int(pid),
                ppid=int(ppid),
                stat=stat,
                command=command,
            )
        )
    return processes


def matches(command: str, patterns: Iterable[str]) -> bool:
    lowered_command = command.lower()
    return any(pattern.lower() in lowered_command for pattern in patterns)


def main():
    args = parse_args()
    patterns = list(DEFAULT_PATTERNS)
    if args.include_rescore_runs:
        patterns.extend(RESCORE_RUN_PATTERNS)
    patterns.extend(args.pattern)

    current_pid = os.getpid()
    targets = [
        process
        for process in list_current_user_processes()
        if process.pid != current_pid and matches(process.command, patterns)
    ]

    if not targets:
        print("No matching vLLM-related processes found.")
        return 0

    print("Matched processes:")
    for process in targets:
        print(
            f"  pid={process.pid} ppid={process.ppid} stat={process.stat} "
            f"cmd={process.command}"
        )

    signal_number = getattr(signal, f"SIG{args.signal}")
    if not args.yes:
        print(
            "\nDry-run only. Re-run with --yes to kill these processes, "
            "for example: python script/kill_vllm_processes.py --yes"
        )
        return 0

    for process in targets:
        try:
            os.kill(process.pid, signal_number)
            print(f"Sent SIG{args.signal} to pid={process.pid}")
        except ProcessLookupError:
            print(f"pid={process.pid} already exited")
        except PermissionError:
            print(f"Permission denied for pid={process.pid}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
