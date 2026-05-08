#!/usr/bin/env python3
"""Start a persistent vLLM OpenAI-compatible server.

This script intentionally does not kill existing vLLM processes. If the target
host/port is already in use, it exits and tells the user to reuse that server or
choose another port.
"""

import argparse
import os
import socket
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
COMPAT_DIR = os.path.join(SCRIPT_DIR, "compat")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preload a persistent vLLM server for ReSCORE runs."
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model name or local model path.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument(
        "--served_model_name",
        default=None,
        help="Model alias exposed by the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="vLLM tensor parallel size.",
    )
    parser.add_argument(
        "--dtype",
        default="half",
        help="Model dtype. Use 'half' on RTX 2080 Ti / Turing GPUs.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory used by vLLM.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Maximum model context length / KV-cache length.",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=20,
        help="Maximum concurrent sequences scheduled by vLLM.",
    )
    parser.add_argument(
        "--swap_space",
        type=float,
        default=0,
        help="CPU swap space in GiB per GPU. Default keeps vLLM GPU-only.",
    )
    parser.add_argument(
        "--cpu_offload_gb",
        type=float,
        default=0,
        help="CPU offload size in GiB. Default keeps vLLM GPU-only.",
    )
    parser.add_argument(
        "--enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable CUDA graph capture to reduce VRAM spikes.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override, e.g. '5,6'.",
    )
    return parser.parse_args()


def port_is_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def main():
    args = parse_args()

    if port_is_open(args.host, args.port):
        print(
            f"vLLM server may already be running at http://{args.host}:{args.port}.",
            file=sys.stderr,
        )
        print("This script does not kill or replace existing vLLM processes.", file=sys.stderr)
        return 0

    env = os.environ.copy()
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    pythonpath_items = [
        COMPAT_DIR,
        REPO_ROOT,
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(item for item in pythonpath_items if item)
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    served_model_name = args.served_model_name or args.model
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--served-model-name",
        served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--swap-space",
        str(args.swap_space),
        "--cpu-offload-gb",
        str(args.cpu_offload_gb),
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")

    print("Starting persistent vLLM server:")
    print(" ".join(command))
    print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<inherited>')}")
    print(f"Endpoint: http://{args.host}:{args.port}/v1")
    print("Leave this process running while training/inference clients use the server.")

    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
