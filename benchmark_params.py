#!/usr/bin/env python3
"""
Llama.cpp Parameter Benchmark Suite for MoE Models (IQ3 / IQ4)
===============================================================
Automated tester to find the sweet spot for:
  - -ngl (GPU offloaded layers)
  - --n-cpu-moe (MoE expert layers processed by CPU)
  - mmap vs --no-mmap
"""

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

DEFAULT_BASE_DIR = r"D:\Program\llama-b10054-bin-win-cuda-13.3-x64"

MODEL_CATALOG = {
    "ornith-iq3": {
        "name": "Ornith 1.5 35B (IQ3_M)",
        "path": r"model\Ornith-1.5-35B-A3B-abliterated-i1-GGUF\Ornith-1.5-35B-A3B-abliterated.i1-IQ3_M.gguf",
        "default_ngl": [30, 32, 33, 34],
        "default_moe": [22, 24, 26],
    },
    "ornith-iq4": {
        "name": "Ornith 1.5 35B (IQ4_XS)",
        "path": r"model\Ornith-1.5-35B-A3B-abliterated-i1-GGUF\Ornith-1.5-35B-A3B-abliterated.i1-IQ4_XS.gguf",
        "default_ngl": [26, 28, 30],
        "default_moe": [24, 26, 27],
    },
    "qwen-iq3": {
        "name": "Qwen 3.6 35B (IQ3_M)",
        "path": r"model\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf",
        "default_ngl": [30, 32, 33, 34],
        "default_moe": [22, 24, 26],
    },
    "qwen-iq4": {
        "name": "Qwen 3.6 35B (IQ4_XS)",
        "path": r"model\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf",
        "default_ngl": [26, 28, 30],
        "default_moe": [24, 26, 27],
    },
}

TEST_PROMPT = (
    "Explain in detail the differences between supervised and unsupervised learning, "
    "and provide two real-world examples."
)


def parse_llama_metrics(output: str) -> Dict[str, Optional[float]]:
    """Parse load time, prompt eval t/s, and generation eval t/s from llama-cli output."""
    metrics = {
        "load_time_ms": None,
        "prompt_eval_tokens": None,
        "prompt_eval_ms": None,
        "prompt_tps": None,
        "gen_eval_tokens": None,
        "gen_eval_ms": None,
        "gen_tps": None,
    }

    load_match = re.search(r"load time\s*=\s*([\d\.]+)\s*ms", output, re.IGNORECASE)
    if load_match:
        metrics["load_time_ms"] = float(load_match.group(1))

    prompt_match = re.search(
        r"prompt eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d\.]+)\s*tokens per second",
        output,
        re.IGNORECASE,
    )
    if prompt_match:
        metrics["prompt_eval_ms"] = float(prompt_match.group(1))
        metrics["prompt_eval_tokens"] = int(prompt_match.group(2))
        metrics["prompt_tps"] = float(prompt_match.group(3))

    eval_match = re.search(
        r"(?:eval time|generation time)\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens).*?([\d\.]+)\s*tokens per second",
        output,
        re.IGNORECASE,
    )
    if eval_match:
        metrics["gen_eval_ms"] = float(eval_match.group(1))
        metrics["gen_eval_tokens"] = int(eval_match.group(2))
        metrics["gen_tps"] = float(eval_match.group(3))

    if metrics["prompt_tps"] is None:
        tps_match = re.search(
            r"\[\s*Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s\s*\]",
            output,
            re.IGNORECASE,
        )
        if tps_match:
            metrics["prompt_tps"] = float(tps_match.group(1))
            metrics["gen_tps"] = float(tps_match.group(2))

    return metrics


def run_single_benchmark(
    exe_path: str,
    model_full_path: str,
    ngl: int,
    n_cpu_moe: int,
    mmap: bool,
    context_size: int = 16384,
    batch_size: int = 2048,
    ubatch_size: int = 1024,
    threads: int = 6,
    gen_tokens: int = 48,
    prompt: str = TEST_PROMPT,
    timeout_sec: int = 240,
) -> Dict[str, Any]:
    """Execute a single llama-cli test run with live elapsed timer."""

    cmd = [
        exe_path,
        "-m", model_full_path,
        "-ngl", str(ngl),
        "--n-cpu-moe", str(n_cpu_moe),
        "-c", str(context_size),
        "-b", str(batch_size),
        "-ub", str(ubatch_size),
        "-fa", "on",
        "-ctk", "q4_0",
        "-ctv", "q4_0",
        "-t", str(threads),
        "-tb", str(threads),
        "-n", str(gen_tokens),
        "-p", prompt,
        "--no-warmup",
    ]

    if not mmap:
        cmd.append("--no-mmap")

    start_wall = time.time()
    is_running = True

    # Live ticker thread to give visual feedback
    def ticker():
        while is_running:
            elapsed = time.time() - start_wall
            sys.stdout.write(f"\r  ⏱️  Running... ({elapsed:.1f}s) ")
            sys.stdout.flush()
            time.sleep(0.5)

    t = threading.Thread(target=ticker, daemon=True)
    t.start()

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="replace",
        )
        is_running = False
        t.join(timeout=1.0)
        # Clear ticker line
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()

        wall_time = time.time() - start_wall
        output = proc.stdout

        if proc.returncode != 0:
            return {
                "status": "ERROR",
                "returncode": proc.returncode,
                "error_snippet": output[-400:] if output else "Process exited with error",
                "wall_time": wall_time,
            }

        metrics = parse_llama_metrics(output)
        metrics["status"] = "OK"
        metrics["wall_time"] = wall_time
        return metrics

    except subprocess.TimeoutExpired:
        is_running = False
        t.join(timeout=1.0)
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()
        return {
            "status": "TIMEOUT",
            "returncode": -1,
            "error_snippet": f"Timed out after {timeout_sec}s",
            "wall_time": timeout_sec,
        }
    except Exception as e:
        is_running = False
        t.join(timeout=1.0)
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()
        return {
            "status": "EXCEPTION",
            "returncode": -1,
            "error_snippet": str(e),
            "wall_time": time.time() - start_wall,
        }


def print_table(results: List[Dict[str, Any]]):
    """Print nicely formatted benchmark comparison table."""
    valid_results = [r for r in results if r.get("status") == "OK" and r.get("gen_tps")]
    error_results = [r for r in results if r not in valid_results]

    valid_results.sort(key=lambda x: x["gen_tps"], reverse=True)

    header = (
        f"{'Rank':<5} | {'Model':<16} | {'-ngl':<5} | {'--n-cpu-moe':<11} | {'mmap':<8} | "
        f"{'Load(s)':<8} | {'Prompt t/s':<11} | {'Gen t/s':<9} | {'Status':<6}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print("🏆 BENCHMARK RESULTS (SORTED BY GENERATION SPEED)")
    print(separator)
    print(header)
    print(separator)

    for i, r in enumerate(valid_results, start=1):
        medal = "🥇 " if i == 1 else ("🥈 " if i == 2 else ("🥉 " if i == 3 else f"#{i:<2}"))
        load_s = f"{r['load_time_ms']/1000.0:.2f}s" if r.get("load_time_ms") else "-"
        p_tps = f"{r['prompt_tps']:.1f}" if r.get("prompt_tps") else "-"
        g_tps = f"{r['gen_tps']:.2f}" if r.get("gen_tps") else "-"
        mmap_str = "mmap" if r.get("mmap") else "no-mmap"

        print(
            f"{medal:<5} | {r.get('model_key', ''):<16} | {r['ngl']:<5} | {r['n_cpu_moe']:<11} | "
            f"{mmap_str:<8} | {load_s:<8} | {p_tps:<11} | {g_tps:<9} | {r['status']:<6}"
        )

    for r in error_results:
        mmap_str = "mmap" if r.get("mmap") else "no-mmap"
        print(
            f"{'FAIL':<5} | {r.get('model_key', ''):<16} | {r['ngl']:<5} | {r['n_cpu_moe']:<11} | "
            f"{mmap_str:<8} | {'-':<8} | {'-':<11} | {'-':<9} | {r['status']:<6}"
        )

    print(separator + "\n")


def generate_markdown_report(results: List[Dict[str, Any]], output_file: str):
    """Write comprehensive markdown benchmark report."""
    valid_results = [r for r in results if r.get("status") == "OK" and r.get("gen_tps")]
    valid_results.sort(key=lambda x: x["gen_tps"], reverse=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# 🚀 Llama.cpp Parameter Benchmark Report\n\n")
        f.write(f"Generated on: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n\n")

        if valid_results:
            best = valid_results[0]
            f.write("## 🌟 Best Configuration Summary\n\n")
            f.write(f"- **Model**: `{best.get('model_key')}` ({best.get('model_name')})\n")
            f.write(f"- **Top Generation Speed**: **`{best['gen_tps']:.2f} t/s`**\n")
            f.write(f"- **Prompt Speed**: `{best.get('prompt_tps', 0.0):.1f} t/s`\n")
            f.write(
                f"- **Optimal Parameters**: `-ngl {best['ngl']} --n-cpu-moe {best['n_cpu_moe']}` "
                f"({'mmap (default)' if best['mmap'] else '--no-mmap'})\n\n"
            )

        f.write("## 📊 Full Test Matrix Rankings\n\n")
        f.write("| Rank | Model | `-ngl` | `--n-cpu-moe` | `mmap` | Load Time | Prompt t/s | Gen t/s | Status |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for i, r in enumerate(valid_results, start=1):
            badge = "🥇 **1**" if i == 1 else ("🥈 **2**" if i == 2 else ("🥉 **3**" if i == 3 else f"**{i}**"))
            load_s = f"{r['load_time_ms']/1000.0:.2f}s" if r.get("load_time_ms") else "-"
            p_tps = f"{r['prompt_tps']:.1f}" if r.get("prompt_tps") else "-"
            g_tps = f"**{r['gen_tps']:.2f}**" if r.get("gen_tps") else "-"
            mmap_str = "mmap" if r.get("mmap") else "`--no-mmap`"
            f.write(
                f"| {badge} | {r.get('model_key')} | {r['ngl']} | {r['n_cpu_moe']} | {mmap_str} | "
                f"{load_s} | {p_tps} | {g_tps} | ✅ OK |\n"
            )

        failed = [r for r in results if r.get("status") != "OK"]
        if failed:
            f.write("\n### ⚠️ Failed Configurations\n\n")
            for r in failed:
                f.write(
                    f"- `{r.get('model_key')}` with `-ngl {r['ngl']} --n-cpu-moe {r['n_cpu_moe']}` "
                    f"({'mmap' if r['mmap'] else '--no-mmap'}): `{r.get('status')}` ({r.get('error_snippet', '')[:120]})\n"
                )

    print(f"📄 Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Llama.cpp Parameter Benchmark Suite for IQ3 / IQ4 MoE Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ornith-iq3",
        choices=["ornith-iq3", "ornith-iq4", "qwen-iq3", "qwen-iq4", "iq3", "iq4", "all"],
        help="Model or category to test",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help="Path to folder containing llama-cli.exe and model folder",
    )
    parser.add_argument(
        "--ngl",
        type=str,
        default=None,
        help="Comma-separated -ngl values to test (e.g. '28,30,32,34')",
    )
    parser.add_argument(
        "--moe",
        type=str,
        default=None,
        help="Comma-separated --n-cpu-moe values to test (e.g. '22,24,26,27')",
    )
    parser.add_argument(
        "--mmap-mode",
        type=str,
        default="mmap",
        choices=["mmap", "no-mmap", "both"],
        help="Test mmap (default), --no-mmap, or both",
    )
    parser.add_argument(
        "--tokens",
        "-n",
        type=int,
        default=48,
        help="Number of tokens to generate per benchmark run",
    )
    parser.add_argument(
        "--context",
        "-c",
        type=int,
        default=16384,
        help="Context size for benchmark test",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=6,
        help="Thread count (-t and -tb)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="benchmark_results.json",
        help="Path to output JSON result file",
    )
    parser.add_argument(
        "--report",
        "-r",
        type=str,
        default="benchmark_report.md",
        help="Path to output Markdown report file",
    )

    args = parser.parse_args()

    exe_path = os.path.join(args.base_dir, "llama-cli.exe")
    if not os.path.exists(exe_path):
        print(f"❌ Error: llama-cli.exe not found at '{exe_path}'")
        sys.exit(1)

    models_to_test = []
    if args.model == "all":
        models_to_test = list(MODEL_CATALOG.keys())
    elif args.model == "iq3":
        models_to_test = ["ornith-iq3", "qwen-iq3"]
    elif args.model == "iq4":
        models_to_test = ["ornith-iq4", "qwen-iq4"]
    else:
        models_to_test = [args.model]

    all_results = []

    print("=" * 70)
    print("⚡ LLAMA.CPP PARAMETER OPTIMIZATION SUITE")
    print("=" * 70)
    print(f"Executable : {exe_path}")
    print(f"Context    : {args.context} | Gen Tokens: {args.tokens} | Threads: {args.threads}")
    print(f"Models     : {', '.join(models_to_test)}")
    print("=" * 70 + "\n")

    for model_key in models_to_test:
        config = MODEL_CATALOG[model_key]
        model_full_path = os.path.join(args.base_dir, config["path"])

        if not os.path.exists(model_full_path):
            print(f"⚠️ Warning: Model file for {model_key} not found at '{model_full_path}'. Skipping.")
            continue

        ngl_list = (
            [int(x.strip()) for x in args.ngl.split(",") if x.strip()]
            if args.ngl
            else config["default_ngl"]
        )
        moe_list = (
            [int(x.strip()) for x in args.moe.split(",") if x.strip()]
            if args.moe
            else config["default_moe"]
        )

        if args.mmap_mode == "mmap":
            mmap_list = [True]
        elif args.mmap_mode == "no-mmap":
            mmap_list = [False]
        else:
            mmap_list = [True, False]

        grid = list(itertools.product(ngl_list, moe_list, mmap_list))
        total_runs = len(grid)

        print(f"\n🔎 Testing [{config['name']}] - {total_runs} combinations...")

        for idx, (ngl, moe, mmap_flag) in enumerate(grid, start=1):
            mmap_desc = "mmap" if mmap_flag else "no-mmap"
            print(
                f"\n▶️  [{idx}/{total_runs}] Model: {model_key} | -ngl {ngl} | --n-cpu-moe {moe} | {mmap_desc}"
            )

            res = run_single_benchmark(
                exe_path=exe_path,
                model_full_path=model_full_path,
                ngl=ngl,
                n_cpu_moe=moe,
                mmap=mmap_flag,
                context_size=args.context,
                threads=args.threads,
                gen_tokens=args.tokens,
            )

            res["model_key"] = model_key
            res["model_name"] = config["name"]
            res["ngl"] = ngl
            res["n_cpu_moe"] = moe
            res["mmap"] = mmap_flag

            all_results.append(res)

            if res.get("status") == "OK" and res.get("gen_tps"):
                print(
                    f"   ✅ Prompt: {res['prompt_tps']:.1f} t/s | Gen: {res['gen_tps']:.2f} t/s "
                    f"| Load: {res['load_time_ms']/1000.0:.2f}s"
                )
            else:
                print(f"   ❌ {res.get('status')} ({res.get('error_snippet', '')[:80]})")

    if all_results:
        print_table(all_results)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"💾 Raw JSON results saved to: {args.output}")

        generate_markdown_report(all_results, args.report)


if __name__ == "__main__":
    main()
