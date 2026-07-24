"""
Plot cumulative hypercube volume (average coordinate diameter) across tasks.

Usage:
  # From saved JSON files (one per seed):
  python plot_volume.py --json_files path/to/volume_history_seed0.json path/to/volume_history_seed1.json

  # From training log files (grep-based fallback):
  python plot_volume.py --log_files path/to/output.log

  # Specify output paths:
  python plot_volume.py --json_files ... --plot_out volume_plot.pdf --table_out volume_table.tex

  # Group size for the table (default 5, matching Table 8 style):
  python plot_volume.py --json_files ... --group_size 5
"""

import argparse
import json
import re
import numpy as np
import os


def load_volumes_from_json(json_files):
    all_volumes = []
    for f in json_files:
        with open(f) as fh:
            data = json.load(fh)
        layer_key = list(data.keys())[0]
        all_volumes.append(data[layer_key]["volumes"])
    return all_volumes


def load_volumes_from_log(log_files):
    pattern = re.compile(r"Volume of the cumulative hypercube for \d+-th layer.*?:\s*([\d.]+)")
    final_pattern = re.compile(r"Final volume of cumulative hypercube for \d+-th layer:\s*([\d.]+)")
    trial_pattern = re.compile(r"STARTING TRIAL")

    all_volumes = []
    for f in log_files:
        with open(f) as fh:
            text = fh.read()

        trials = re.split(r"\*\s*STARTING TRIAL\s+\d+\s*\*", text)
        for trial_text in trials:
            volumes = []
            for m in pattern.finditer(trial_text):
                volumes.append(float(m.group(1)))
            for m in final_pattern.finditer(trial_text):
                volumes.append(float(m.group(1)))
            if volumes:
                all_volumes.append(volumes)

    return all_volumes


def print_table(volumes_mean, volumes_std, group_size=5):
    n = len(volumes_mean)
    print(f"\n{'Task Interval':<20} {'1/d * Σ diam_j(H)'}")
    print("-" * 50)
    for start in range(0, n, group_size):
        end = min(start + group_size, n) - 1
        v_start = volumes_mean[start]
        v_end = volumes_mean[end]
        if volumes_std is not None:
            s_start = volumes_std[start]
            s_end = volumes_std[end]
            print(f"Tasks {start+1:>2}–{end+1:<2}          "
                  f"{v_start:.3f}±{s_start:.3f} → {v_end:.3f}±{s_end:.3f}")
        else:
            print(f"Tasks {start+1:>2}–{end+1:<2}          "
                  f"{v_start:.3f} → {v_end:.3f}")


def generate_latex_table(volumes_mean, volumes_std, group_size=5):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Growth of the cumulative hypercube's average coordinate diameter.}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"Task Interval & $\frac{1}{d}\sum \mathrm{diam}_j(\mathcal{H})$ \\")
    lines.append(r"\midrule")

    n = len(volumes_mean)
    for start in range(0, n, group_size):
        end = min(start + group_size, n) - 1
        v_start = volumes_mean[start]
        v_end = volumes_mean[end]
        if volumes_std is not None:
            s_start = volumes_std[start]
            s_end = volumes_std[end]
            lines.append(
                f"Tasks {start+1}--{end+1} & "
                f"${v_start:.3f}\\pm{s_start:.3f} \\to {v_end:.3f}\\pm{s_end:.3f}$ \\\\"
            )
        else:
            lines.append(
                f"Tasks {start+1}--{end+1} & "
                f"${v_start:.3f} \\to {v_end:.3f}$ \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def plot_volumes(volumes_mean, volumes_std, plot_out, dataset_name=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return

    n = len(volumes_mean)
    tasks = np.arange(1, n + 1)

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ax.plot(tasks, volumes_mean, 'k-o', markersize=4, linewidth=1.5, label="Avg. coordinate diameter")
    if volumes_std is not None:
        ax.fill_between(tasks,
                        np.array(volumes_mean) - np.array(volumes_std),
                        np.array(volumes_mean) + np.array(volumes_std),
                        alpha=0.2, color='gray')

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel(r"$\frac{1}{d}\sum \mathrm{diam}_j(\mathcal{H})$", fontsize=13)
    title = "Cumulative Hypercube Growth"
    if dataset_name:
        title += f" — {dataset_name}"
    ax.set_title(title, fontsize=13)
    ax.set_xticks(tasks[::max(1, n // 10)])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_out, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot/table of cumulative hypercube volume over tasks")
    parser.add_argument("--json_files", nargs="+", help="volume_history JSON files (one per seed)")
    parser.add_argument("--log_files", nargs="+", help="training output.log files (fallback)")
    parser.add_argument("--plot_out", default="volume_plot.pdf", help="output plot path")
    parser.add_argument("--table_out", default=None, help="output LaTeX table path")
    parser.add_argument("--group_size", type=int, default=5, help="task group size for table")
    parser.add_argument("--dataset", default="", help="dataset name for plot title")
    args = parser.parse_args()

    if args.json_files:
        all_volumes = load_volumes_from_json(args.json_files)
    elif args.log_files:
        all_volumes = load_volumes_from_log(args.log_files)
    else:
        parser.error("Provide either --json_files or --log_files")

    min_len = min(len(v) for v in all_volumes)
    trimmed = [v[:min_len] for v in all_volumes]
    arr = np.array(trimmed)

    volumes_mean = arr.mean(axis=0).tolist()
    volumes_std = arr.std(axis=0).tolist() if len(trimmed) > 1 else None

    print_table(volumes_mean, volumes_std, group_size=args.group_size)

    latex = generate_latex_table(volumes_mean, volumes_std, group_size=args.group_size)
    print(f"\n{latex}")
    if args.table_out:
        with open(args.table_out, 'w') as f:
            f.write(latex)
        print(f"\nLaTeX table saved to {args.table_out}")

    plot_volumes(volumes_mean, volumes_std, args.plot_out, dataset_name=args.dataset)


if __name__ == "__main__":
    main()
