"""Summarize results/train_*.jsonl: a markdown table and a loss-curve PNG."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"fp32": "#555555", "bf16": "#1f77b4", "bf16+tbf16": "#d62728", "fp16-amp": "#2ca02c",
          "bf16-pure": "#9467bd", "bf16-pure+tbf16": "#ff7f0e"}


def main(path, png):
    runs = [json.loads(l) for l in open(path)]
    base = next((r for r in runs if r["recipe"] == "bf16"), None)
    print("| recipe | step (ms) | tokens/s | vs bf16 | peak GB | final train loss | eval loss | skipped steps |")
    print("|---|---|---|---|---|---|---|---|")
    for r in runs:
        sp = f"{base['median_step_ms'] / r['median_step_ms']:.2f}x" if base else ""
        print(f"| {r['recipe']} | {r['median_step_ms']:.0f} | {r['tokens_per_s']:.0f} | {sp} | {r['peak_gb']:.2f} | "
              f"{r['final_train_loss']:.4f} | {r['eval_loss']:.4f} | {r['skipped_steps']} |")
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    for r in runs:
        c = r["loss_curve"]
        sm = [sum(c[max(0, i - 9):i + 1]) / len(c[max(0, i - 9):i + 1]) for i in range(len(c))]
        ax.plot(sm, label=f"{r['recipe']} ({r['median_step_ms']:.0f} ms/step)", color=COLORS.get(r["recipe"]), lw=1.4)
    ax.set_xlabel("step"); ax.set_ylabel("train loss (10-step mean)")
    ax.set_title(f"{runs[0]['model']}, batch {runs[0]['batch']} x {runs[0]['seq']}, {runs[0]['gpu']}", fontsize=9)
    ax.set_ylim(top=min(ax.get_ylim()[1], 3.5)); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(png)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
