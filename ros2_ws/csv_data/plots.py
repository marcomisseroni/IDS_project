import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

CSV_DIR = Path(__file__).parent


def plot_by_measurer(df, title_prefix):
    has_msg_id = "msg_id" in df.columns
    if has_msg_id:
        df = df.sort_values("msg_id")

    for id_a, group in df.groupby("id_a"):
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
        fig.suptitle(f"{title_prefix} - misure fatte da limo {id_a}")

        for id_b, sub in group.groupby("id_b"):
            x_vals = sub["msg_id"] if has_msg_id else sub.index
            axes[0].plot(x_vals, sub["x"], marker=".", linestyle="-", label=f"verso {id_b}")
            axes[1].plot(x_vals, sub["y"], marker=".", linestyle="-", label=f"verso {id_b}")
            axes[2].plot(x_vals, sub["dtheta"], marker=".", linestyle="-", label=f"verso {id_b}")

        axes[0].set_ylabel("x")
        axes[1].set_ylabel("y")
        axes[2].set_ylabel("dtheta")
        axes[2].set_xlabel("msg_id" if has_msg_id else "indice campione")
        for ax in axes:
            ax.legend(title="misurato", fontsize="small")
            ax.grid(True)

        fig.tight_layout()


def main():
    data_raw = pd.read_csv(CSV_DIR / "data.csv")
    data_routed = pd.read_csv(CSV_DIR / "data_routed.csv")

    plot_by_measurer(data_raw, "raw")
    plot_by_measurer(data_routed, "routed")
    plt.show()


if __name__ == '__main__':
    main()
