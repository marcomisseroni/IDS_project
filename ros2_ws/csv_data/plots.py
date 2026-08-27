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


AGENT_CSV_SUFFIXES = {"limo0": "0", "limo1": "1", "limo2": "2", "person": "person"}

# fixed set of cross-covariance blocks every agent's pred_states_*.csv logs as
# separate cc_i_j columns (see EKF_node.py's _CROSS_COV_PAIRS): limo0, limo1,
# limo2, person(=3), all combinations i<j
CROSS_COV_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def plot_trace_P(skip_seconds=15):
    # trace(P) logged by EKF_node.py in pred_states_{0,1,2,person}.csv, one
    # line per robot/person. skip_seconds discards the initial transient (P
    # still climbing from its startup value, before the first measurement) so
    # the steady-state behaviour isn't squashed by the y-axis autoscale.
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, suffix in AGENT_CSV_SUFFIXES.items():
        path = CSV_DIR / f"pred_states_{suffix}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        t = df["timestamp"] - df["timestamp"].iloc[0]
        mask = t >= skip_seconds
        ax.plot(t[mask], df["trace_P"][mask], label=name)

    ax.set_ylabel("Tr(P)")
    ax.set_xlabel("time [s]")
    ax.grid(True)
    ax.legend(fontsize="small")

    fig.suptitle("Covariance trace")
    fig.tight_layout()
    fig.savefig(CSV_DIR / "trace_P_plot.png")


def plot_cross_cov_elements(agent="limo0", skip_seconds=15):
    # every cross-covariance block Pi_ij (not summed into a single norm), all
    # taken from one agent's local copy of Pi - the copies kept by the other
    # agents are similar, so any one of them is representative.
    suffix = AGENT_CSV_SUFFIXES[agent]
    path = CSV_DIR / f"pred_states_{suffix}.csv"
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(10, 5))

    t = df["timestamp"] - df["timestamp"].iloc[0]
    mask = t >= skip_seconds
    for i, j in CROSS_COV_PAIRS:
        col = f"cc_{i}_{j}"
        if col not in df.columns:
            continue
        ax.plot(t[mask], df[col][mask], label=f"{i}-{j}")

    ax.set_ylabel("||Pi_ij||")
    ax.set_xlabel("time [s]")
    ax.grid(True)
    ax.legend(title="Agent pair", fontsize="small")

    fig.suptitle(f"Cross-covariance norm (from {agent})")
    fig.tight_layout()
    fig.savefig(CSV_DIR / "cross_cov_plot.png")


def main():
    data_raw = pd.read_csv(CSV_DIR / "data.csv")
    data_routed = pd.read_csv(CSV_DIR / "data_routed.csv")

    plot_by_measurer(data_raw, "raw")
    plot_by_measurer(data_routed, "routed")
    plot_trace_P()
    plot_cross_cov_elements()
    plt.show()


if __name__ == '__main__':
    main()
