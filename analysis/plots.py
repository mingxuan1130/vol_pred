import pandas as pd
import matplotlib.pyplot as plt


def plot_pred_vs_true_time(
    ts,
    model_dict,
    *,
    subsample="5min",
    tick_freq="30min",
    figsize=(16, 5),
):
    """
    画预测 vs 真实值 
    """

    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)

    n_models = len(model_dict)

    fig, axes = plt.subplots(
        n_models,
        1,
        figsize=(figsize[0], figsize[1] * n_models),
        sharex=True
    )

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, d) in zip(axes, model_dict.items()):

        df = pd.DataFrame({
            "ts": ts,
            "y_pred": d["y_pred"],
            "y_true": d["y_true"]
        }).dropna()

        # ---- 时间降采样 ----
        df["grid"] = df["ts"].dt.floor(subsample)
        df = df.groupby("grid").first().reset_index()

        ax.plot(df["grid"], df["y_true"], label="true", linewidth=2)
        ax.plot(df["grid"], df["y_pred"], label="pred", alpha=0.8)

        ax.set_title(model_name)
        ax.grid(alpha=0.3)
        ax.legend()

    # ---- x轴时间刻度 ----
    tick_times = pd.date_range(
        start=df["grid"].min(),
        end=df["grid"].max(),
        freq=tick_freq
    )

    axes[-1].set_xticks(tick_times)
    axes[-1].set_xticklabels(
        [t.strftime("%H:%M") for t in tick_times],
        rotation=45
    )

    axes[-1].set_xlabel("time")

    plt.tight_layout()
    plt.show()

    return fig, axes



def plot_pred_true_scatter(
    model_dict,
    *,
    subsample=None,
    figsize=(6, 6),
):
    """
    画 y_pred vs y_true 散点图
    """

    n_models = len(model_dict)

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(figsize[0] * n_models, figsize[1])
    )

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, d) in zip(axes, model_dict.items()):

        y_pred = pd.Series(d["y_pred"]).reset_index(drop=True)
        y_true = pd.Series(d["y_true"]).reset_index(drop=True)

        df = pd.DataFrame({
            "y_pred": y_pred,
            "y_true": y_true
        }).dropna()

        if subsample is not None:
            df = df.iloc[::subsample]

        ax.scatter(
            df["y_pred"],
            df["y_true"],
            alpha=0.3,
            s=5
        )

        # ---- 45° reference line ----
        min_val = min(df["y_pred"].min(), df["y_true"].min())
        max_val = max(df["y_pred"].max(), df["y_true"].max())

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--"
        )

        ax.set_xlabel("y_hat")
        ax.set_ylabel("y_true")
        ax.set_title(model_name)

        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig, axes