import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_loss(loss_history: list, title: str, out_path: Path) -> None:
    """Save a mean-loss-per-epoch curve."""
    fig, ax = plt.subplots()
    ax.plot(loss_history, marker='o', markersize=1)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss')
    ax.grid()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
