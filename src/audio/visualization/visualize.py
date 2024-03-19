from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


def plot_conf_matrix(
    cm: np.ndarray,
    labels: list[str] = None,
    title: str = "Confusion Matrix",
    save_path: str = None,
    colorbar: bool = True,
    figsize: tuple = (8, 6),
    color_map: LinearSegmentedColormap = plt.cm.Blues,
    xticks_rotation: float | str = "horizontal",
) -> None:
    """Plot confusion matrix

    Args:
        cm (np.ndarray): Confusion matrix values
        labels (list[str], optional): List of labels (classes). Defaults to None.
        title (str, optional): Title of confusion matrix. Defaults to 'Confusion Matrix'.
        save_path (str, optional): The path where the drawn matrix will be saved, if specified. Defaults to None.
        colorbar (bool, optional): Drawes colorbar on the right side of matrix. Defaults to True.
        figsize (tuple, optional): Figure size of confusion matrix. Defaults to (8, 6).
        color_map (LinearSegmentedColormap, optional): Color map. Defaults to plt.cm.Blues.
        xticks_rotation (float | str, optional): Text rotation of X axe, could be float, 'horizontal' or 'vertical'. Defaults to 'horizontal'.
    """
    float_cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10) * 100

    fig, ax = plt.subplots(figsize=figsize)
    n_classes = cm.shape[0]
    im = ax.imshow(float_cm, interpolation="nearest", cmap=color_map)
    im.set_clim(0, 100)
    color_map_min, color_map_max = im.cmap(0), im.cmap(256)

    # Loop over data dimensions and create text annotations.
    thresh = (float_cm.max() + float_cm.min()) / 2.0
    for i, j in product(range(n_classes), range(n_classes)):
        color = color_map_max if float_cm[i, j] < thresh else color_map_min
        ax.text(
            j,
            i,
            "{0}\n{1:.1f}%".format(cm[i, j], float_cm[i, j]),
            ha="center",
            va="center",
            color=color,
        )

    if labels is None:
        labels = np.arange(n_classes)

    if colorbar:
        fig.colorbar(im, ax=ax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                "{0}\n{1:.1f}%".format(cm[i, j], float_cm[i, j]),
                ha="center",
                va="center",
                color="white" if float_cm[i, j] > thresh else "black",
            )

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        title=title,
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
    )

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, format=save_path.rsplit(".", 1)[1])

    # plt.show(block=False)
