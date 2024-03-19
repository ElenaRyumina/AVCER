from itertools import product
from typing import Union
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_conf_matrix(
    cm: np.ndarray,
    labels: list[str] = None,
    title: str = "Confusion Matrix",
    save_path: str = None,
    colorbar: bool = True,
    figsize: tuple = (8, 6),
    color_map: LinearSegmentedColormap = plt.cm.Blues,
    xticks_rotation: Union[float, str] = "horizontal",
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


def plot_weights_matrix(
    cm: np.ndarray,
    labels: list[str] = None,
    title: str = "Confusion Matrix",
    save_path: str = None,
    colorbar: bool = True,
    figsize: tuple = (8, 6),
    color_map: LinearSegmentedColormap = plt.cm.Blues,
    xticks_rotation: Union[float, str] = "horizontal",
    x_labels: bool = True,
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
        x_labels (bool, optional): Drawes xticks. Defaults to True.
    """

    fig, ax = plt.subplots(figsize=figsize)
    n_models = cm.shape[0]
    n_classes = cm.shape[1]
    im = ax.imshow(cm, interpolation="nearest", cmap=color_map)

    im.set_clim(0, 1)
    color_map_min, color_map_max = im.cmap(0), im.cmap(256)

    # Loop over data dimensions and create text annotations.
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in product(range(n_models), range(n_classes)):
        color = color_map_max if cm[i, j] < thresh else color_map_min
        ax.text(j, i, "{0:.2f}".format(cm[i, j]), ha="center", va="center", color=color)

    if colorbar:
        fig.colorbar(im, ax=ax, shrink=0.25)

    for i in range(n_models):
        for j in range(n_classes):
            ax.text(
                j,
                i,
                "{0:.2f}".format(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    if x_labels:
        ax.set(
            yticks=np.arange(n_models),
            xticks=np.arange(n_classes),
            title=title,
            yticklabels=labels[0],
            xticklabels=labels[1],
            ylabel="Model",
            xlabel="Weights of emotions and models",
        )
    else:
        ax.set(
            yticks=np.arange(n_models),
            title=title,
            yticklabels=labels[0],
            ylabel="Model",
        )

        ax.set_xticks([])

    ax.set_ylim((n_models - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

    fig.tight_layout()
    if save_path:
        plt.savefig(
            save_path,
            format=save_path.rsplit(".", 1)[1],
            bbox_inches="tight",
            pad_inches=0,
        )

    plt.show(block=False)


def plot_compound_expression_prediction(
    dict_preds: dict[str, list[float]],
    save_path: str = None,
    colors: list[str] = ["green", "orange", "red", "purple", "blue"],
    figsize: tuple = (12, 2),
    title: str = "Confusion Matrix",
) -> None:

    plt.figure(figsize=figsize)

    for idx, (k, v) in enumerate(dict_preds.items()):

        plt.plot(
            range(1, len(v) + 1), v, color=colors[idx], linestyle="dotted", label=k
        )

    plt.legend()
    plt.xlabel("Number of frames")
    plt.ylabel("Compound expression")
    plt.title(title)

    plt.yticks(
        [0, 1, 2, 3, 4, 5, 6],
        [
            "Fearfully Surprised",
            "Happily Surprised",
            "Sadly Surprised",
            "Disgustedly Surprised",
            "Angrily Surprised",
            "Sadly Fearful",
            "Sadly Angry",
        ],
    )

    if save_path:
        plt.savefig(
            save_path,
            format=save_path.rsplit(".", 1)[1],
            bbox_inches="tight",
            pad_inches=0,
        )


def show_cam_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    use_rgb: bool = False,
    colormap: int = cv2.COLORMAP_JET,
    image_weight: float = 0.5,
) -> np.ndarray:
    """This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.

    Implemented by https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/utils/image.py
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}"
        )

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
