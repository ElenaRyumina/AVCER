import numpy as np
import pandas as pd

from sklearn import metrics


def proba_of_positive_class(predicts: list[np.ndarray]) -> np.ndarray:
    """
    Converts binary predictions (2 classes) to single value

    Args:
        predicts (list[np.ndarray]): Predicts array

    Returns:
        np.ndarray: Probability of positive class
    """
    return predicts[:, 1]


def proba_to_class_num(predicts: list[np.ndarray]) -> np.ndarray:
    """Converts probability to class number

    Args:
        predicts (list[np.ndarray]): Predicts array

    Returns:
        np.ndarray: Class number with maximum probability
    """
    return np.argmax(predicts, axis=1)


def conf_matrix(
    targets: list[np.ndarray], predicts: list[np.ndarray], c_names: list[str]
) -> np.ndarray:
    """Wrapper for sklearn confusion_matrix. Converts probability to class number and computes confusion_matrix

    Args:
        targets (list[np.ndarray]): Targets array
        predicts (list[np.ndarray]): Predicts array
        c_names (str, optional): List of classes to index the matrix. Converted to list of integers.

    Returns:
        np.ndarray: Confusion matrix
    """
    predicts = proba_to_class_num(predicts)
    return metrics.confusion_matrix(
        targets, predicts, labels=[l for l in range(len(c_names))]
    )


def recall(
    targets: list[np.ndarray], predicts: list[np.ndarray], average: str = None
) -> float:
    """Wrapper for sklearn recall. Converts probability to class number and computes recall
    Returns UAR, if `average` is `macro`

    Args:
        targets (list[np.ndarray]): Targets array
        predicts (list[np.ndarray]): Predicts array
        average (str, optional): This parameter is required for multiclass/multilabel targets. Defaults to None.

    Returns:
        float: Recall value
    """
    predicts = proba_to_class_num(predicts)
    return metrics.recall_score(
        targets, predicts, average=average, zero_division=np.nan
    )


def precision(
    targets: list[np.ndarray], predicts: list[np.ndarray], average: str = None
) -> float:
    """Wrapper for sklearn precision. Converts probability to class number and computes precision

    Args:
        targets (list[np.ndarray]): Targets array
        predicts (list[np.ndarray]): Predicts array
        average (str, optional): This parameter is required for multiclass/multilabel targets. Defaults to None.

    Returns:
        float: Precision value
    """
    predicts = proba_to_class_num(predicts)
    return metrics.precision_score(
        targets, predicts, average=average, zero_division=np.nan
    )


def f1(
    targets: list[np.ndarray], predicts: list[np.ndarray], average: str = None
) -> float:
    """Wrapper for sklearn f1. Converts probability to class number and computes f1

    Args:
        targets (list[np.ndarray]): Targets array
        predicts (list[np.ndarray]): Predicts array
        average (str, optional): This parameter is required for multiclass/multilabel targets. Defaults to None.

    Returns:
        float: F1 value
    """
    predicts = proba_to_class_num(predicts)
    return metrics.f1_score(targets, predicts, average=average, zero_division=np.nan)


def accuracy(
    targets: list[np.ndarray], predicts: list[np.ndarray], average: str = None
) -> float:
    """Wrapper for sklearn accuracy. Converts probability to class number and computes accuracy

    Args:
        targets (list[np.ndarray]): Targets array
        predicts (list[np.ndarray]): Predicts array
        average (str, optional): Not used here. Defaults to None.

    Returns:
        float: Accuracy value
    """
    predicts = proba_to_class_num(predicts)
    return metrics.accuracy_score(targets, predicts)


def ccc_score(
    targets: list[np.ndarray], predicts: list[np.ndarray], average: str = None
) -> float:
    """Computes Concordance correlation coefficient
    https://en.wikipedia.org/wiki/Concordance_correlation_coefficient

    Args:
        targets (list[np.ndarray]): Targets array
        predicts (list[np.ndarray]): Predicts array
        average (str, optional): Not used here. Defaults to None.

    Returns:
        float: ccc_score
    """

    mean_targets = np.mean(targets)
    mean_predicts = np.mean(predicts)
    vy = targets - mean_targets
    vx = predicts - mean_predicts
    cor = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))

    sd_targets = np.std(targets)
    sd_predicts = np.std(predicts)

    numerator = 2 * cor * sd_targets * sd_predicts

    denominator = sd_targets**2 + sd_predicts**2 + (mean_targets - mean_predicts) ** 2

    return numerator / denominator


def v_score(
    targets: list[np.ndarray] | np.ndarray,
    predicts: list[np.ndarray] | np.ndarray,
    average: str = None,
) -> float:
    """Computes CCC score for valence. It is first item ([:, :, 0]) in predicts/targets

    Args:
        targets (list[np.ndarray] | np.ndarray): Targets array
        predicts (list[np.ndarray] | np.ndarray): Predicts array
        average (str, optional): Not used here. Defaults to None.

    Returns:
        float: CCC score for valence
    """
    if isinstance(targets, list):
        targets = np.stack(targets)

    if isinstance(predicts, list):
        predicts = np.stack(predicts)

    targets_v = targets[:, :, 0].flatten().tolist()
    predicts_v = predicts[:, :, 0].flatten().tolist()

    return ccc_score(targets_v, predicts_v)


def a_score(
    targets: list[np.ndarray] | np.ndarray,
    predicts: list[np.ndarray] | np.ndarray,
    average: str = None,
) -> float:
    """Computes CCC score for arousal. It is second item ([:, :, 1]) in predicts/targets

    Args:
        targets (list[np.ndarray] | np.ndarray): Targets array
        predicts (list[np.ndarray] | np.ndarray): Predicts array
        average (str, optional): Not used here. Defaults to None.

    Returns:
        float: CCC score for valence
    """
    if isinstance(targets, list):
        targets = np.stack(targets)

    if isinstance(predicts, list):
        predicts = np.stack(predicts)

    targets_a = targets[:, :, 1].flatten().tolist()
    predicts_a = predicts[:, :, 1].flatten().tolist()

    return ccc_score(targets_a, predicts_a)


def va_score(
    targets: list[np.ndarray], predicts: list[np.ndarray], average: str = None
) -> float:
    """Computes valence/arousal CCC score.
    It computes average of valence CCC score and arousal CCC score

    Args:
        targets (list[np.ndarray]): Targets array
        predicts (list[np.ndarray]): Predicts array
        average (str, optional): Not used here. Defaults to None.

    Returns:
        float: CCC score for valence
    """
    return 0.5 * (v_score(targets, predicts) + a_score(targets, predicts))
