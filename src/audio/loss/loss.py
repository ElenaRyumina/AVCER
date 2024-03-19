import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    Implemented by Denis Dresvyanskiy.

    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.

    Args:
        alpha (torch.Tensor, optional): Weights for each class. Defaults to None.
        gamma (float, optional): A constant, as described in the paper. Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'. Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore. Defaults to -100.

    Raises:
        ValueError: Supported reduction types: "mean", "sum", "none"
    """

    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction="none", ignore_index=ignore_index
        )

    def __repr__(self) -> str:
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes Focal Loss

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: Focal Loss value
        """
        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class SoftFocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002 with soft targets,
    For example, target can be [0, 0.3, 0.7, 1]
    Class FocalLoss takes only digit targets.
    Implemented by Denis Dresvyanskiy.

    Args:
        softmax (bool): Apply softmax or not. Defaults to True.
        alpha (torch.Tensor, optional): Weights for each class. Defaults to None.
        gamma (float, optional): A constant, as described in the paper. Defaults to 0.
    """

    def __init__(
        self, softmax: bool = True, alpha: torch.Tensor = None, gamma: float = 0.0
    ) -> None:
        super().__init__()
        self.alpha = 1 if alpha is None else alpha
        self.gamma = gamma
        self.softmax = softmax

    def __repr__(self) -> str:
        arg_keys = ["alpha", "gamma"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes Focal Loss for soft targets

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: Focal Loss value
        """
        if self.softmax:
            p = F.softmax(x, dim=-1)
        else:
            p = x

        epsilon = 1e-7
        p = torch.clip(p, epsilon, 1.0 - epsilon)
        cross_entropy = -y * torch.log(p)

        # focal loss
        loss = self.alpha * torch.pow(1.0 - p, self.gamma) * cross_entropy
        loss = torch.sum(loss, dim=-1).mean()
        return loss


class SoftFocalLossWrapper(nn.Module):
    """Wrapper for FocalLoss class
    Performs one-hot encoding

    Args:
        focal_loss (nn.Module): Focal loss
        num_classes (int): Number of classes
    """

    def __init__(self, focal_loss: nn.Module, num_classes: int) -> None:

        super().__init__()
        self.focal_loss = focal_loss
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes Focal Loss for soft targets

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: Focal Loss value
        """
        new_y = F.one_hot(y, num_classes=self.num_classes)
        return self.focal_loss(x, new_y)


if __name__ == "__main__":
    num_classes = 4

    x = torch.randn(3, num_classes)
    y = torch.randn(3, num_classes)
    y = torch.argmax(y, axis=1)

    sfl = SoftFocalLoss(alpha=None)
    sflw = SoftFocalLossWrapper(focal_loss=sfl, num_classes=num_classes)
    print(sflw(x, y))
