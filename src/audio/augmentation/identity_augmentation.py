import torch


class IdentityAugmentation(torch.nn.Module):
    """Placeholder for augmentation"""

    def __init__(self) -> None:
        super(IdentityAugmentation, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return input data without modification

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Input tensor without changes
        """
        return x
