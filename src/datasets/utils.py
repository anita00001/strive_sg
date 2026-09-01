import torch
from torch_geometric.data import Data

class MeanStdNormalizer:
    """
    Normalize tensor features using fixed mean and standard deviation.

    Normalization:
        (x - mean) / std

    Unnormalization:
        x * std + mean
    """

    def __init__(
        self,
        mean_vals: torch.Tensor,
        std_vals: torch.Tensor,
    ) -> None:

        mean_vals = torch.as_tensor(
            mean_vals,
            dtype=torch.float32,
        )

        std_vals = torch.as_tensor(
            std_vals,
            dtype=torch.float32,
        )

        if mean_vals.ndim != 1:
            raise ValueError(
                "mean_vals must be one-dimensional"
            )

        if std_vals.ndim != 1:
            raise ValueError(
                "std_vals must be one-dimensional"
            )

        if mean_vals.shape != std_vals.shape:
            raise ValueError(
                "mean_vals and std_vals must have the same shape"
            )

        if torch.any(std_vals <= 0):
            raise ValueError(
                "all standard deviations must be positive"
            )

        self.mean_vals = mean_vals
        self.std_vals = std_vals

    @property
    def dim(self) -> int:
        return self.mean_vals.numel()

    def _stats_for(
        self,
        state_data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        feature_dim = state_data.shape[-1]

        if feature_dim > self.dim:
            raise ValueError(
                "state feature dimension exceeds normalizer dimension"
            )

        mean = self.mean_vals[:feature_dim].to(
            device=state_data.device,
            dtype=state_data.dtype,
        )

        std = self.std_vals[:feature_dim].to(
            device=state_data.device,
            dtype=state_data.dtype,
        )

        return mean, std

    def normalize(
        self,
        state_data: torch.Tensor,
    ) -> torch.Tensor:

        mean, std = self._stats_for(state_data)

        return (
            state_data - mean
        ) / std

    def unnormalize(
        self,
        state_data: torch.Tensor,
    ) -> torch.Tensor:

        mean, std = self._stats_for(state_data)

        return (
            state_data * std
        ) + mean

    def normalize_single(
        self,
        state_data: torch.Tensor,
        state_idx: int,
    ) -> torch.Tensor:

        mean = self.mean_vals[state_idx].to(
            state_data.device
        )

        std = self.std_vals[state_idx].to(
            state_data.device
        )

        return (
            state_data - mean
        ) / std

    def unnormalize_single(
        self,
        state_data: torch.Tensor,
        state_idx: int,
    ) -> torch.Tensor:

        mean = self.mean_vals[state_idx].to(
            state_data.device
        )

        std = self.std_vals[state_idx].to(
            state_data.device
        )

        return (
            state_data * std
        ) + mean

#Construct fully connected scene graph

def build_fully_connected_edge_index(
    num_nodes: int,
) -> torch.Tensor:
    """
    Build a directed fully connected graph without self-loops.

    For N nodes, the number of directed edges is:

        N * (N - 1)
    """
    if num_nodes < 0:
        raise ValueError(
            "num_nodes cannot be negative"
        )

    if num_nodes <= 1:
        return torch.empty(
            (2, 0),
            dtype=torch.long,
        )

    nodes = torch.arange(
        num_nodes,
        dtype=torch.long,
    )

    source = nodes.repeat_interleave(
        num_nodes
    )

    target = nodes.repeat(
        num_nodes
    )

    keep = source != target

    return torch.stack(
        [
            source[keep],
            target[keep],
        ],
        dim=0,
    ).contiguous()
