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

def build_scene_graph(
    past: torch.Tensor,
    future: torch.Tensor,
    sem: torch.Tensor,
    lw: torch.Tensor,
    past_vis: torch.Tensor,
    future_vis: torch.Tensor,
    *,
    past_gt: torch.Tensor | None = None,
    future_gt: torch.Tensor | None = None,
) -> Data:
    """
    Create one STRIVE-style PyTorch Geometric scene graph.

    Each agent is one graph node.
    """
    num_agents = past.shape[0]

    node_attributes = {
        "past": past,
        "future": future,
        "sem": sem,
        "lw": lw,
        "past_vis": past_vis,
        "future_vis": future_vis,
    }

    for name, value in node_attributes.items():
        if value.shape[0] != num_agents:
            raise ValueError(
                f"{name} has {value.shape[0]} agents; "
                f"expected {num_agents}"
            )

    edge_index = build_fully_connected_edge_index(
        num_agents
    )

    graph = Data(
        edge_index=edge_index,
        past=past,
        future=future,
        sem=sem,
        lw=lw,
        past_vis=past_vis,
        future_vis=future_vis,
        num_nodes=num_agents,
    )

    if past_gt is not None:
        graph.past_gt = past_gt

    if future_gt is not None:
        graph.future_gt = future_gt

    return graph

#First agent - Ego-index
def get_ego_inds(scene_graph) -> torch.Tensor:
    """
    Return a boolean mask selecting the first agent of each graph
    inside a PyTorch Geometric batch.

    Example
    -------
    batch = [0, 0, 0, 1, 1, 1, 1]

    returns:
        [True, False, False, True, False, False, False]
    """
    if not hasattr(scene_graph, "batch"):
        raise ValueError(
            "scene_graph must be a batched PyG graph with a batch vector"
        )

    batch = scene_graph.batch

    if batch.numel() == 0:
        return torch.empty(
            0,
            dtype=torch.bool,
            device=batch.device,
        )

    ego_mask = torch.zeros_like(
        batch,
        dtype=torch.bool,
    )

    ego_mask[0] = True

    ego_mask[1:] = (
        batch[1:] != batch[:-1]
    )

    return ego_mask

def normalize_scene_graph(
    scene_graph,
    state_normalizer: MeanStdNormalizer,
    att_normalizer: MeanStdNormalizer,
    *,
    unnorm: bool = False,
):
    """
    Normalize or unnormalize STRIVE graph attributes in-place.

    State normalization applies to:
        past
        past_gt
        future
        future_gt
        pos

    Attribute normalization applies to:
        lw
    """
    state_fn = (
        state_normalizer.unnormalize
        if unnorm
        else state_normalizer.normalize
    )

    att_fn = (
        att_normalizer.unnormalize
        if unnorm
        else att_normalizer.normalize
    )

    state_attributes = (
        "past",
        "past_gt",
        "future",
        "future_gt",
        "pos",
    )

    for name in state_attributes:
        if hasattr(scene_graph, name):
            value = getattr(
                scene_graph,
                name,
            )

            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 1
            ):
                setattr(
                    scene_graph,
                    name,
                    state_fn(value),
                )

    if hasattr(scene_graph, "lw"):
        value = scene_graph.lw

        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 1
        ):
            scene_graph.lw = (
                att_fn(value)
            )

    return scene_graph

