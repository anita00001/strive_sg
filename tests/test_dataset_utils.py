import torch

from src.datasets.utils import MeanStdNormalizer
from src.datasets.utils import (
    build_fully_connected_edge_index,
    build_scene_graph,
)
from torch_geometric.data import Batch

from src.datasets.utils import (get_ego_inds)

from src.datasets.utils import (
    normalize_scene_graph,
)

def test_normalizer():
    normalizer = MeanStdNormalizer(
        mean_vals=torch.tensor(
            [10.0, 20.0]
        ),
        std_vals=torch.tensor(
            [2.0, 5.0]
        ),
    )

    values = torch.tensor(
        [
            [12.0, 25.0],
            [8.0, 15.0],
        ]
    )

    result = normalizer.normalize(values)

    expected = torch.tensor(
        [
            [1.0, 1.0],
            [-1.0, -1.0],
        ]
    )

    assert torch.allclose(
        result,
        expected,
    )


def test_normalizer_round_trip():
    normalizer = MeanStdNormalizer(
        mean_vals=torch.tensor(
            [1.0, 2.0, 3.0]
        ),
        std_vals=torch.tensor(
            [2.0, 4.0, 5.0]
        ),
    )

    original = torch.tensor(
        [
            [
                [3.0, 6.0, 8.0],
                [5.0, 10.0, 13.0],
            ]
        ]
    )

    normalized = normalizer.normalize(original)
    reconstructed = normalizer.unnormalize(normalized)

    assert torch.allclose(
        reconstructed,
        original,
        atol=1e-6,
    )


def test_normalizer_supports_partial_features():
    normalizer = MeanStdNormalizer(
        mean_vals=torch.tensor(
            [10.0, 20.0, 30.0]
        ),
        std_vals=torch.tensor(
            [2.0, 5.0, 10.0]
        ),
    )

    values = torch.tensor(
        [
            [12.0, 25.0]
        ]
    )

    result = normalizer.normalize(values)

    expected = torch.tensor(
        [
            [1.0, 1.0]
        ]
    )

    assert torch.allclose(
        result,
        expected,
    )

# test graph
def test_fully_connected_edges_three_nodes():
    edge_index = build_fully_connected_edge_index(3)

    assert edge_index.shape == (2, 6)

    edges = set(
        map(
            tuple,
            edge_index.T.tolist(),
        )
    )

    expected = {
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    }

    assert edges == expected


def test_fully_connected_edges_one_node():
    edge_index = build_fully_connected_edge_index(1)

    assert edge_index.shape == (2, 0)


def test_scene_graph():
    num_agents = 3
    past_len = 4
    future_len = 12
    num_classes = 2

    past = torch.zeros(
        num_agents,
        past_len,
        6,
    )

    future = torch.zeros(
        num_agents,
        future_len,
        6,
    )

    sem = torch.zeros(
        num_agents,
        num_classes,
    )

    lw = torch.zeros(
        num_agents,
        2,
    )

    past_vis = torch.ones(
        num_agents,
        past_len,
    )

    future_vis = torch.ones(
        num_agents,
        future_len,
    )

    graph = build_scene_graph(
        past=past,
        future=future,
        sem=sem,
        lw=lw,
        past_vis=past_vis,
        future_vis=future_vis,
    )

    assert graph.num_nodes == 3
    assert graph.edge_index.shape == (2, 6)

    assert graph.past.shape == (3, 4, 6)
    assert graph.future.shape == (3, 12, 6)
    assert graph.sem.shape == (3, 2)
    assert graph.lw.shape == (3, 2)

def test_get_ego_inds():
    def make_scene(num_agents):
        return build_scene_graph(
            past=torch.zeros(num_agents, 4, 6),
            future=torch.zeros(num_agents, 12, 6),
            sem=torch.zeros(num_agents, 2),
            lw=torch.zeros(num_agents, 2),
            past_vis=torch.ones(num_agents, 4),
            future_vis=torch.ones(num_agents, 12),
        )

    scene_a = make_scene(3)
    scene_b = make_scene(5)

    batch = Batch.from_data_list(
        [scene_a, scene_b]
    )

    mask = get_ego_inds(batch)

    expected = torch.tensor(
        [
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
        ]
    )

    assert torch.equal(
        mask.cpu(),
        expected,
    )

def test_normalize_scene_graph():
    graph = build_scene_graph(
        past=torch.tensor(
            [
                [
                    [12.0, 25.0],
                ]
            ]
        ),
        future=torch.tensor(
            [
                [
                    [8.0, 15.0],
                ]
            ]
        ),
        sem=torch.zeros(1, 2),
        lw=torch.tensor(
            [
                [5.0, 2.0]
            ]
        ),
        past_vis=torch.ones(1, 1),
        future_vis=torch.ones(1, 1),
    )

    state_normalizer = MeanStdNormalizer(
        mean_vals=torch.tensor(
            [10.0, 20.0]
        ),
        std_vals=torch.tensor(
            [2.0, 5.0]
        ),
    )

    att_normalizer = MeanStdNormalizer(
        mean_vals=torch.tensor(
            [4.0, 1.0]
        ),
        std_vals=torch.tensor(
            [1.0, 0.5]
        ),
    )

    normalize_scene_graph(
        graph,
        state_normalizer,
        att_normalizer,
    )

    assert torch.allclose(
        graph.past,
        torch.tensor(
            [
                [
                    [1.0, 1.0],
                ]
            ]
        ),
    )

    assert torch.allclose(
        graph.future,
        torch.tensor(
            [
                [
                    [-1.0, -1.0],
                ]
            ]
        ),
    )

    assert torch.allclose(
        graph.lw,
        torch.tensor(
            [
                [1.0, 2.0]
            ]
        ),
    )
