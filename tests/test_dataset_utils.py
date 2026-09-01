import torch

from src.datasets.utils import MeanStdNormalizer


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
    