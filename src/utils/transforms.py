import torch


def kinematics2angle(kinematics: torch.Tensor) -> torch.Tensor:
    """
    Convert heading-vector representation to heading-angle representation.

    Input shape:
        (..., 6)

    Input state:
        [x, y, hx, hy, speed, heading_rate]

    Output state:
        [x, y, heading, speed, heading_rate]
    """
    hx = kinematics[..., 2]
    hy = kinematics[..., 3]

    heading = torch.atan2(hy, hx).unsqueeze(-1)

    return torch.cat(
        [
            kinematics[..., :2],
            heading,
            kinematics[..., 4:],
        ],
        dim=-1,
    )


def kinematics2vec(kinematics: torch.Tensor) -> torch.Tensor:
    """
    Convert heading-angle representation to heading-vector representation.

    Input shape:
        (..., 5)

    Input state:
        [x, y, heading, speed, heading_rate]

    Output state:
        [x, y, hx, hy, speed, heading_rate]
    """
    heading = kinematics[..., 2]

    hx = torch.cos(heading)
    hy = torch.sin(heading)

    heading_vector = torch.stack(
        [hx, hy],
        dim=-1,
    )

    return torch.cat(
        [
            kinematics[..., :2],
            heading_vector,
            kinematics[..., 3:],
        ],
        dim=-1,
    )