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

def transform2frame(
    frame: torch.Tensor,
    poses: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """
    Transform poses between global coordinates and a local reference frame.

    Parameters
    ----------
    frame:
        Shape (B, 3) or (B, 4)

        Heading-angle representation:
            [x, y, heading]

        Heading-vector representation:
            [x, y, hx, hy]

    poses:
        Shape (B, N, 3) or (B, N, 4)

    inverse:
        False:
            global -> local

        True:
            local -> global

    Returns
    -------
    torch.Tensor
        Transformed poses with the same shape as `poses`.
    """
    if frame.ndim != 2:
        raise ValueError("frame must have shape (B, D)")

    if poses.ndim != 3:
        raise ValueError("poses must have shape (B, N, D)")

    batch_size, _, pose_dim = poses.shape

    if frame.shape[0] != batch_size:
        raise ValueError(
            "frame and poses must have the same batch size"
        )

    if frame.shape[-1] != pose_dim:
        raise ValueError(
            "frame and poses must use the same pose representation"
        )

    if pose_dim not in (3, 4):
        raise ValueError(
            "pose dimension must be 3 or 4"
        )

    # ---------------------------------------------------------
    # Obtain frame heading as cosine and sine.
    # ---------------------------------------------------------

    if pose_dim == 3:
        frame_heading = frame[:, 2]

        cos_h = torch.cos(frame_heading)
        sin_h = torch.sin(frame_heading)

    else:
        cos_h = frame[:, 2]
        sin_h = frame[:, 3]

    # ---------------------------------------------------------
    # Rotation matrix.
    #
    # global -> local:
    #
    # [ cos  sin]
    # [-sin  cos]
    #
    # local -> global:
    #
    # [cos -sin]
    # [sin  cos]
    # ---------------------------------------------------------

    if inverse:
        rotation = torch.stack(
            [
                cos_h,
                -sin_h,
                sin_h,
                cos_h,
            ],
            dim=-1,
        )
    else:
        rotation = torch.stack(
            [
                cos_h,
                sin_h,
                -sin_h,
                cos_h,
            ],
            dim=-1,
        )

    rotation = rotation.reshape(
        batch_size,
        2,
        2,
    )

    # ---------------------------------------------------------
    # Transform position.
    # ---------------------------------------------------------

    frame_position = frame[:, :2].unsqueeze(1)
    pose_position = poses[..., :2]

    if inverse:
        # Local -> global
        transformed_position = torch.matmul(
            rotation.unsqueeze(1),
            pose_position.unsqueeze(-1),
        ).squeeze(-1)

        transformed_position = (
            transformed_position + frame_position
        )

    else:
        # Global -> local
        relative_position = (
            pose_position - frame_position
        )

        transformed_position = torch.matmul(
            rotation.unsqueeze(1),
            relative_position.unsqueeze(-1),
        ).squeeze(-1)

    # ---------------------------------------------------------
    # Transform heading.
    # ---------------------------------------------------------

    if pose_dim == 3:
        pose_heading = poses[..., 2]

        if inverse:
            transformed_heading = (
                pose_heading + frame[:, 2].unsqueeze(1)
            )
        else:
            transformed_heading = (
                pose_heading - frame[:, 2].unsqueeze(1)
            )

        transformed_heading = torch.atan2(
            torch.sin(transformed_heading),
            torch.cos(transformed_heading),
        )

        transformed_heading = (
            transformed_heading.unsqueeze(-1)
        )

    else:
        pose_heading_vector = poses[..., 2:4]

        transformed_heading = torch.matmul(
            rotation.unsqueeze(1),
            pose_heading_vector.unsqueeze(-1),
        ).squeeze(-1)

    return torch.cat(
        [
            transformed_position,
            transformed_heading,
        ],
        dim=-1,
    )