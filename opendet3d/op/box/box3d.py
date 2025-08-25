"""Box3D ops."""

from torch import Tensor
from vis4d_cuda_ops import iou_box3d

from opendet3d.op.box.iou_box3d import check_coplanar, check_nonzero


def box3d_overlap(
    boxes_dt: Tensor,
    boxes_gt: Tensor,
    eps_coplanar: float = 1e-4,
    eps_nonzero: float = 1e-8,
) -> Tensor:
    """
    Computes the intersection of 3D boxes_dt and boxes_gt.

    Inputs boxes_dt, boxes_gt are tensors of shape (B, 8, 3)
    (where B doesn't have to be the same for boxes_dt and boxes_gt),
    containing the 8 corners of the boxes, as follows:

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)


    NOTE: Throughout this implementation, we assume that boxes
    are defined by their 8 corners exactly in the order specified in the
    diagram above for the function to give correct results. In addition
    the vertices on each plane must be coplanar.
    As an alternative to the diagram, this is a unit bounding
    box which has the correct vertex ordering:

    box_corner_vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

    Args:
        boxes_dt: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
        boxes_gt: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
    Returns:
        iou: (N, M) tensor of the intersection over union which is
            defined as: `iou = vol / (vol1 + vol2 - vol)`
    """
    # Make sure predictions are coplanar and nonzero
    invalid_coplanar = ~check_coplanar(boxes_dt, eps=eps_coplanar)
    invalid_nonzero = ~check_nonzero(boxes_dt, eps=eps_nonzero)

    ious = iou_box3d(boxes_dt, boxes_gt)[1]

    # Offending boxes are set to zero IoU
    if invalid_coplanar.any():
        ious[invalid_coplanar] = 0
        print(
            "Warning: skipping {:d} non-coplanar boxes at eval.".format(
                int(invalid_coplanar.float().sum())
            )
        )

    if invalid_nonzero.any():
        ious[invalid_nonzero] = 0
        print(
            "Warning: skipping {:d} zero volume boxes at eval.".format(
                int(invalid_nonzero.float().sum())
            )
        )

    return ious
