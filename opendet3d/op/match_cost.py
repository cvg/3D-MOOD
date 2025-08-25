"""Matcher cost op."""

import torch
from torch import Tensor
from vis4d.op.box.box2d import bbox_iou

from opendet3d.op.box.box2d import bbox_overlaps, bbox_xyxy_to_cxcywh


class MatchCost:

    def __init__(self, weight: float = 1.0) -> None:
        """Create an instance of the class."""
        self.weight = weight


class ClassificationCost(MatchCost):
    """ClsSoftmaxCost.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ...  match_costs.match_cost import ClassificationCost
        >>> import torch
        >>> self = ClassificationCost()
        >>> cls_pred = torch.rand(4, 3)
        >>> gt_labels = torch.tensor([0, 1, 2])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(cls_pred, gt_labels)
        tensor([[-0.3430, -0.3525, -0.3045],
            [-0.3077, -0.2931, -0.3992],
            [-0.3664, -0.3455, -0.2881],
            [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight: float = 1.0) -> None:
        """Create an instance of the class."""
        super().__init__(weight=weight)

    def __call__(self, cls_pred, gt_labels) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``scores`` inside is
                predicted classification logits, of shape
                (num_queries, num_class).
            gt_instances (:obj:`InstanceData`): ``labels`` inside should have
                shape (num_gt, ).
            img_meta (Optional[dict]): _description_. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_scores = cls_pred.softmax(-1)
        cls_cost = -pred_scores[:, gt_labels]

        return cls_cost * self.weight


class BBoxL1Cost(MatchCost):
    """BBoxL1Cost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
            Defaults to 'xyxy'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ... match_costs.match_cost import BBoxL1Cost
        >>> import torch
        >>> self = BBoxL1Cost()
        >>> bbox_pred = torch.rand(1, 4)
        >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(bbox_pred, gt_bboxes, factor)
        tensor([[1.6172, 1.6422]])
    """

    def __init__(self, box_format: str = "xyxy", weight: float = 1.0) -> None:
        """Create an instance of the class."""
        super().__init__(weight=weight)
        assert box_format in ["xyxy", "xywh"]
        self.box_format = box_format

    def __call__(
        self,
        pred_bboxes,
        gt_bboxes,
        img_h,
        img_w,
    ) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        # convert box format
        if self.box_format == "xywh":
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
            pred_bboxes = bbox_xyxy_to_cxcywh(pred_bboxes)

        # normalized
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(
            0
        )
        gt_bboxes = gt_bboxes / factor
        pred_bboxes = pred_bboxes / factor

        bbox_cost = torch.cdist(pred_bboxes, gt_bboxes, p=1)

        return bbox_cost * self.weight


class IoUCost(MatchCost):
    """IoUCost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        iou_mode (str): iou mode such as 'iou', 'giou'. Defaults to 'giou'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ... match_costs.match_cost import IoUCost
        >>> import torch
        >>> self = IoUCost()
        >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
        >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> self(bboxes, gt_bboxes)
        tensor([[-0.1250,  0.1667],
            [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode: str = "giou", weight: float = 1.0):
        super().__init__(weight=weight)
        self.iou_mode = iou_mode

    def __call__(
        self,
        pred_bboxes,
        gt_bboxes,
    ):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        # avoid fp16 overflow
        if pred_bboxes.dtype == torch.float16:
            fp16 = True
            pred_bboxes = pred_bboxes.to(torch.float32)
        else:
            fp16 = False

        if self.iou_mode == "iou":
            overlaps = bbox_iou(pred_bboxes, gt_bboxes)
        else:
            overlaps = bbox_overlaps(
                pred_bboxes, gt_bboxes, mode=self.iou_mode
            )

        if fp16:
            overlaps = overlaps.to(torch.float16)

        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight


class BinaryFocalLossCost(MatchCost):
    """BinaryFocalLossCost.

    Args:
        alpha (Union[float, int]): focal_loss alpha. Defaults to 0.25.
        gamma (Union[float, int]): focal_loss gamma. Defaults to 2.
        eps (float): Defaults to 1e-12.
        binary_input (bool): Whether the input is binary. Currently,
            binary_input = True is for masks input, binary_input = False
            is for label input. Defaults to False.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        eps: float = 1e-12,
        binary_input: bool = False,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight=weight)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def _focal_loss_cost(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        cls_pred = cls_pred.sigmoid()
        neg_cost = (
            -(1 - cls_pred + self.eps).log()
            * (1 - self.alpha)
            * cls_pred.pow(self.gamma)
        )
        pos_cost = (
            -(cls_pred + self.eps).log()
            * self.alpha
            * (1 - cls_pred).pow(self.gamma)
        )

        cls_cost = torch.einsum(
            "nc,mc->nm", pos_cost, gt_labels
        ) + torch.einsum("nc,mc->nm", neg_cost, (1 - gt_labels))
        return cls_cost * self.weight

    def __call__(
        self,
        cls_pred: Tensor,
        text_token_mask: Tensor,
        positive_map: Tensor,
    ) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels`` or ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        text_token_mask = torch.nonzero(text_token_mask).squeeze(-1)

        pred_scores = cls_pred[:, text_token_mask]
        gt_labels = positive_map[:, text_token_mask]

        return self._focal_loss_cost(pred_scores, gt_labels)
