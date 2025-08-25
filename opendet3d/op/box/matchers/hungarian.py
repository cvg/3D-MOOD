"""Box Hungarian Assigner."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from vis4d.op.box.box2d import bbox_iou
from vis4d.op.box.matchers.base import MatchResult


class HungarianMatcher:
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The targets don't include the no_object, so generally
    there are more predictions than targets. After the one-to-one matching, the
    un-matched are treated as backgrounds. Thus each query prediction will be
    assigned with `0` or a positive integer indicating the ground truth index:

        - 0: negative sample, no assigned gt
        - positive integer: positive sample, index (1-based) of assigned gt
    """

    def __call__(
        self,
        cost: Tensor,
        boxes: Tensor,
        targets: Tensor,
        target_classes: Tensor,
    ) -> MatchResult:
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            boxes (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            boxes_classes (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            targets (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `targets`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

            gt_depth is a single channel map
            depth_pred is per-label maps

        Returns:
            MatchResult: Matching results.
        """
        num_gts, num_bboxes = targets.size(0), boxes.size(0)

        match_iou = boxes.new_zeros((len(boxes),))

        # 1. assign -1 by default
        assigned_gt_inds = boxes.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = boxes.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return MatchResult(assigned_gt_inds, match_iou, assigned_labels)

        # 2. compute the weighted costs.
        # NOTE: We dissentangle the cost computation and Hungarian matching

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        cost = np.nan_to_num(cost)

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)

        matched_row_inds = torch.from_numpy(matched_row_inds).to(boxes.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(boxes.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0

        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = target_classes[matched_col_inds]

        pos_inds = (
            torch.nonzero(assigned_gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )

        _ious = bbox_iou(boxes[pos_inds], targets)

        for i, pid in enumerate(pos_inds):
            matched_gt_idx = assigned_gt_inds[pid] - 1
            match_iou[pid] = _ious[i, matched_gt_idx]

        return MatchResult(assigned_gt_inds, match_iou, assigned_labels)
