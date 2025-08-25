"""3D Multiple Object Detection Evaluator."""

import contextlib
import copy
import datetime
import io
import itertools
import json
import os
import time
from collections import defaultdict

import numpy as np
import pycocotools.mask as maskUtils
import torch
from pycocotools.cocoeval import COCOeval
from scipy.spatial.distance import cdist
from terminaltables import AsciiTable
from vis4d.common.array import array_to_numpy
from vis4d.common.distributed import all_gather_object_cpu
from vis4d.common.typing import (
    ArrayLike,
    DictStrAny,
    GenericFunc,
    MetricLogs,
    NDArrayF32,
    NDArrayI64,
)
from vis4d.data.const import AxisMode
from vis4d.eval.base import Evaluator
from vis4d.eval.coco.detect import xyxy_to_xywh
from vis4d.op.box.box3d import boxes3d_to_corners
from vis4d.op.geometry.rotation import quaternion_to_matrix

from opendet3d.data.datasets.coco3d import COCO3D
from opendet3d.op.box.box3d import box3d_overlap
from opendet3d.op.geometric.rotation import so3_relative_angle


class Detect3DEvaluator(Evaluator):
    """3D object detection evaluation with COCO format."""

    def __init__(
        self,
        det_map: dict[str, int],
        cat_map: dict[str, int],
        annotation: str,
        id2name: dict[int, str] | None = None,
        per_class_eval: bool = True,
        eval_prox: bool = False,
        iou_type: str = "bbox",
        num_columns: int = 6,
        base_classes: list[str] | None = None,
    ) -> None:
        """Create an instance of the class."""
        if id2name is None:
            self.id2name = {v: k for k, v in det_map.items()}
        else:
            self.id2name = id2name

        self.annotation = annotation
        self.per_class_eval = per_class_eval
        self.eval_prox = eval_prox
        self.iou_type = iou_type
        self.num_columns = num_columns
        self.base_classes = base_classes

        self.tp_errors = ["ATE", "AOE", "ASE"]

        category_names = sorted(det_map, key=det_map.get)

        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_gt = COCO3D([annotation], category_names)

        self.cat_map = cat_map

        self.bbox_2D_evals_per_cat_area: DictStrAny = {}
        self.bbox_3D_evals_per_cat_area: DictStrAny = {}
        self._predictions: list[DictStrAny] = []

    def __repr__(self) -> str:
        """Returns the string representation of the object."""
        return f"3D Object Detection Evaluator with {self.annotation}"

    @property
    def metrics(self) -> list[str]:
        """Supported metrics.

        Returns:
            list[str]: Metrics to evaluate.
        """
        return ["2D", "3D"]

    def gather(self, gather_func: GenericFunc) -> None:
        """Accumulate predictions across processes."""
        all_preds = all_gather_object_cpu(
            self._predictions, use_system_tmp=False
        )
        if all_preds is not None:
            self._predictions = list(itertools.chain(*all_preds))

    def reset(self) -> None:
        """Reset the saved predictions to start new round of evaluation."""
        self._predictions.clear()
        self.bbox_2D_evals_per_cat_area.clear()
        self.bbox_3D_evals_per_cat_area.clear()

    def process_batch(
        self,
        coco_image_id: list[int],
        pred_boxes: list[ArrayLike],
        pred_scores: list[ArrayLike],
        pred_classes: list[ArrayLike],
        pred_boxes3d: list[ArrayLike] | None = None,
    ) -> None:
        """Process sample and convert detections to coco format."""
        for i, image_id in enumerate(coco_image_id):
            boxes = array_to_numpy(
                pred_boxes[i].to(torch.float32), n_dims=None, dtype=np.float32
            )
            scores = array_to_numpy(
                pred_scores[i].to(torch.float32), n_dims=None, dtype=np.float32
            )
            classes = array_to_numpy(
                pred_classes[i], n_dims=None, dtype=np.int64
            )

            if pred_boxes3d is not None:
                boxes3d = array_to_numpy(
                    pred_boxes3d[i].to(torch.float32),
                    n_dims=None,
                    dtype=np.float32,
                )
            else:
                boxes3d = None

            self._predictions_to_coco(
                image_id, boxes, boxes3d, scores, classes
            )

    def _predictions_to_coco(
        self,
        img_id: int,
        boxes: NDArrayF32,
        boxes3d: NDArrayF32 | None,
        scores: NDArrayF32,
        classes: NDArrayI64,
    ) -> None:
        """Convert predictions to COCO format."""
        boxes_xyxy = copy.deepcopy(boxes)
        boxes_xywh = xyxy_to_xywh(boxes_xyxy)

        if boxes3d is not None:
            # FIXME: Make axismode configurable
            corners_3d = boxes3d_to_corners(
                torch.from_numpy(boxes3d), AxisMode.OPENCV
            )

        for i, (box, box_score, box_class) in enumerate(
            zip(boxes_xywh, scores, classes)
        ):
            xywh = box.tolist()

            result = {
                "image_id": img_id,
                "bbox": xywh,
                "category_id": self.cat_map[self.id2name[box_class.item()]],
                "score": box_score.item(),
            }

            # mapping to Omni3D format
            if boxes3d is not None:
                result["center_cam"] = boxes3d[i][:3].tolist()

                # wlh to whl
                result["dimensions"] = boxes3d[i][[3, 5, 4]].tolist()

                result["R_cam"] = (
                    quaternion_to_matrix(torch.from_numpy(boxes3d[i][6:10]))
                    .numpy()
                    .tolist()
                )

                corners = corners_3d[i].numpy().tolist()

                result["bbox3D"] = [
                    corners[6],
                    corners[4],
                    corners[0],
                    corners[2],
                    corners[7],
                    corners[5],
                    corners[1],
                    corners[3],
                ]

                result["depth"] = boxes3d[i][2].item()

            self._predictions.append(result)

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate predictions."""
        if metric == "2D":
            metrics = ["AP", "AP50", "AP75", "AP95", "APs", "APm", "APl"]
        else:
            if self.iou_type == "bbox":
                metrics = ["AP", "AP15", "AP25", "AP50", "APn", "APm", "APf"]
                main_metric = "AP"
            else:
                metrics = ["AP", "ATE", "ASE", "AOE", "ODS"]
                main_metric = "ODS"

            if self.base_classes is not None:
                metrics += [f"{main_metric}_Base", f"{main_metric}_Novel"]

        if len(self._predictions) == 0:
            return {m: 0.0 for m in metrics}, "No predictions to evaluate."

        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = self._coco_gt.loadRes(self._predictions)

            assert coco_dt is not None
            evaluator = Detect3Deval(
                self._coco_gt,
                coco_dt,
                mode=metric,
                eval_prox=self.eval_prox,
                iou_type=self.iou_type,
            )
            evaluator.evaluate()
            evaluator.accumulate()

        if self.iou_type == "bbox":
            log_str = "\n" + evaluator.summarize()

        # precision: (iou, recall, cls, area range, max dets)
        precisions = evaluator.eval["precision"]
        assert len(self._coco_gt.getCatIds()) == precisions.shape[2]

        if metric == "2D":
            self.bbox_2D_evals_per_cat_area = evaluator.evals_per_cat_area

            score_dict = dict(zip(metrics, evaluator.stats))
        else:
            if self.iou_type == "bbox":
                self.bbox_3D_evals_per_cat_area = evaluator.evals_per_cat_area

                score_dict = dict(zip(metrics, evaluator.stats))
            else:
                trans_tp_errors = evaluator.eval["trans_tp_errors"]
                rot_tp_errors = evaluator.eval["rot_tp_errors"]
                scale_tp_errors = evaluator.eval["scale_tp_errors"]

                precision = precisions[:, :, :, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    mAP = np.mean(precision).item()
                else:
                    mAP = float("nan")

                trans_tp = trans_tp_errors[:, :, :, 0, -1]
                trans_tp = trans_tp[trans_tp > -1]

                rot_tp = rot_tp_errors[:, :, :, 0, -1]
                rot_tp = rot_tp[rot_tp > -1]

                scale_tp = scale_tp_errors[:, :, :, 0, -1]
                scale_tp = scale_tp[scale_tp > -1]

                if trans_tp.size:
                    mATE = np.mean(trans_tp).item()
                    mAOE = np.mean(rot_tp).item()
                    mASE = np.mean(scale_tp).item()

                    mODS = (
                        np.sum(mAP * 3 + (1 - mATE) + (1 - mAOE) + (1 - mASE))
                        / 6
                    )

                else:
                    mATE = float("nan")
                    mAOE = float("nan")
                    mASE = float("nan")
                    mODS = float("nan")

                score_dict = {
                    "AP": mAP,
                    "ATE": mATE,
                    "ASE": mASE,
                    "AOE": mAOE,
                    "ODS": mODS,
                }

                log_str = "\nHigh-level metrics:"
                for k, v in score_dict.items():
                    log_str += f"\n{k}: {v:.4f}"

        if self.per_class_eval:
            results_per_category = []
            score_base_list = []
            score_novel_list = []

            for idx, cat_id in enumerate(self._coco_gt.getCatIds()):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = self._coco_gt.loadCats(cat_id)[0]
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    ap = np.mean(precision).item()
                else:
                    ap = float("nan")

                if self.iou_type == "dist":
                    trans_tp = trans_tp_errors[:, :, idx, 0, -1]
                    trans_tp = trans_tp[trans_tp > -1]

                    rot_tp = rot_tp_errors[:, :, idx, 0, -1]
                    rot_tp = rot_tp[rot_tp > -1]

                    scale_tp = scale_tp_errors[:, :, idx, 0, -1]
                    scale_tp = scale_tp[scale_tp > -1]

                    if trans_tp.size:
                        ate = np.mean(trans_tp).item()
                        aoe = np.mean(rot_tp).item()
                        ase = np.mean(scale_tp).item()

                        ods = (
                            np.sum(ap * 3 + (1 - ate) + (1 - aoe) + (1 - ase))
                            / 6
                        )

                    else:
                        ate = float("nan")
                        aoe = float("nan")
                        ase = float("nan")
                        ods = float("nan")

                    results_per_category.append(
                        (
                            f'{nm["name"]}',
                            f"{ap:0.3f}",
                            f"{ate:0.3f}",
                            f"{ase:0.3f}",
                            f"{aoe:0.3f}",
                            f"{ods:0.3f}",
                        )
                    )
                else:
                    results_per_category.append(
                        (f'{nm["name"]}', f"{ap:0.3f}")
                    )

                if self.base_classes is not None:
                    if self.iou_type == "dist":
                        score = ods
                    else:
                        score = ap

                    if nm["name"] in self.base_classes:
                        score_base_list.append(score)
                    else:
                        score_novel_list.append(score)

            results_flatten = list(itertools.chain(*results_per_category))

            if self.iou_type == "dist":
                num_columns = 6
                headers = ["category", "AP", "ATE", "ASE", "AOE", "ODS"]
            else:
                num_columns = min(
                    self.num_columns, len(results_per_category) * 2
                )
                headers = ["category", "AP"] * (num_columns // 2)
            results = itertools.zip_longest(
                *[results_flatten[i::num_columns] for i in range(num_columns)]
            )
            table_data = [headers] + list(results)
            table = AsciiTable(table_data)
            log_str = f"\n{table.table}\n{log_str}"

        if self.base_classes is not None:
            score_dict[f"{main_metric}_Base"] = np.mean(score_base_list).item()
            score_dict[f"{main_metric}_Novel"] = np.mean(
                score_novel_list
            ).item()

        return score_dict, log_str

    def save(
        self, metric: str, output_dir: str, prefix: str | None = None
    ) -> None:
        """Save the results to json files."""
        assert metric in self.metrics

        if prefix is not None:
            result_folder = os.path.join(output_dir, prefix)
            os.makedirs(result_folder, exist_ok=True)
        else:
            result_folder = output_dir

        result_file = os.path.join(
            result_folder, f"detect_{metric}_results.json"
        )

        with open(result_file, mode="w", encoding="utf-8") as f:
            json.dump(self._predictions, f)


class Detect3Deval(COCOeval):
    """COCOeval Wrapper for 2D and 3D box evaluation.

    Now it support bbox IoU matching only.
    """

    def __init__(
        self,
        cocoGt=None,
        cocoDt=None,
        mode: str = "2D",
        iou_type: str = "bbox",
        eval_prox: bool = False,
    ):
        """Initialize Detect3Deval using coco APIs for Gt and Dt.

        Args:
            cocoGt: COCO object with ground truth annotations
            cocoDt: COCO object with detection results
            mode: (str) defines whether to evaluate 2D or 3D performance.
                One of {"2D", "3D"}
            eval_prox: (bool) if True, performs "Proximity Evaluation", i.e.
                evaluates detections in the proximity of the ground truth2D
                boxes. This is used for datasets which are not exhaustively
                annotated.
        """
        if mode not in {"2D", "3D"}:
            raise Exception(f"{mode} mode is not supported")
        self.mode = mode
        self.iou_type = iou_type
        self.eval_prox = eval_prox

        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API

        # per-image per-category evaluation results [KxAxI] elements
        self.evalImgs = defaultdict(list)

        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Detect3DParams(mode=mode, iouType=iou_type)  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts

        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

        self.evals_per_cat_area = None

    def _prepare(self) -> None:
        """Prepare ._gts and ._dts for evaluation based on params."""
        p = self.params

        if p.useCats:
            gts = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
            dts = self.cocoDt.loadAnns(
                self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )

        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # set ignore flag
        ignore_flag = "ignore2D" if self.mode == "2D" else "ignore3D"
        for gt in gts:
            gt[ignore_flag] = gt[ignore_flag] if ignore_flag in gt else 0

        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation

        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)

        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)

        self.evalImgs = defaultdict(
            list
        )  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def accumulate(self, p=None) -> None:
        """Accumulate per image evaluation and store the result in self.eval.

        Args:
            p: input params for evaluation
        """
        print("Accumulating evaluation results...")
        assert self.evalImgs, "Please run evaluate() first"

        tic = time.time()

        # allows input customized parameters
        if p is None:
            p = self.params

        p.catIds = p.catIds if p.useCats == 1 else [-1]

        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)

        precision = -np.ones(
            (T, R, K, A, M)
        )  # -1 for the precision of absent categories
        trans_tp_errors = -np.ones((T, R, K, A, M))
        rot_tp_errors = -np.ones((T, R, K, A, M))
        scale_tp_errors = -np.ones((T, R, K, A, M))
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval

        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)

        # get inds to evaluate
        catid_list = [k for n, k in enumerate(p.catIds) if k in setK]
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [
            n
            for n, a in enumerate(map(lambda x: tuple(x), p.areaRng))
            if a in setA
        ]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]

        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)

        has_precomputed_evals = not (self.evals_per_cat_area is None)

        if has_precomputed_evals:
            evals_per_cat_area = self.evals_per_cat_area
        else:
            evals_per_cat_area = {}

        # retrieve E at each category, area range, and max number of detections
        for k, (k0, catId) in enumerate(zip(k_list, catid_list)):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0

                if has_precomputed_evals:
                    E = evals_per_cat_area[(catId, a)]

                else:
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    evals_per_cat_area[(catId, a)] = E

                if len(E) == 0:
                    continue

                for m, maxDet in enumerate(m_list):

                    dtScores = np.concatenate(
                        [e["dtScores"][0:maxDet] for e in E]
                    )

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e["dtMatches"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]
                    dtIg = np.concatenate(
                        [e["dtIgnore"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])
                    npig = np.count_nonzero(gtIg == 0)

                    if npig == 0:
                        continue

                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(
                        np.logical_not(dtm), np.logical_not(dtIg)
                    )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float64)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float64)

                    # Compute TP error
                    if self.iou_type == "dist":
                        tems = np.concatenate(
                            [e["dtTranslationError"][:, 0:maxDet] for e in E],
                            axis=1,
                        )[:, inds]

                        oems = np.concatenate(
                            [e["dtOrientationError"][:, 0:maxDet] for e in E],
                            axis=1,
                        )[:, inds]

                        sems = np.concatenate(
                            [e["dtScaleError"][:, 0:maxDet] for e in E], axis=1
                        )[:, inds]

                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))

                        q = np.zeros((R,))
                        ss = np.zeros((R,))
                        tran_tp_error = np.ones((R,))
                        rot_tp_error = np.ones((R,))
                        scale_tp_error = np.ones((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]

                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()
                        tran_tp_error = tran_tp_error.tolist()
                        rot_tp_error = rot_tp_error.tolist()
                        scale_tp_error = scale_tp_error.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side="left")

                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                                if self.iou_type == "dist":
                                    tran_tp_error[ri] = tems[t][pi]
                                    rot_tp_error[ri] = oems[t][pi]
                                    scale_tp_error[ri] = sems[t][pi]
                        except:
                            pass

                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)

                        if self.iou_type == "dist":
                            trans_tp_errors[t, :, k, a, m] = np.array(
                                tran_tp_error
                            )
                            rot_tp_errors[t, :, k, a, m] = np.array(
                                rot_tp_error
                            )
                            scale_tp_errors[t, :, k, a, m] = np.array(
                                scale_tp_error
                            )

        self.evals_per_cat_area = evals_per_cat_area

        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
            "trans_tp_errors": trans_tp_errors,
            "rot_tp_errors": rot_tp_errors,
            "scale_tp_errors": scale_tp_errors,
        }

        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def evaluate(self) -> None:
        """Run per image evaluation on given images.

        It will store results (a list of dict) in self.evalImgs
        """
        print("Running per image evaluation...")

        p = self.params
        print(f"Evaluate annotation type *{p.iouType}*")

        tic = time.time()

        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))

        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()

        catIds = p.catIds if p.useCats else [-1]

        # loop through images, area range, max detection number
        self.ious = {
            (imgId, catId): self.computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds
        }

        maxDet = p.maxDets[-1]

        self.evalImgs = [
            self.evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]

        self._paramsEval = copy.deepcopy(self.params)

        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def computeIoU(self, imgId, catId) -> tuple[NDArrayF32, NDArrayF32]:
        """Computes the IoUs by sorting based on score"""
        p = self.params

        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]

        if len(gt) == 0 and len(dt) == 0:
            return []

        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        if self.mode == "2D":
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        elif self.mode == "3D":
            g = [g["bbox3D"] for g in gt]
            d = [d["bbox3D"] for d in dt]

        # compute iou between each dt and gt region
        # iscrowd is required in builtin maskUtils so we
        # use a dummy buffer for it
        iscrowd = [0 for _ in gt]
        if self.mode == "2D":
            ious = maskUtils.iou(d, g, iscrowd)
        elif len(d) > 0 and len(g) > 0:
            if p.iouType == "bbox":
                dd = torch.tensor(d, dtype=torch.float32)
                gg = torch.tensor(g, dtype=torch.float32)

                ious = box3d_overlap(dd, gg).cpu().numpy()
            else:
                ious = np.zeros((len(d), len(g)))

                dd = [d["center_cam"] for d in dt]
                gg = [g["center_cam"] for g in gt]

                ious = cdist(dd, gg, metric="euclidean")
        else:
            ious = []

        in_prox = None

        if self.eval_prox:
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
            iscrowd = [0 for o in gt]
            ious2d = maskUtils.iou(d, g, iscrowd)

            if type(ious2d) == list:
                in_prox = []

            else:
                in_prox = ious2d > p.proximity_thresh

        return ious, in_prox

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        Perform evaluation for single category and image
        Returns:
            dict (single image results)
        """

        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]

        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]

        if len(gt) == 0 and len(dt) == 0:
            return None

        flag_range = "area" if self.mode == "2D" else "depth"
        flag_ignore = "ignore2D" if self.mode == "2D" else "ignore3D"

        for g in gt:
            if g[flag_ignore] or (
                g[flag_range] < aRng[0] or g[flag_range] > aRng[1]
            ):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]

        # load computed ious
        ious = (
            self.ious[imgId, catId][0][:, gtind]
            if len(self.ious[imgId, catId][0]) > 0
            else self.ious[imgId, catId][0]
        )

        if self.eval_prox:
            in_prox = (
                self.ious[imgId, catId][1][:, gtind]
                if len(self.ious[imgId, catId][1]) > 0
                else self.ious[imgId, catId][1]
            )

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        tem = np.ones((T, D))  # Translation Error
        sem = np.ones((T, D))  # Scale Error
        oem = np.ones((T, D))  # Oritentation Error
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))

        dist_thres = 1
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):

                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1

                    for gind, g in enumerate(gt):
                        # in case of proximity evaluation, if not in proximity continue
                        if self.eval_prox and not in_prox[dind, gind]:
                            continue

                        # if this gt already matched, continue
                        if gtm[tind, gind] > 0:
                            continue

                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break

                        # continue to next gt unless better match made
                        if p.iouType == "bbox" and ious[dind, gind] < iou:
                            continue

                        if p.iouType == "dist":
                            # Compute Object Radius
                            gt_obj_radius = (
                                np.linalg.norm(np.array(g["dimensions"])) / 2
                            )
                            if ious[dind, gind] > gt_obj_radius * iou:
                                continue
                            else:
                                dist_thres = gt_obj_radius * iou

                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind

                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue

                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]

                    if p.iouType == "dist":
                        # Translation Error
                        tem[tind, dind] = np.linalg.norm(
                            np.array(d["center_cam"])
                            - np.array(gt[m]["center_cam"])
                        ) / (dist_thres)

                        # Orientation Error
                        oem[tind, dind] = (
                            so3_relative_angle(
                                torch.tensor(d["R_cam"])[None],
                                torch.tensor(gt[m]["R_cam"])[None],
                                cos_bound=1e-2,
                                eps=1e-2,
                            ).item()
                            / np.pi
                        )

                        # Scale Error
                        min_whl = np.minimum(
                            d["dimensions"], gt[m]["dimensions"]
                        )
                        volume_annotation = np.prod(gt[m]["dimensions"])
                        volume_result = np.prod(d["dimensions"])

                        intersection = np.prod(min_whl)
                        union = (
                            volume_annotation + volume_result - intersection
                        )
                        scale_iou = intersection / union

                        sem[tind, dind] = 1 - scale_iou

        # set unmatched detections outside of area range to ignore
        a = np.array(
            [d[flag_range] < aRng[0] or d[flag_range] > aRng[1] for d in dt]
        ).reshape((1, len(dt)))

        dtIg = np.logical_or(
            dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0))
        )

        # in case of proximity evaluation, ignore detections which are far from gt regions
        if self.eval_prox and len(in_prox) > 0:
            dt_far = in_prox.any(1) == 0
            dtIg = np.logical_or(
                dtIg, np.repeat(dt_far.reshape((1, len(dt))), T, 0)
            )

        # store results for given image and category
        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
            "dtTranslationError": tem,
            "dtScaleError": sem,
            "dtOrientationError": oem,
        }

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(
            mode, ap=1, iouThr=None, areaRng="all", maxDets=100, log_str=""
        ):
            p = self.params
            eval = self.eval

            if mode == "2D":
                if self.iou_type == "bbox":
                    iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
                else:
                    iStr = " {:<18} {} @[ Dist={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"

            elif mode == "3D":
                if self.iou_type == "bbox":
                    iStr = " {:<18} {} @[ IoU={:<9} | depth={:>6s} | maxDets={:>3d} ] = {:0.3f}"
                else:
                    iStr = " {:<18} {} @[ Dist={:<9} | depth={:>6s} | maxDets={:>3d} ] = {:0.3f}"

            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"

            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [
                i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng
            ]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if ap == 1:

                # dimension of precision: [TxRxKxAxM]
                s = eval["precision"]

                # IoU
                if iouThr is not None:
                    t = np.where(np.isclose(iouThr, p.iouThrs.astype(float)))[
                        0
                    ]
                    s = s[t]

                s = s[:, :, :, aind, mind]

            else:
                # dimension of recall: [TxKxAxM]
                s = eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]

            if len(s[s > -1]) == 0:
                mean_s = -1

            else:
                mean_s = np.mean(s[s > -1])

            if log_str != "":
                log_str += "\n"

            log_str += "mode={} ".format(mode) + iStr.format(
                titleStr, typeStr, iouStr, areaRng, maxDets, mean_s
            )

            return mean_s, log_str

        def _summarizeDets(mode):

            params = self.params

            # Define the thresholds to be printed
            if mode == "2D":
                thres = [0.5, 0.75, 0.95]
            else:
                if self.iou_type == "bbox":
                    thres = [0.15, 0.25, 0.50]
                else:
                    thres = [0.5, 0.75, 1.0]

            stats = np.zeros((13,))
            stats[0], log_str = _summarize(mode, 1)

            stats[1], log_str = _summarize(
                mode,
                1,
                iouThr=thres[0],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[2], log_str = _summarize(
                mode,
                1,
                iouThr=thres[1],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[3], log_str = _summarize(
                mode,
                1,
                iouThr=thres[2],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[4], log_str = _summarize(
                mode,
                1,
                areaRng=params.areaRngLbl[1],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[5], log_str = _summarize(
                mode,
                1,
                areaRng=params.areaRngLbl[2],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[6], log_str = _summarize(
                mode,
                1,
                areaRng=params.areaRngLbl[3],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[7], log_str = _summarize(
                mode, 0, maxDets=params.maxDets[0], log_str=log_str
            )

            stats[8], log_str = _summarize(
                mode, 0, maxDets=params.maxDets[1], log_str=log_str
            )

            stats[9], log_str = _summarize(
                mode, 0, maxDets=params.maxDets[2], log_str=log_str
            )

            stats[10], log_str = _summarize(
                mode,
                0,
                areaRng=params.areaRngLbl[1],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[11], log_str = _summarize(
                mode,
                0,
                areaRng=params.areaRngLbl[2],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[12], log_str = _summarize(
                mode,
                0,
                areaRng=params.areaRngLbl[3],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            return stats, log_str

        if not self.eval:
            raise Exception("Please run accumulate() first")

        stats, log_str = _summarizeDets(self.mode)
        self.stats = stats

        return log_str


class Detect3DParams:
    """Params for the 3d detection evaluation API."""

    def __init__(
        self,
        mode: str = "2D",
        iouType: str = "bbox",
        proximity_thresh: float = 0.3,
    ) -> None:
        """Create an instance of Detect3DParams.

        Args:
            mode: (str) defines whether to evaluate 2D or 3D performance.
            iouType: (str) defines the type of IoU to be used for evaluation.
            proximity_thresh (float): It defines the neighborhood when
                evaluating on non-exhaustively annotated datasets.
        """
        assert iouType in {"bbox", "dist"}, f"Invalid iouType {iouType}."
        self.iouType = iouType

        if mode == "2D":
            self.setDet2DParams()
        elif mode == "3D":
            self.setDet3DParams()
        else:
            raise Exception(f"{mode} mode is not supported")
        self.mode = mode
        self.proximity_thresh = proximity_thresh

    def setDet2DParams(self) -> None:
        """Set parameters for 2D detection evaluation."""
        self.imgIds = []
        self.catIds = []

        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )

        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )

        self.maxDets = [1, 10, 100]
        self.areaRng = [
            [0**2, 1e5**2],
            [0**2, 32**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]

        self.areaRngLbl = ["all", "small", "medium", "large"]
        self.useCats = 1

    def setDet3DParams(self) -> None:
        """Set parameters for 3D detection evaluation."""
        self.imgIds = []
        self.catIds = []

        # np.arange causes trouble. The data point on arange is slightly
        # larger than the true value
        if self.iouType == "bbox":
            self.iouThrs = np.linspace(
                0.05,
                0.5,
                int(np.round((0.5 - 0.05) / 0.05)) + 1,
                endpoint=True,
            )
        else:
            self.iouThrs = np.linspace(
                0.5, 1.0, int(np.round((1.00 - 0.5) / 0.05)) + 1, endpoint=True
            )

        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )

        self.maxDets = [1, 10, 100]
        self.areaRng = [[0, 1e5], [0, 10], [10, 35], [35, 1e5]]
        self.areaRngLbl = ["all", "near", "medium", "far"]
        self.useCats = 1
