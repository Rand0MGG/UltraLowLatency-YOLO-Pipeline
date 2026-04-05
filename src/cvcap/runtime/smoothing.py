from __future__ import annotations

from typing import List

import numpy as np

from cvcap.core.detections import DetBox


class SimpleKalman:
    def __init__(self):
        self.F = np.eye(8, dtype=np.float32)
        for index in range(4):
            self.F[index, index + 4] = 1.0
        self.H = np.eye(4, 8, dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.R = np.eye(4, dtype=np.float32) * 10.0
        self.Q = np.eye(8, dtype=np.float32) * 0.1
        self.x = np.zeros((8, 1), dtype=np.float32)

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:4].flatten()

    def update(self, measurement):
        z = np.array(measurement).reshape((4, 1))
        y = z - np.dot(self.H, self.x)
        s = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        try:
            k = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(s))
        except np.linalg.LinAlgError:
            k = np.zeros((8, 4), dtype=np.float32)
        self.x = self.x + np.dot(k, y)
        self.P = np.dot((np.eye(8, dtype=np.float32) - np.dot(k, self.H)), self.P)


def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    return wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )


class BoxSmoother:
    def __init__(self, alpha: float = 0.6, iou_thresh: float = 0.3):
        self.trackers = []
        self.iou_thresh = iou_thresh
        self.R_std = 10.0 * (1.0 - alpha) + 1.0

    def update(self, current_boxes: List[DetBox]) -> List[DetBox]:
        trks = np.zeros((len(self.trackers), 5))
        to_delete = []
        for tracker_index, tracker in enumerate(self.trackers):
            position = tracker.predict()
            trks[tracker_index, :] = [position[0], position[1], position[2], position[3], 0]
            if np.any(np.isnan(position)):
                to_delete.append(tracker_index)
        for tracker_index in reversed(to_delete):
            self.trackers.pop(tracker_index)

        dets = np.array([list(box.xyxy) for box in current_boxes]) if current_boxes else np.empty((0, 4))
        matched, unmatched_dets, unmatched_trks = [], [], []

        if len(self.trackers) > 0 and len(dets) > 0:
            iou_matrix = iou_batch(dets, trks[:, :4])
            if iou_matrix.size > 0:
                for det_index, _ in enumerate(dets):
                    best_iou = 0.0
                    best_tracker = -1
                    for tracker_index, _ in enumerate(trks):
                        if tracker_index not in [match[1] for match in matched] and iou_matrix[det_index, tracker_index] > best_iou:
                            best_iou = iou_matrix[det_index, tracker_index]
                            best_tracker = tracker_index
                    if best_iou >= self.iou_thresh:
                        matched.append([det_index, best_tracker])
                    else:
                        unmatched_dets.append(det_index)
                for tracker_index in range(len(self.trackers)):
                    if tracker_index not in [match[1] for match in matched]:
                        unmatched_trks.append(tracker_index)
            else:
                unmatched_dets = list(range(len(dets)))
                unmatched_trks = list(range(len(self.trackers)))
        else:
            unmatched_dets = list(range(len(dets)))
            unmatched_trks = list(range(len(self.trackers)))

        results = []
        for det_index, tracker_index in matched:
            self.trackers[tracker_index].update(dets[det_index])
            smooth_xyxy = self.trackers[tracker_index].x[:4].flatten()
            raw = current_boxes[det_index]
            results.append(
                type(raw)(
                    cls_id=raw.cls_id,
                    cls_name=raw.cls_name,
                    conf=raw.conf,
                    xyxy=tuple(smooth_xyxy),
                    kpts_xy=raw.kpts_xy,
                    kpts_conf=raw.kpts_conf,
                )
            )

        for det_index in unmatched_dets:
            tracker = SimpleKalman()
            tracker.R = np.eye(4, dtype=np.float32) * self.R_std
            tracker.x[:4] = np.array(dets[det_index]).reshape((4, 1))
            self.trackers.append(tracker)
            results.append(current_boxes[det_index])

        self.trackers = [tracker for index, tracker in enumerate(self.trackers) if index not in unmatched_trks]
        return results
