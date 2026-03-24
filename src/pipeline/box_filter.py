# src/pipeline/box_filter.py
import numpy as np
from typing import List
from inference.yolo_detector import DetBox

class SimpleKalman:
    """
    简易卡尔曼滤波器 (针对 [x1, y1, x2, y2] 坐标)
    """
    def __init__(self):
        self.ndim = 4 
        self.dt = 1.0 

        # 状态转移矩阵 F (模型: 位置 = 位置 + 速度*dt)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i+4] = self.dt

        # 测量矩阵 H (只测量位置)
        self.H = np.eye(4, 8, dtype=np.float32)

        # 协方差矩阵 P
        self.P = np.eye(8, dtype=np.float32) * 10.0

        # 测量噪声 R (抗抖动)
        self.R = np.eye(4, dtype=np.float32) * 10.0  

        # 过程噪声 Q (跟手度)
        self.Q = np.eye(8, dtype=np.float32) * 0.1
        
        # 初始状态 x (8, 1)
        self.x = np.zeros((8, 1), dtype=np.float32)

    def predict(self):
        # x = Fx
        self.x = np.dot(self.F, self.x)
        # P = FPF' + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        # [修复] 必须返回一维数组 (4,)，防止外部赋值报错
        return self.x[:4].flatten()

    def update(self, z):
        # [修复] 确保输入 z 是 (4, 1) 形状
        z = np.array(z).reshape((4, 1))
        
        # y = z - Hx (残差)
        y = z - np.dot(self.H, self.x)
        
        # S = HPH' + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # K = PH'S^-1 (卡尔曼增益)
        try:
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        except np.linalg.LinAlgError:
            # 极少数情况矩阵不可逆，降级处理
            K = np.zeros((8, 4), dtype=np.float32)
        
        # x = x + Ky
        self.x = self.x + np.dot(K, y)
        
        # P = (I - KH)P
        I = np.eye(8, dtype=np.float32)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)


def iou_batch(bb_test, bb_gt):
    """计算 IOU 矩阵"""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) + 
              (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                      
    return o


class BoxSmoother:
    def __init__(self, alpha: float = 0.6, iou_thresh: float = 0.3):
        self.trackers = []
        self.iou_thresh = iou_thresh
        # 动态调整噪声参数
        self.R_std = 10.0 * (1.0 - alpha) + 1.0

    def update(self, current_boxes: List[DetBox]) -> List[DetBox]:
        # 1. 预测
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict() # 现在 pos 是 (4,) 形状
            # [修复] 现在可以直接赋值了
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 2. 准备观测数据
        dets = []
        for b in current_boxes:
            dets.append(list(b.xyxy))
        dets = np.array(dets)

        matched, unmatched_dets, unmatched_trks = [], [], []
        
        # 3. IOU 匹配
        if len(self.trackers) > 0 and len(dets) > 0:
            iou_matrix = iou_batch(dets, trks[:, :4])
            
            if iou_matrix.size > 0:
                for d, det in enumerate(dets):
                    best_iou = 0
                    best_t = -1
                    for t, trk in enumerate(trks):
                        if t not in [m[1] for m in matched]:
                            if iou_matrix[d, t] > best_iou:
                                best_iou = iou_matrix[d, t]
                                best_t = t
                    
                    if best_iou >= self.iou_thresh:
                        matched.append([d, best_t])
                    else:
                        unmatched_dets.append(d)
                        
                for t in range(len(self.trackers)):
                    if t not in [m[1] for m in matched]:
                        unmatched_trks.append(t)
            else:
                unmatched_dets = list(range(len(dets)))
                unmatched_trks = list(range(len(self.trackers)))
        else:
            unmatched_dets = list(range(len(dets)))
            unmatched_trks = list(range(len(self.trackers)))

        # 4. 更新匹配成功的 Tracker
        ret_boxes = []
        for d, t in matched:
            self.trackers[t].update(dets[d])
            # 这里的 predict 返回已经是扁平的了，但在 update 后我们直接拿 self.trackers[t].x[:4].flatten() 也是一样的
            # 不过最准的是 update 后的后验估计 (Posterior)
            
            smooth_xyxy = self.trackers[t].x[:4].flatten()
            
            raw_box = current_boxes[d]
            s_box = type(raw_box)(
                cls_id=raw_box.cls_id,
                cls_name=raw_box.cls_name,
                conf=raw_box.conf,
                xyxy=tuple(smooth_xyxy),
                kpts_xy=raw_box.kpts_xy,
                kpts_conf=raw_box.kpts_conf
            )
            ret_boxes.append(s_box)

        # 5. 创建新 Tracker
        for i in unmatched_dets:
            trk = SimpleKalman()
            trk.R = np.eye(4, dtype=np.float32) * self.R_std
            
            z = np.array(dets[i]).reshape((4, 1))
            trk.x[:4] = z
            
            self.trackers.append(trk)
            ret_boxes.append(current_boxes[i])

        # 6. 移除旧 Tracker
        new_trackers = []
        for i, trk in enumerate(self.trackers):
             if i not in unmatched_trks:
                 new_trackers.append(trk)
        self.trackers = new_trackers

        return ret_boxes