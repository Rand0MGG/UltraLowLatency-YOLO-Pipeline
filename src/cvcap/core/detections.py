from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class DetBox:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: Tuple[float, float, float, float]
    kpts_xy: Optional[Tuple[Tuple[float, float], ...]] = None
    kpts_conf: Optional[Tuple[float, ...]] = None
