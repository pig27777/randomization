"""F12：空袋端袋膜捏取改用“聚拢”条件，承载只计拉离和水平分量（flex 提速第 2 轮计划）。

与 F7 的 FilmPinch 相比只改两处：
1. 捏住条件：两指尖都接触上层膜，且合拢结束时指间带内的上层膜被聚起，聚拢量 ≥ GATHER_MIN。
   聚拢量 = 压完时两指尖正下方的上层膜材料点（用所在三角形的重心坐标固定在膜上），到合拢结束时
   二者水平距离缩短了多少；指尖在膜上打滑时材料点不动、聚拢量约为 0。35 mm 网格下指间带内往往
   只有 1–2 个顶点，按膜的拱起长度量不出来，故按材料点距离量。压入深度条件不再使用（平铺膜贴桌时
   压不进去）。压完时（夹紧弹簧是否启用的判定）只要求两指尖都接触上层膜，聚拢在合拢结束时判。
2. 承载：连接力中把膜压向桌面的竖直分量不计，只计拉离（向上）与水平分量；仍是 5 ms 低通、
   超过 bag_pinch_coeff × 当前夹持力即全部解开。
GATHER_MIN 取四部分软包探索中用户给的聚拢阈值初值 8 mm（test/bag_model/four_part_pack_demo.py），
不是验收标准；正式取值待真机结构检查（论文计划 3.1 第 3 条）。
"""
import numpy as np

from film_pinch import FilmPinch

GATHER_MIN = .008


class FilmGatherPinch(FilmPinch):
    def __init__(self, sim, coefficient):
        super().__init__(sim, coefficient)
        self.last_gather = None
        self.anchors = None
        self.upper_face_ids = np.flatnonzero(self.upper_faces)

    def _material_point(self, xy):
        """上层膜上水平投影包含 xy 的三角形及重心坐标（不包含时取重心坐标截断后最近的一块）。"""
        v = self.sim.bag_points()
        best = None
        for f in self.upper_face_ids:
            a, b, c = v[self.sim.meta['faces'][f], :2]
            m = np.c_[b - a, c - a]
            if abs(np.linalg.det(m)) < 1e-12:
                continue
            l1, l2 = np.linalg.solve(m, xy - a)
            w = np.array([1 - l1 - l2, l1, l2])
            outside = -min(w.min(), 0.)
            if best is None or outside < best[0]:
                best = (outside, f, np.clip(w, 0, None) / np.clip(w, 0, None).sum())
        return best[1], best[2]

    def _where(self, anchor):
        f, w = anchor
        return w @ self.sim.bag_points()[self.sim.meta['faces'][f]]

    def gather(self, tips):
        """压完时记下的两个材料点之间水平距离的缩短量（m）；未记录时为 None。"""
        if self.anchors is None:
            return None
        now = np.linalg.norm(self._where(self.anchors[0])[:2] - self._where(self.anchors[1])[:2])
        return float(self.anchor_distance - now)

    def press_eligible(self, tips):
        if self.active is not None or self.broken:
            return None
        if not all(self.top_contacts()):
            return None
        if self.anchors is None:
            self.anchors = [self._material_point(t[:2]) for t in tips]
            self.anchor_distance = float(np.linalg.norm(self._where(self.anchors[0])[:2] -
                                                        self._where(self.anchors[1])[:2]))
        return self.covered(tips) or None

    def candidate(self, tips):
        found = self.press_eligible(tips)
        self.last_gather = self.gather(tips)
        if found is None or self.last_gather is None or self.last_gather < GATHER_MIN:
            return None
        return found

    def try_bind(self, tips):
        bound = super().try_bind(tips)
        if bound:
            self.events[-1]['gather_mm'] = 1e3 * self.last_gather
        return bound

    def check_load(self, dt, grip_force, force_vector=None):
        if self.active is None:
            return False
        f = self.force_vector() if force_vector is None else np.asarray(force_vector, float)
        # force_vector 是夹持框架一侧的反力：膜被压向桌面时框架受力 z > 0，这部分不计。
        return super().check_load(dt, grip_force, np.array([f[0], f[1], min(f[2], 0.)]))
