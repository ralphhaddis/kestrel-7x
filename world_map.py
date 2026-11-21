"""Terrain, building prism spatial grid, walkability, and line-of-sight queries for mechfight/1.

Standard library only (Python 3.9+). Loads ``v2/world.json`` and ``v2/terrain.i16`` from
``hello["map"]["dir"]`` and indexes building, stadium, and bridge prisms into a 16.0 m 2D
spatial hash grid for sub-millisecond ``ground_z``, ``is_walkable``, ``path_clear``, and
``has_los`` queries.
"""
from __future__ import annotations

import array
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import mechfight as mf
except ImportError:
    mf = None  # type: ignore[assignment]


class WorldMap:
    """Static arena terrain and 3D prism geometry indexed into a 16.0 m spatial hash grid."""

    def __init__(self, map_dir: str, mech_cfg: Optional[Dict[str, Any]] = None) -> None:
        world_path = os.path.join(map_dir, "v2", "world.json")
        terrain_path = os.path.join(map_dir, "v2", "terrain.i16")
        with open(world_path, "r", encoding="utf-8") as f:
            w = json.load(f)
        with open(terrain_path, "rb") as f:
            raw = f.read()

        b = w.get("bounds", {})
        self.bounds: Dict[str, float] = {
            "minX": float(b.get("minX", -509.31)),
            "maxX": float(b.get("maxX", 551.84)),
            "minY": float(b.get("minY", -330.15)),
            "maxY": float(b.get("maxY", 389.07)),
        }

        t = w.get("terrain", {})
        self.min_x: float = float(t.get("minX", -513.0))
        self.min_y: float = float(t.get("minY", -334.0))
        self.step: float = float(t.get("step", 1.0))
        self.nx: int = int(t.get("nx", 1070))
        self.ny: int = int(t.get("ny", 724))
        self.scale: float = float(t.get("scale", 0.01))

        self.heights = array.array("h")
        self.heights.frombytes(raw)
        if sys.byteorder != "little":
            self.heights.byteswap()

        cfg = mech_cfg or {}
        self.mech_r: float = float(cfg.get("radius", 2.5))
        self.mech_h: float = float(cfg.get("height", 9.0))
        self.step_up: float = float(cfg.get("step_up", 1.2))
        self.max_slope_deg: float = float(cfg.get("max_slope_deg", 50.0))
        self.eye_height: float = float(cfg.get("eye_height", 8.0))
        self.max_slope_tan: float = math.tan(math.radians(self.max_slope_deg))

        self.cell: float = 16.0
        self.prisms: List[Dict[str, Any]] = []
        self.grid: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

        cell = self.cell
        grid = self.grid
        for idx, p in enumerate(w.get("prisms", ())):
            ring = p.get("ring", ())
            if len(ring) < 6:
                continue
            xs = [float(v) for v in ring[0::2]]
            ys = [float(v) for v in ring[1::2]]
            poly: List[Tuple[float, float]] = list(zip(xs, ys))
            holes: List[List[Tuple[float, float]]] = []
            for h in p.get("holes", ()):
                if len(h) >= 6:
                    holes.append(list(zip((float(v) for v in h[0::2]), (float(v) for v in h[1::2]))))
            bbox = (min(xs), min(ys), max(xs), max(ys))
            p_dict: Dict[str, Any] = {
                "id": idx,
                "kind": str(p.get("kind", "building")),
                "base": float(p.get("base", 0.0)),
                "top": float(p.get("top", 0.0)),
                "poly": poly,
                "holes": holes,
                "bbox": bbox,
            }
            self.prisms.append(p_dict)
            gx0 = int(math.floor(bbox[0] / cell))
            gy0 = int(math.floor(bbox[1] / cell))
            gx1 = int(math.floor(bbox[2] / cell))
            gy1 = int(math.floor(bbox[3] / cell))
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    bucket = grid.get((gx, gy))
                    if bucket is None:
                        grid[(gx, gy)] = [p_dict]
                    else:
                        bucket.append(p_dict)

    _CACHE: Dict[str, "WorldMap"] = {}

    @classmethod
    def load(cls, hello: Dict[str, Any]) -> Optional["WorldMap"]:
        """Load WorldMap from a hello message, returning None cleanly if map files are missing."""
        try:
            map_dir = str(hello.get("map", {}).get("dir", "/arena"))
            mech_cfg = hello.get("config", {}).get("mech")
            key = f"{map_dir}:{ (mech_cfg or {}).get('radius', 2.5) }:{ (mech_cfg or {}).get('height', 9.0) }"
            cached = cls._CACHE.get(key)
            if cached is not None:
                return cached
            inst = cls(map_dir, mech_cfg)
            cls._CACHE[key] = inst
            return inst
        except OSError as exc:
            if mf is not None:
                mf.log(f"WorldMap.load failed: {exc}")
            return None

    def ground_z(self, x: float, y: float) -> float:
        """Bilinear interpolation of terrain height in meters at world coordinates (x, y)."""
        fx = (x - self.min_x) / self.step
        fy = (y - self.min_y) / self.step
        max_fx = self.nx - 1.001
        max_fy = self.ny - 1.001
        if fx < 0.0:
            fx = 0.0
        elif fx > max_fx:
            fx = max_fx
        if fy < 0.0:
            fy = 0.0
        elif fy > max_fy:
            fy = max_fy
        i = int(fx)
        j = int(fy)
        tx = fx - i
        ty = fy - j
        nx = self.nx
        heights = self.heights
        row0 = j * nx + i
        row1 = row0 + nx
        h00 = heights[row0]
        h10 = heights[row0 + 1]
        h01 = heights[row1]
        h11 = heights[row1 + 1]
        h = (1.0 - ty) * ((1.0 - tx) * h00 + tx * h10) + ty * ((1.0 - tx) * h01 + tx * h11)
        return h * self.scale

    def ground_slope_tan(self, x: float, y: float) -> float:
        """Finite-difference terrain slope tangent magnitude hypot(dz/dx, dz/dy) using 1 m neighbors."""
        s = self.step
        dzdx = (self.ground_z(x + s, y) - self.ground_z(x - s, y)) / (2.0 * s)
        dzdy = (self.ground_z(x, y + s) - self.ground_z(x, y - s)) / (2.0 * s)
        return math.hypot(dzdx, dzdy)

    @staticmethod
    def _in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
        inside = False
        px1, py1 = poly[-1]
        for px2, py2 in poly:
            if ((py2 > y) != (py1 > y)) and (x < (px1 - px2) * (y - py2) / (py1 - py2) + px2):
                inside = not inside
            px1, py1 = px2, py2
        return inside

    def _in_prism_xy(self, x: float, y: float, p: Dict[str, Any]) -> bool:
        b0, b1, b2, b3 = p["bbox"]
        if x < b0 or x > b2 or y < b1 or y > b3:
            return False
        if not self._in_poly(x, y, p["poly"]):
            return False
        for h in p["holes"]:
            if self._in_poly(x, y, h):
                return False
        return True

    @staticmethod
    def _circle_hits_poly_edges(x: float, y: float, r_sq: float, poly: List[Tuple[float, float]]) -> bool:
        ax, ay = poly[-1]
        for bx, by in poly:
            vx = bx - ax
            vy = by - ay
            wx = x - ax
            wy = y - ay
            len_sq = vx * vx + vy * vy
            if len_sq <= 1e-12:
                dx = wx
                dy = wy
            else:
                t = (wx * vx + wy * vy) / len_sq
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                dx = wx - t * vx
                dy = wy - t * vy
            if dx * dx + dy * dy <= r_sq:
                return True
            ax, ay = bx, by
        return False

    def _circle_hits_prism_xy(self, x: float, y: float, r: float, r_sq: float, p: Dict[str, Any]) -> bool:
        b0, b1, b2, b3 = p["bbox"]
        if x + r < b0 or x - r > b2 or y + r < b1 or y - r > b3:
            return False
        if (
            self._in_prism_xy(x, y, p)
            or (
                r > 0.0
                and (
                    self._in_prism_xy(x + r, y, p)
                    or self._in_prism_xy(x - r, y, p)
                    or self._in_prism_xy(x, y + r, p)
                    or self._in_prism_xy(x, y - r, p)
                )
            )
        ):
            return True
        if r <= 0.0:
            return False
        if self._circle_hits_poly_edges(x, y, r_sq, p["poly"]):
            return True
        for h in p["holes"]:
            if self._circle_hits_poly_edges(x, y, r_sq, h):
                return True
        return False

    def is_walkable(
        self,
        x: float,
        y: float,
        z: Optional[float] = None,
        radius: Optional[float] = None,
        height: Optional[float] = None,
    ) -> bool:
        """Return True if a cylinder of (radius, height) with feet at (x, y, z) is unobstructed and on walkable slope."""
        r = self.mech_r if radius is None else float(radius)
        h = self.mech_h if height is None else float(height)
        b = self.bounds
        if x < b["minX"] + r or x > b["maxX"] - r or y < b["minY"] + r or y > b["maxY"] - r:
            return False

        gz = self.ground_z(x, y)
        fz = gz if z is None else float(z)

        if abs(fz - gz) <= self.step_up and self.ground_slope_tan(x, y) > self.max_slope_tan:
            return False

        z_low = fz + self.step_up
        z_high = fz + h
        r_sq = r * r
        cell = self.cell
        gx0 = int(math.floor((x - r) / cell))
        gy0 = int(math.floor((y - r) / cell))
        gx1 = int(math.floor((x + r) / cell))
        gy1 = int(math.floor((y + r) / cell))

        seen_ids = set()
        grid = self.grid
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                bucket = grid.get((gx, gy))
                if not bucket:
                    continue
                for p in bucket:
                    pid = p["id"]
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    # Standing on top of prism (z + step_up >= top) or walking under bridge (z + h <= base)
                    if z_low >= p["top"] or z_high <= p["base"]:
                        continue
                    if self._circle_hits_prism_xy(x, y, r, r_sq, p):
                        return False
        return True

    def support_z(self, x: float, y: float, cur_z: float) -> float:
        """Compute floor height under (x, y) considering terrain and any flat prism tops <= cur_z + step_up."""
        floor_z = self.ground_z(x, y)
        z_reach = cur_z + self.step_up
        bucket = self.grid.get((int(math.floor(x / self.cell)), int(math.floor(y / self.cell))))
        if bucket:
            for p in bucket:
                top = p["top"]
                if floor_z < top <= z_reach and self._in_prism_xy(x, y, p):
                    floor_z = top
        return floor_z

    _support_z = support_z

    def path_clear(
        self,
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
        radius: Optional[float] = None,
        height: Optional[float] = None,
        step: float = 2.5,
    ) -> bool:
        """Check whether a mech can walk in a straight 2D line from a to b without hitting walls or steep slopes."""
        ax, ay, az = float(a[0]), float(a[1]), float(a[2])
        bx, by = float(b[0]), float(b[1])
        dx = bx - ax
        dy = by - ay
        dist = math.hypot(dx, dy)
        step_sz = max(0.5, float(step))
        steps = max(1, int(math.ceil(dist / step_sz)))
        inv = 1.0 / steps

        cur_z = max(az, self.ground_z(ax, ay))
        for k in range(1, steps + 1):
            t = k * inv
            px = ax + dx * t
            py = ay + dy * t
            cur_z = self.support_z(px, py, cur_z)
            if not self.is_walkable(px, py, cur_z, radius=radius, height=height):
                return False
        return True

    def _blocks_ray(self, p, x, y, z, dx, dy, dz):
        # Clip the segment to the prism's bounding box, including its vertical slab.
        lo, hi = 0.0, 1.0
        xmin, ymin, xmax, ymax = p["bbox"]
        for start, delta, low, high in ((x, dx, xmin, xmax), (y, dy, ymin, ymax),
                                        (z, dz, p["base"], p["top"])):
            if delta == 0:
                if start < low or start > high:
                    return False
                continue
            a, b = (low - start) / delta, (high - start) / delta
            lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
            if lo > hi:
                return False
        mid = (lo + hi) * 0.5
        if self._in_prism_xy(x + mid * dx, y + mid * dy, p):
            return True
        # A ray can enter and exit the footprint on the same side of the midpoint.
        for ring in (p["poly"], *p["holes"]):
            ax, ay = ring[-1]
            for bx, by in ring:
                ex, ey = bx - ax, by - ay
                det = dx * ey - dy * ex
                if det:
                    rx, ry = ax - x, ay - y
                    t = (rx * ey - ry * ex) / det
                    u = (rx * dy - ry * dx) / det
                    if lo <= t <= hi and 0 <= u <= 1:
                        return True
                ax, ay = bx, by
        return False

    def has_los(
        self,
        a_xyz: Tuple[float, float, float],
        b_xyz: Tuple[float, float, float],
        step: float = 3.0,
    ) -> bool:
        """Fast 3D line-of-sight check between a_xyz and b_xyz against terrain and building/stadium/bridge prisms."""
        ax, ay, az = float(a_xyz[0]), float(a_xyz[1]), float(a_xyz[2])
        bx, by, bz = float(b_xyz[0]), float(b_xyz[1]), float(b_xyz[2])
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist_3d <= 1e-4:
            return True

        eff_step = min(float(step), 3.0) if step > 0.0 else 3.0
        steps = max(2, int(math.ceil(dist_3d / eff_step)))
        inv = 1.0 / steps
        sx = dx * inv
        sy = dy * inv
        sz = dz * inv

        cell = self.cell
        grid = self.grid
        # Visit every grid cell crossed in each x-column, even at a corner.
        checked = set()
        for gx in range(math.floor(min(ax, bx) / cell), math.floor(max(ax, bx) / cell) + 1):
            if dx:
                a = (gx * cell - ax) / dx
                b = ((gx + 1) * cell - ax) / dx
                y0 = ay + dy * max(0.0, min(a, b))
                y1 = ay + dy * min(1.0, max(a, b))
            else:
                y0, y1 = ay, by
            for gy in range(math.floor(min(y0, y1) / cell), math.floor(max(y0, y1) / cell) + 1):
                for p in grid.get((gx, gy), ()):
                    if p["id"] in checked:
                        continue
                    checked.add(p["id"])
                    if self._blocks_ray(p, ax, ay, az, dx, dy, dz):
                        return False
        x, y, z = ax, ay, az
        for _ in range(1, steps):
            x += sx
            y += sy
            z += sz
            if z < self.ground_z(x, y):
                return False
        return True
