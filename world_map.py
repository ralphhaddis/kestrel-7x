"""Terrain, building prism spatial grid, walkability, and line-of-sight queries for mechfight/1.

Standard library only (Python 3.9+). Loads ``v2/world.json`` and ``v2/terrain.i16`` from
``hello["map"]["dir"]`` and indexes building, stadium, and bridge prisms into a 16.0 m 2D
spatial hash grid for sub-millisecond ``ground_z``, ``is_walkable``, ``path_clear``, and
``has_los`` queries.
"""
from __future__ import annotations

import array
import heapq
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
        self.max_ground_z: float = (max(self.heights) * self.scale) if self.heights else 0.0

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

        # Precompute coarse 5.0 m 2D walkability grid for fast A* routing around campus buildings
        self.nav_step: float = 5.0
        self.nav_min_x: float = self.bounds["minX"]
        self.nav_min_y: float = self.bounds["minY"]
        self.nav_nx: int = max(1, int(math.ceil((self.bounds["maxX"] - self.nav_min_x) / self.nav_step)))
        self.nav_ny: int = max(1, int(math.ceil((self.bounds["maxY"] - self.nav_min_y) / self.nav_step)))
        self.nav_grid: bytearray = self._build_nav_grid()
        self.corner_nodes: List[Tuple[float, float]] = []
        self.corner_grid: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        self._build_corner_nodes()
        self.canopy_nodes: List[Tuple[float, float]] = []
        self.canopy_grid: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        self._build_canopy_nodes(w)
        self.roof_nodes: List[Tuple[float, float, float, float, float, float]] = []
        self._build_roof_nodes()

    def _build_roof_nodes(self) -> None:
        """Precompute climbable flat building rooftops (8..36 m above ground) with paired ground launch points.
        Each entry is (rx, ry, rtop, lx, ly, lgz) where (rx, ry, rtop) is a walkable spot on the roof
        and (lx, ly, lgz) is a walkable ground spot just outside the roof edge to jump+jet from."""
        for p in self.prisms:
            if p["kind"] == "bridge":
                continue
            poly = p["poly"]
            n = len(poly)
            if n < 3:
                continue
            top = p["top"]
            b0, b1, b2, b3 = p["bbox"]
            # Need a roof wide enough to stand and peek on (>= 10 m)
            if (b2 - b0) < 10.0 or (b3 - b1) < 10.0:
                continue
            # Compute polygon orientation for outward edge normals
            area2 = 0.0
            px1, py1 = poly[-1]
            for px2, py2 in poly:
                area2 += px1 * py2 - px2 * py1
                px1, py1 = px2, py2
            if abs(area2) < 60.0:
                continue
            ccw = 1.0 if area2 > 0.0 else -1.0
            # Sample candidate roof points near edges (4.5 m inset) so we can easily climb and peek over the edge
            best_pair = None
            for i in range(n):
                ax, ay = poly[i - 1]
                bx, by = poly[i]
                dx, dy = bx - ax, by - ay
                elen = math.hypot(dx, dy)
                if elen < 8.0:
                    continue
                mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
                # Outward normal
                nx, ny = (dy / elen) * ccw, -(dx / elen) * ccw
                lx = mx + nx * (self.mech_r + 1.8)
                ly = my + ny * (self.mech_r + 1.8)
                rx = mx - nx * (self.mech_r + 2.2)
                ry = my - ny * (self.mech_r + 2.2)
                if not self._in_prism_xy(rx, ry, p):
                    continue
                lgz = self.ground_z(lx, ly)
                climb_h = top - lgz
                if not (7.0 <= climb_h <= 34.0):
                    continue
                if not self.is_walkable(lx, ly, lgz, _gz=lgz):
                    continue
                if not self.is_walkable(rx, ry, top):
                    continue
                best_pair = (rx, ry, top, lx, ly, lgz)
                break
            if best_pair is not None:
                self.roof_nodes.append(best_pair)

    def nearby_roofs(
        self,
        x: float,
        y: float,
        cx: float,
        cy: float,
        max_center_r: float,
        max_dist: float = 95.0,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """Return climbable roof nodes within max_dist of (x, y) and inside max_center_r of (cx, cy), sorted by distance."""
        out: List[Tuple[float, Tuple[float, float, float, float, float, float]]] = []
        max_d2 = max_dist * max_dist
        max_c2 = max_center_r * max_center_r
        for node in self.roof_nodes:
            rx, ry, rtop, lx, ly, lgz = node
            if (rx - cx) ** 2 + (ry - cy) ** 2 > max_c2:
                continue
            d2 = (lx - x) ** 2 + (ly - y) ** 2
            if d2 <= max_d2:
                out.append((d2, node))
        out.sort(key=lambda item: item[0])
        return [node for _, node in out]

    def _build_canopy_nodes(self, w: Dict[str, Any]) -> None:
        """Precompute walkable tree-canopy concealment positions (crownR >= 2.2 m) into a 16 m grid."""
        t_obj = w.get("trees", {})
        t_fields = t_obj.get("fields", ())
        t_data = t_obj.get("data", ())
        nf = len(t_fields)
        if nf == 0 or not t_data:
            return
        f_idx = {name: i for i, name in enumerate(t_fields)}
        if not all(k in f_idx for k in ("x", "y", "z", "height", "crownR", "crownBase", "trunkR")):
            return
        ix, iy, iz = f_idx["x"], f_idx["y"], f_idx["z"]
        ih, icr, icb, itr = f_idx["height"], f_idx["crownR"], f_idx["crownBase"], f_idx["trunkR"]
        icx = f_idx.get("crownX", ix)
        icy = f_idx.get("crownY", iy)
        cell = self.cell
        nodes = self.canopy_nodes
        cgrid = self.canopy_grid
        seen_cells = set()
        eye_h = self.eye_height
        mech_r = self.mech_r

        for k in range(0, len(t_data), nf):
            cr = float(t_data[k + icr])
            if cr < 2.2:
                continue
            tx = float(t_data[k + ix])
            ty = float(t_data[k + iy])
            tz = float(t_data[k + iz])
            th = float(t_data[k + ih])
            cb = float(t_data[k + icb])
            tr = float(t_data[k + itr])
            cx_t = float(t_data[k + icx])
            cy_t = float(t_data[k + icy])
            # Eye height at ground_z + eye_h must sit inside or near the crown vertical band [tz+cb, tz+th]
            if not (tz + cb - 1.5 <= tz + eye_h <= tz + th + 1.0):
                continue
            # Tree trunks block vision only, not movement (kit/protocol.md): place at crown center (cx_t, cy_t)
            # for maximum 360-deg canopy chord thickness (~5.0 m >= 3.96 m vision block threshold).
            px = cx_t
            py = cy_t
            dkey = (int(math.floor(px / 6.0)), int(math.floor(py / 6.0)))
            if dkey in seen_cells:
                continue
            gz = self.ground_z(px, py)
            if not self.is_walkable(px, py, gz, _gz=gz):
                continue
            seen_cells.add(dkey)
            pt = (px, py)
            nodes.append(pt)
            ckey = (int(math.floor(px / cell)), int(math.floor(py / cell)))
            bucket = cgrid.get(ckey)
            if bucket is None:
                cgrid[ckey] = [pt]
            else:
                bucket.append(pt)

    def nearby_canopies(
        self,
        x: float,
        y: float,
        max_r: float = 28.0,
        limit: int = 8,
    ) -> List[Tuple[float, float]]:
        """Return up to `limit` precomputed tree-canopy concealment positions within max_r m."""
        cell = self.cell
        gx0 = int(math.floor((x - max_r) / cell))
        gy0 = int(math.floor((y - max_r) / cell))
        gx1 = int(math.floor((x + max_r) / cell))
        gy1 = int(math.floor((y + max_r) / cell))
        max_r2 = max_r * max_r
        cgrid = self.canopy_grid
        found: List[Tuple[float, float, float]] = []
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                bucket = cgrid.get((gx, gy))
                if not bucket:
                    continue
                for px, py in bucket:
                    dx = px - x
                    dy = py - y
                    d2 = dx * dx + dy * dy
                    if d2 <= max_r2:
                        found.append((d2, px, py))
        found.sort(key=lambda item: item[0])
        return [(px, py) for _, px, py in found[:limit]]

    def _build_corner_nodes(self) -> None:
        """Precompute walkable exterior building corners offset by mech_r + 1.5 m into a 16 m grid."""
        offset = self.mech_r + 1.5
        cell = self.cell
        nodes = self.corner_nodes
        cgrid = self.corner_grid
        seen_cells = set()

        for p in self.prisms:
            if p["kind"] == "bridge":
                continue
            poly = p["poly"]
            n = len(poly)
            if n < 3:
                continue
            area2 = 0.0
            px1, py1 = poly[-1]
            for px2, py2 in poly:
                area2 += px1 * py2 - px2 * py1
                px1, py1 = px2, py2
            if abs(area2) < 1e-3:
                continue
            ccw = 1.0 if area2 > 0.0 else -1.0
            top = p["top"]

            for i in range(n):
                ax, ay = poly[i - 1]
                bx, by = poly[i]
                cx, cy = poly[(i + 1) % n]
                dx1, dy1 = bx - ax, by - ay
                dx2, dy2 = cx - bx, cy - by
                l1 = math.hypot(dx1, dy1)
                l2 = math.hypot(dx2, dy2)
                if l1 < 1e-3 or l2 < 1e-3:
                    continue
                e1x, e1y = dx1 / l1, dy1 / l1
                e2x, e2y = dx2 / l2, dy2 / l2
                cross = (e1x * e2y - e1y * e2x) * ccw
                if cross < 0.25:
                    continue
                n1x, n1y = e1y * ccw, -e1x * ccw
                n2x, n2y = e2y * ccw, -e2x * ccw
                bis_x, bis_y = n1x + n2x, n1y + n2y
                bnorm = math.hypot(bis_x, bis_y)
                if bnorm < 1e-6:
                    continue
                px = bx + (bis_x / bnorm) * offset
                py = by + (bis_y / bnorm) * offset
                dkey = (int(math.floor(px / 4.0)), int(math.floor(py / 4.0)))
                if dkey in seen_cells:
                    continue
                gz = self.ground_z(px, py)
                if top < gz + 6.0:
                    continue
                if not self.is_walkable(px, py, gz, _gz=gz):
                    continue
                seen_cells.add(dkey)
                pt = (px, py)
                nodes.append(pt)
                ckey = (int(math.floor(px / cell)), int(math.floor(py / cell)))
                bucket = cgrid.get(ckey)
                if bucket is None:
                    cgrid[ckey] = [pt]
                else:
                    bucket.append(pt)

    def nearby_corners(
        self,
        x: float,
        y: float,
        max_r: float = 42.0,
        limit: int = 20,
    ) -> List[Tuple[float, float]]:
        """Return up to `limit` precomputed building corner nodes within [2.0, max_r] m, sorted by distance."""
        cell = self.cell
        gx0 = int(math.floor((x - max_r) / cell))
        gy0 = int(math.floor((y - max_r) / cell))
        gx1 = int(math.floor((x + max_r) / cell))
        gy1 = int(math.floor((y + max_r) / cell))
        max_r2 = max_r * max_r
        cgrid = self.corner_grid
        found: List[Tuple[float, float, float]] = []
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                bucket = cgrid.get((gx, gy))
                if not bucket:
                    continue
                for px, py in bucket:
                    dx = px - x
                    dy = py - y
                    d2 = dx * dx + dy * dy
                    if 4.0 <= d2 <= max_r2:
                        found.append((d2, px, py))
        found.sort(key=lambda item: item[0])
        return [(px, py) for _, px, py in found[:limit]]

    def _build_nav_grid(self) -> bytearray:
        """Build a 1-byte-per-cell walkability grid (1 = walkable, 0 = blocked) at startup."""
        nx = self.nav_nx
        ny = self.nav_ny
        step = self.nav_step
        x0 = self.nav_min_x + 0.5 * step
        y0 = self.nav_min_y + 0.5 * step
        nav = bytearray(nx * ny)
        is_walk = self.is_walkable
        for j in range(ny):
            wy = y0 + j * step
            row = j * nx
            for i in range(nx):
                if is_walk(x0 + i * step, wy):
                    nav[row + i] = 1
        return nav

    def _world_to_nav(self, x: float, y: float) -> Tuple[int, int]:
        i = int((x - self.nav_min_x) / self.nav_step)
        j = int((y - self.nav_min_y) / self.nav_step)
        if i < 0:
            i = 0
        elif i >= self.nav_nx:
            i = self.nav_nx - 1
        if j < 0:
            j = 0
        elif j >= self.nav_ny:
            j = self.nav_ny - 1
        return i, j

    def _nav_to_world(self, i: int, j: int) -> Tuple[float, float]:
        return (self.nav_min_x + (i + 0.5) * self.nav_step, self.nav_min_y + (j + 0.5) * self.nav_step)

    def _nearest_walkable_cell(self, ci: int, cj: int, max_r: int = 8) -> Optional[Tuple[int, int]]:
        nx, ny, nav = self.nav_nx, self.nav_ny, self.nav_grid
        if nav[cj * nx + ci]:
            return ci, cj
        for r in range(1, max_r + 1):
            best = None
            best_d2 = 10**9
            for di in range(-r, r + 1):
                for dj in (-r, r) if abs(di) < r else range(-r, r + 1):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < nx and 0 <= nj < ny and nav[nj * nx + ni]:
                        d2 = di * di + dj * dj
                        if d2 < best_d2:
                            best_d2 = d2
                            best = (ni, nj)
            if best is not None:
                return best
        return None

    def find_path(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        max_expansions: int = 2000,
    ) -> Optional[List[Tuple[float, float]]]:
        """8-connected A* on the precomputed 5.0 m nav grid; returns world waypoints or None."""
        si, sj = self._world_to_nav(start_xy[0], start_xy[1])
        gi, gj = self._world_to_nav(goal_xy[0], goal_xy[1])
        start = self._nearest_walkable_cell(si, sj, max_r=6)
        goal = self._nearest_walkable_cell(gi, gj, max_r=10)
        if start is None or goal is None:
            return None
        if start == goal:
            return [goal_xy]

        nx, ny, nav = self.nav_nx, self.nav_ny, self.nav_grid
        si, sj = start
        gi, gj = goal
        s_idx = sj * nx + si
        g_idx = gj * nx + gi

        # Octile heuristic * 10 (integer costs: 10 orthogonal, 14 diagonal)
        dx0 = abs(gi - si)
        dy0 = abs(gj - sj)
        h0 = 10 * (dx0 + dy0) - 6 * (dx0 if dx0 < dy0 else dy0)

        g_score: Dict[int, int] = {s_idx: 0}
        came_from: Dict[int, int] = {}
        heap: List[Tuple[int, int, int]] = [(h0, 0, s_idx)]
        closed = set()
        expansions = 0

        neighbors = (
            (1, 0, 10), (-1, 0, 10), (0, 1, 10), (0, -1, 10),
            (1, 1, 14), (1, -1, 14), (-1, 1, 14), (-1, -1, 14),
        )
        while heap and expansions < max_expansions:
            _, cur_g, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            if cur == g_idx:
                path_cells: List[Tuple[int, int]] = []
                node = cur
                while node != s_idx:
                    path_cells.append((node % nx, node // nx))
                    node = came_from[node]
                path_cells.reverse()
                pts = [self._nav_to_world(ci, cj) for ci, cj in path_cells]
                if nav[gj * nx + gi]:
                    pts[-1] = goal_xy if self.is_walkable(goal_xy[0], goal_xy[1]) else pts[-1]
                return pts
            closed.add(cur)
            expansions += 1
            ci = cur % nx
            cj = cur // nx
            for di, dj, step_cost in neighbors:
                ni = ci + di
                nj = cj + dj
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                    continue
                n_idx = nj * nx + ni
                if not nav[n_idx] or n_idx in closed:
                    continue
                # Prevent diagonal corner-cutting through building corners
                if di != 0 and dj != 0:
                    if not (nav[cj * nx + ni] and nav[nj * nx + ci]):
                        continue
                ng = cur_g + step_cost
                old_g = g_score.get(n_idx)
                if old_g is None or ng < old_g:
                    g_score[n_idx] = ng
                    came_from[n_idx] = cur
                    adx = abs(gi - ni)
                    ady = abs(gj - nj)
                    h = 10 * (adx + ady) - 6 * (adx if adx < ady else ady)
                    # Weighted A* (1.5x heuristic) finds near-optimal paths in ~10x fewer expansions
                    heapq.heappush(heap, (ng + (h * 3) // 2, ng, n_idx))
        return None

    def nav_line_clear(self, a_xy: Tuple[float, float], b_xy: Tuple[float, float]) -> bool:
        """Fast O(steps) line-of-travel check directly on the precomputed 5.0 m nav_grid."""
        ax, ay = float(a_xy[0]), float(a_xy[1])
        bx, by = float(b_xy[0]), float(b_xy[1])
        dx = bx - ax
        dy = by - ay
        dist = math.hypot(dx, dy)
        if dist <= 1e-3:
            return True
        steps = max(1, int(math.ceil(dist / (self.nav_step * 0.6))))
        inv = 1.0 / steps
        nx, ny, nav = self.nav_nx, self.nav_ny, self.nav_grid
        min_x, min_y, inv_step = self.nav_min_x, self.nav_min_y, 1.0 / self.nav_step
        for k in range(1, steps + 1):
            t = k * inv
            i = int((ax + dx * t - min_x) * inv_step)
            j = int((ay + dy * t - min_y) * inv_step)
            if i < 0 or i >= nx or j < 0 or j >= ny or not nav[j * nx + i]:
                return False
        return True

    def next_waypoint(
        self,
        start_xyz: Tuple[float, float, float],
        path: List[Tuple[float, float]],
        radius: Optional[float] = None,
        height: Optional[float] = None,
    ) -> Tuple[float, float]:
        """String-pull along an A* path using fast nav_grid ray checks."""
        if not path:
            return (start_xyz[0], start_xyz[1])
        sxy = (start_xyz[0], start_xyz[1])
        limit = min(len(path), 16)
        for idx in range(limit - 1, 0, -1):
            if self.nav_line_clear(sxy, path[idx]):
                return path[idx]
        return path[0]

    _CACHE: Dict[str, "WorldMap"] = {}

    @classmethod
    def load(cls, hello: Dict[str, Any]) -> Optional["WorldMap"]:
        """Load WorldMap from a hello message, returning None cleanly if map files are missing."""
        try:
            map_dir = str(hello.get("map", {}).get("dir", "/arena"))
            mech_cfg = hello.get("config", {}).get("mech") or {}
            key = (
                f"{map_dir}:{mech_cfg.get('radius', 2.5)}:{mech_cfg.get('height', 9.0)}"
                f":{mech_cfg.get('step_up', 1.2)}:{mech_cfg.get('max_slope_deg', 50.0)}"
                f":{mech_cfg.get('eye_height', 8.0)}"
            )
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
        if self._in_prism_xy(x, y, p):
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
        _gz: Optional[float] = None,
    ) -> bool:
        """Return True if a cylinder of (radius, height) with feet at (x, y, z) is unobstructed and on walkable slope."""
        r = self.mech_r if radius is None else float(radius)
        h = self.mech_h if height is None else float(height)
        b = self.bounds
        if x < b["minX"] + r or x > b["maxX"] - r or y < b["minY"] + r or y > b["maxY"] - r:
            return False

        gz = self.ground_z(x, y) if _gz is None else _gz
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

        grid = self.grid
        if gx0 == gx1 and gy0 == gy1:
            bucket = grid.get((gx0, gy0))
            if bucket:
                for p in bucket:
                    if z_low >= p["top"] or z_high <= p["base"]:
                        continue
                    if self._circle_hits_prism_xy(x, y, r, r_sq, p):
                        return False
            return True

        seen_ids = set()
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

    def _support_and_ground(self, x: float, y: float, cur_z: float) -> Tuple[float, float]:
        gz = self.ground_z(x, y)
        floor_z = gz
        z_reach = cur_z + self.step_up
        bucket = self.grid.get((int(math.floor(x / self.cell)), int(math.floor(y / self.cell))))
        if bucket:
            for p in bucket:
                top = p["top"]
                if floor_z < top <= z_reach and self._in_prism_xy(x, y, p):
                    floor_z = top
        return floor_z, gz

    def support_z(self, x: float, y: float, cur_z: float) -> float:
        """Compute floor height under (x, y) considering terrain and any flat prism tops <= cur_z + step_up."""
        return self._support_and_ground(x, y, cur_z)[0]

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
            cur_z, gz = self._support_and_ground(px, py, cur_z)
            if not self.is_walkable(px, py, cur_z, radius=radius, height=height, _gz=gz):
                return False
        return True

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
        steps = max(4, int(math.ceil(dist_3d / eff_step)))
        inv = 1.0 / steps
        sx = dx * inv
        sy = dy * inv
        sz = dz * inv

        cell = self.cell
        grid = self.grid
        max_gz = self.max_ground_z
        x, y, z = ax, ay, az
        last_cell: Optional[Tuple[int, int]] = None
        bucket = None
        for _ in range(1, steps):
            x += sx
            y += sy
            z += sz
            c_key = (int(x // cell), int(y // cell))
            if c_key != last_cell:
                last_cell = c_key
                bucket = grid.get(c_key)
            if bucket:
                for p in bucket:
                    if p["base"] <= z <= p["top"] and self._in_prism_xy(x, y, p):
                        return False
            if z <= max_gz and z < self.ground_z(x, y):
                return False
        return True
