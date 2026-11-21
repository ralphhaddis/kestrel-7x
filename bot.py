"""survivor-v1: a minimal mechfight/1 bot (Python 3.9+, standard library only).

The kit wrapper (mechfight.py) owns the I/O loop: hello/ready, newest-observation-only, tick echo,
and re-sending the last move/aim/boost if act() raises. This file only defines the Bot.

act() is split into small, replaceable steps:
    update_memory -> choose_pick -> choose_take -> movement -> combat
Each returns a partial action dict (or updates state) so any one can be rewritten on its own.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import mechfight as mf
import world_map as wm

FLAGS = {
    "loot": True,    # detour to and take weapon pickups (kinds in LOOT_PREF)
    "fight": True,   # aim at and shoot enemies
    "lead": True,    # lead moving targets by projectile flight time (gun)
    "cover": True,   # seek cover when inside the zone and threatened
    "heal": True,    # Experiment 1: HEAL_COVER ceasefire when low HP
    "kite": True,    # Experiment 2: perimeter settle + open-field kiting/strafing
    "cubes": True,   # Experiment 3: pick up nearby power cubes
    "sound": True,   # Experiment 3: face recent "near" sounds in idle_aim
    "debug": True,   # attach debug drawings (never affects the match)
}

# Upgrade ids in order of preference; anything not listed is only taken as the server's choice 0.
UPGRADE_PREF = ["armor", "plating", "nanites", "gun_rate", "gun_damage", "gyros", "gun_mag",
                "gun_velocity", "servos", "sensors", "capacitor", "reactor", "jets"]
# Weapon kinds worth taking. No mortar yet: it needs its own ballistic aim.
LOOT_PREF = ["laser", "lightning"]

TAU = 2.0 * math.pi
THREAT_MEMORY_S = 2.0      # face the last attacker's bearing this long after being hit
AGGRO_MEMORY_S = 3.0       # shoot an NPC only if it hit us this recently
NPC_PENALTY_M = 30.0       # target choice: an NPC counts as this much farther than a mech
LOOT_DETOUR_M = 40.0       # ignore pickups farther than this
STUCK_SPEED = 1.0          # m/s: slower than this while walking counts as stuck
STUCK_AFTER_S = 1.0
UNSTICK_S = 1.5
LIGHTNING_ENERGY_RESERVE = 20.0  # keep this much energy for jets
COVER_HOLD_S = 1.2         # hysteresis commit window so we don't jitter between two corners
COVER_REACH_M = 2.5        # considered at cover point within this distance


def wrap(a: float) -> float:
    """Angle to [-pi, pi)."""
    return (a + math.pi) % TAU - math.pi


class Bot:
    name = "survivor-v1"

    def __init__(self, hello: mf.Hello):
        cfg = hello["config"]
        self.cfg = cfg
        self.rng = random.Random(hello["seed"])
        self.memory_s = cfg["vision"].get("memory_s", 5.0)
        self.mech_h = cfg["mech"]["height"]
        self.mech_r = cfg["mech"]["radius"]
        self.size_per_cube = cfg["cubes"].get("size_per_cube", 0.0)
        self.npc_h = cfg["npc"]["height"]
        self.npc_r = cfg["npc"]["radius"]
        self.reach = cfg["loot"]["pickup_radius"]

        self.world = wm.WorldMap.load(hello)
        self.seen = {}          # actor id -> (t, pos, vel), last time we saw it
        self.aggressors = {}    # attacker id -> t it last damaged us
        self.threat = None      # (t, bearing) of the last non-zone damage
        self.last_sound: Optional[Tuple[float, float]] = None  # (t, bearing) of recent danger sound
        self.healing = False
        self.last_damaged_t = -99.0
        self.late = False
        self.last_t = None
        self.dt = 0.1
        self.loot_goal = None   # (x, y) of a pickup to walk to, set by choose_take
        self.cover_goal: Optional[Tuple[float, float]] = None
        self.cover_until = -1.0
        self.move_yaw = None    # yaw we are walking along, None when holding still
        self.stuck_s = 0.0
        self.unstick_until = -1.0
        self.unstick_side = 1
        mf.log(f"{self.name}: slot {hello['slot']} seed {hello['seed']}")

    # ------------------------------------------------------------------ main step
    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        self.update_memory(obs)
        act: dict = {"tick": obs["tick"]}
        act.update(self.choose_pick(obs))
        act.update(self.choose_take(obs, me))
        act.update(self.movement(obs, me))
        act.update(self.combat(obs, me))
        return act

    # ------------------------------------------------------------------ memory
    def update_memory(self, obs: mf.Observation) -> None:
        t = obs["t"]
        if self.last_t is not None:
            self.dt = max(1e-3, t - self.last_t)
        self.last_t = t
        for m in obs["mechs"]:
            self.seen[m["id"]] = (t, m["pos"], m["vel"])
        for e in obs["events"]:
            if e.get("type") == "damaged":
                self.last_damaged_t = t
                if e.get("by") is not None:
                    self.aggressors[e["by"]] = t
                    if e.get("bearing") is not None:
                        self.threat = (t, e["bearing"])
        cutoff = t - self.memory_s
        self.seen = {k: v for k, v in self.seen.items() if v[0] >= cutoff}
        self.aggressors = {k: v for k, v in self.aggressors.items() if v >= t - AGGRO_MEMORY_S}

        me = obs["self"]
        if not FLAGS.get("heal", True) or not me["in_zone"]:
            self.healing = False
        elif me["hp"] < 0.65 * me["max_hp"]:
            self.healing = True
        elif me["hp"] >= 0.85 * me["max_hp"]:
            self.healing = False

        if FLAGS.get("sound", True):
            best_prio = 0
            best_bearing: Optional[float] = None
            for s in obs.get("sounds", ()):
                k = s.get("kind")
                d = s.get("dist")
                b = s.get("bearing")
                if b is None:
                    continue
                if k in ("gun", "laser", "laser_charge", "lightning", "mortar"):
                    prio = 3 if d == "near" else (1 if d == "mid" else 0)
                elif k in ("boost", "step", "jet"):
                    prio = 2 if d == "near" else 0
                else:
                    prio = 0
                if prio > best_prio:
                    best_prio = prio
                    best_bearing = float(b)
            if best_bearing is not None:
                self.last_sound = (t, best_bearing)

    # ------------------------------------------------------------------ upgrades
    def choose_pick(self, obs: mf.Observation) -> dict:
        offer = obs.get("offer")
        if not offer:
            return {}
        ids = [c["id"] for c in offer["choices"]]
        for want in UPGRADE_PREF:
            if want in ids:
                return {"pick": ids.index(want)}
        return {"pick": 0}

    # ------------------------------------------------------------------ loot
    def choose_take(self, obs: mf.Observation, me) -> dict:
        """Pick a pickup to walk to (self.loot_goal) and `take` it once in reach. Empty slot only."""
        self.loot_goal = None
        if not FLAGS.get("loot") or None not in me["weapons"]:
            return {}
        held = {w["kind"] for w in me["weapons"] if w}
        x, y, z = me["pos"]
        mr, mh = me["stats"].get("radius"), me["stats"].get("height")
        zone = obs["zone"]
        cx, cy = zone["next_center"]
        inner = zone["next_radius"] * 0.8
        best, best_d = None, LOOT_DETOUR_M
        for p in obs["pickups"]:
            if p["kind"] not in LOOT_PREF or p["kind"] in held:
                continue
            if abs(p["pos"][2] - z) > 4.0:  # take works within 4 m vertically; roofs need map awareness
                continue
            if math.dist((x, y), p["pos"][:2]) >= self.reach - 0.25 and math.hypot(p["pos"][0] - cx, p["pos"][1] - cy) > inner:  # don't pull back out of the next circle
                continue
            d = math.hypot(p["pos"][0] - x, p["pos"][1] - y)
            if d >= best_d:
                continue
            if self.world is not None and d >= self.reach - 0.25 and not self.world.path_clear(me["pos"], p["pos"], radius=mr, height=mh):
                continue
            best, best_d = p, d
        if best is None:
            return {}
        self.loot_goal = (best["pos"][0], best["pos"][1])
        return {"take": best["id"]} if best_d < self.reach - 0.25 else {}

    def choose_cube(self, obs: mf.Observation, me, cx: float, cy: float, inner: float) -> Optional[Tuple[float, float]]:
        """Pick a nearby visible power cube inside `inner` with a clear walking path."""
        if not FLAGS.get("cubes", True):
            return None
        cubes = obs.get("cubes", ())
        if not cubes:
            return None
        x, y, z = me["pos"]
        mr, mh = me["stats"].get("radius"), me["stats"].get("height")
        best_pos: Optional[Tuple[float, float]] = None
        best_d = 25.0
        for c in cubes:
            cp = c["pos"]
            if abs(cp[2] - z) > 4.0:
                continue
            if inner > 3.0 and math.hypot(cp[0] - cx, cp[1] - cy) > inner:
                continue
            d = math.hypot(cp[0] - x, cp[1] - y)
            if d > best_d:
                continue
            if self.world is not None and not self.world.path_clear(me["pos"], cp, radius=mr, height=mh):
                continue
            best_d = d
            best_pos = (cp[0], cp[1])
        return best_pos

    # ------------------------------------------------------------------ cover & movement
    def known_threats(self, obs: mf.Observation, me) -> List[Tuple[float, float, float]]:
        """Known enemy eye/muzzle 3D positions from visible/remembered mechs or recent damage bearing."""
        by_id = {}
        for m in obs["mechs"]:
            if m["kind"] == "mech" or m["id"] in self.aggressors:
                p = m["pos"]
                by_id[m["id"]] = (p[0], p[1], p[2] + 8.0)
        for aid, (_, p, _) in self.seen.items():
            if aid not in by_id and (aid < 16 or aid in self.aggressors):
                by_id[aid] = (p[0], p[1], p[2] + 8.0)
        threats = list(by_id.values())
        if not threats and self.threat is not None and obs["t"] - self.threat[0] <= THREAT_MEMORY_S:
            x, y, z = me["pos"]
            b = self.threat[1]
            tx = x + 60.0 * math.cos(b)
            ty = y + 60.0 * math.sin(b)
            tz = self.world.ground_z(tx, ty) if self.world is not None else z
            threats.append((tx, ty, tz + 8.0))
        return threats

    def choose_cover(self, obs: mf.Observation, me, cx: float, cy: float, inner: float) -> Optional[Tuple[float, float]]:
        """Pick a nearby cover position inside `inner` that breaks LOS to known threats."""
        if not (FLAGS.get("cover") and self.world is not None and me["in_zone"]):
            self.cover_goal = None
            return None
        x, y, z = me["pos"]
        if math.hypot(cx - x, cy - y) > inner:
            self.cover_goal = None
            return None
        threats = self.known_threats(obs, me)
        if not threats:
            self.cover_goal = None
            return None

        t = obs["t"]
        st = me["stats"]
        eye_h = st.get("eye_height", 8.0)
        mr, mh = st.get("radius"), st.get("height")
        if t < self.cover_until:
            if self.cover_goal is None:
                return None
            gx, gy = self.cover_goal
            if math.hypot(gx - cx, gy - cy) <= inner:
                gz = self.world.ground_z(gx, gy)
                goal_eye = (gx, gy, gz + eye_h)
                if any(not self.world.has_los(goal_eye, t_eye) for t_eye in threats) and self.world.path_clear(me["pos"], (gx, gy, gz), radius=mr, height=mh):
                    return self.cover_goal

        base_angle = math.atan2(cy - y, cx - x)
        candidates: List[Tuple[float, float]] = [(x, y)]
        for r in (15.0, 30.0, 45.0):
            for k in range(10):
                ang = base_angle + TAU * k / 10.0
                candidates.append((x + r * math.cos(ang), y + r * math.sin(ang)))

        best_pos: Optional[Tuple[float, float]] = None
        best_score = -math.inf
        for px, py in candidates:
            d_center = math.hypot(px - cx, py - cy)
            if inner > 3.0 and d_center > inner:
                continue
            pz = self.world.ground_z(px, py)
            cand_eye = (px, py, pz + eye_h)
            hidden = sum(1 for t_eye in threats if not self.world.has_los(cand_eye, t_eye))
            if hidden == 0:
                continue
            travel = math.hypot(px - x, py - y)
            score = hidden * 100.0 - travel * 1.0 - d_center * 0.25
            if score <= best_score:
                continue
            if not self.world.is_walkable(px, py, pz, radius=mr, height=mh):
                continue
            if (px != x or py != y) and not self.world.path_clear(me["pos"], (px, py, pz), radius=mr, height=mh):
                continue
            best_score = score
            best_pos = (px, py)

        self.cover_goal = best_pos
        self.cover_until = t + (COVER_HOLD_S if best_pos is not None else 0.4)
        return self.cover_goal

    def choose_kite(self, obs: mf.Observation, me, cx: float, cy: float, inner: float) -> Tuple[Optional[Tuple[float, float]], float]:
        """Open-field kiting/strafing when no building cover is available and at least one threat has LOS."""
        if not (FLAGS.get("kite", True) and self.world is not None and me["in_zone"]):
            return None, math.inf
        x, y, z = me["pos"]
        st = me["stats"]
        eye_h = st.get("eye_height", 8.0)
        mr, mh = st.get("radius"), st.get("height")
        me_eye = (x, y, z + eye_h)
        exposed_threats = [t_eye for t_eye in self.known_threats(obs, me) if self.world.has_los(me_eye, t_eye)]
        if not exposed_threats:
            return None, math.inf

        nt = min(exposed_threats, key=lambda te: math.hypot(te[0] - x, te[1] - y))
        td = math.hypot(nt[0] - x, nt[1] - y)
        tb = math.atan2(nt[1] - y, nt[0] - x)
        step_r = 16.0

        if td < 60.0 or len(exposed_threats) >= 2 or self.healing:
            away = tb + math.pi
            angles = [away, away + 0.5, away - 0.5, away + 1.0, away - 1.0, away + 1.5, away - 1.5]
            best_pt: Optional[Tuple[float, float]] = None
            best_score = -math.inf
            for idx, ang in enumerate(angles):
                kx, ky = x + step_r * math.cos(ang), y + step_r * math.sin(ang)
                d_center = math.hypot(kx - cx, ky - cy)
                if inner > 3.0 and d_center > inner:
                    continue
                kz = self.world.ground_z(kx, ky)
                if not self.world.is_walkable(kx, ky, kz, radius=mr, height=mh):
                    continue
                if not self.world.path_clear(me["pos"], (kx, ky, kz), radius=mr, height=mh):
                    continue
                d_threat = math.hypot(kx - nt[0], ky - nt[1])
                score = d_threat - 0.25 * d_center - 1.5 * idx
                if score > best_score:
                    best_score = score
                    best_pt = (kx, ky)
            return best_pt, td
        else:
            side = 1 if int(obs["t"] / 1.4) % 2 == 0 else -1
            strafe = tb + side * (math.pi * 0.55)
            angles = [strafe, tb - side * (math.pi * 0.55), tb + math.pi]
            for ang in angles:
                kx, ky = x + step_r * math.cos(ang), y + step_r * math.sin(ang)
                if inner > 3.0 and math.hypot(kx - cx, ky - cy) > inner:
                    continue
                kz = self.world.ground_z(kx, ky)
                if self.world.is_walkable(kx, ky, kz, radius=mr, height=mh) and self.world.path_clear(me["pos"], (kx, ky, kz), radius=mr, height=mh):
                    return (kx, ky), td
            return None, td

    def movement(self, obs: mf.Observation, me) -> dict:
        x, y = me["pos"][0], me["pos"][1]
        t = obs["t"]
        zone = obs["zone"]
        cx, cy = zone["next_center"]
        d = math.hypot(cx - x, cy - y)
        inner = zone["next_radius"] * 0.8     # comfortably inside the next circle
        if FLAGS.get("kite", True) and zone["next_radius"] > 100.0 and obs.get("alive", 16) > 4:
            settle = max(zone["next_radius"] * 0.60, 3.0)
        else:
            settle = max(zone["next_radius"] * 0.5, 3.0)

        self.late = d > inner and (d - inner) > 0.8 * me["stats"]["walk_speed"] * max(zone["shrink_end"] - t, 0.0)
        kite_boost = False
        goal = None
        if not me["in_zone"] or d > inner:
            self.cover_goal = None
            goal = (cx, cy)
        else:
            cover = self.choose_cover(obs, me, cx, cy, inner)
            if cover is not None:
                if math.hypot(cover[0] - x, cover[1] - y) <= COVER_REACH_M:
                    goal = None
                else:
                    goal = cover
            else:
                kite_goal, kite_td = self.choose_kite(obs, me, cx, cy, inner)
                if kite_goal is not None:
                    goal = kite_goal
                    if (self.healing or kite_td < 45.0) and me["energy"] > 35.0:
                        kite_boost = True
                else:
                    cube_goal = self.choose_cube(obs, me, cx, cy, inner)
                    if cube_goal is not None:
                        goal = cube_goal
                    elif self.loot_goal is not None:
                        goal = self.loot_goal
                    elif d > settle:
                        goal = (cx, cy)

        if goal is None:  # hold still: silent, and the gun is steadier
            self.move_yaw = None
            self.stuck_s = 0.0
            self.unstick_until = -1.0
            return {"move": [0.0, 0.0], "boost": False, "jump": False}

        yaw = math.atan2(goal[1] - y, goal[0] - x)
        if self.world is not None:
            mr, mh = me["stats"].get("radius"), me["stats"].get("height")
            px, py = x + 10.0 * math.cos(yaw), y + 10.0 * math.sin(yaw)
            if not self.world.path_clear(me["pos"], (px, py, self.world.ground_z(px, py)), radius=mr, height=mh):
                for delta in (0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6):
                    ay = yaw + delta
                    ax, ay_pos = x + 10.0 * math.cos(ay), y + 10.0 * math.sin(ay)
                    az = self.world.ground_z(ax, ay_pos)
                    if self.world.is_walkable(ax, ay_pos, az, radius=mr, height=mh) and self.world.path_clear(me["pos"], (ax, ay_pos, az), radius=mr, height=mh):
                        yaw = ay
                        break
        # Boost when outside the zone, when walking won't get us inside before the shrink ends, or when escaping threats.
        boost = not me["in_zone"] or self.late or kite_boost
        jump = False

        # Stuck against something: sidestep, jump, and jet up for a moment.
        if me["grounded"] and math.hypot(me["vel"][0], me["vel"][1]) < STUCK_SPEED:
            self.stuck_s += self.dt
        else:
            self.stuck_s = 0.0
        if self.stuck_s >= STUCK_AFTER_S and t >= self.unstick_until:
            self.unstick_until = t + UNSTICK_S
            self.unstick_side = -self.unstick_side
            self.stuck_s = 0.0
        if t < self.unstick_until:
            yaw += self.unstick_side * 1.1
            jump = me["grounded"]
            boost = boost or (not me["grounded"] and me["energy"] > 10)

        self.move_yaw = yaw
        return {"move": [math.cos(yaw), math.sin(yaw)], "boost": bool(boost), "jump": bool(jump)}

    # ------------------------------------------------------------------ combat
    def pick_target(self, obs: mf.Observation, me):
        """Best enemy mech weighted by LOS, distance, and HP fraction; NPCs only if they shot us recently."""
        mx, my, mz = me["pos"]
        src = (mx, my, mz + me["stats"]["muzzle_height"])
        best, best_s = None, math.inf
        for m in obs["mechs"]:
            if m["kind"] == "npc" and m["id"] not in self.aggressors:
                continue
            tp, _, _ = self.body(m)
            dist = math.dist(src, tp)
            los = (self.world is None) or self.world.has_los(src, tp)
            hp_frac = m.get("hp", 1400.0) / max(1.0, m.get("max_hp", 1400.0))
            score = dist * (0.5 + 0.5 * hp_frac)
            if m["kind"] == "npc":
                score += NPC_PENALTY_M
            if not los:
                score += 500.0
            if score < best_s:
                best, best_s = m, score
        return best

    def body(self, m):
        """(center point, radius, half height) of a mech or NPC."""
        if m["kind"] == "npc":
            h, r = self.npc_h, self.npc_r
        else:
            k = 1.0 + self.size_per_cube * m.get("cubes", 0)
            h, r = self.mech_h * k, self.mech_r * k
        p = m["pos"]
        return (p[0], p[1], p[2] + h / 2.0), r, h / 2.0

    def aim_at(self, src, tp, tv, speed, me):
        """[yaw, pitch] from src to tp, led by flight time at `speed` (None or 0 = no lead)."""
        px, py, pz = tp
        if speed:
            for _ in range(3):
                tt = math.dist(src, (px, py, pz)) / speed
                px, py, pz = tp[0] + tv[0] * tt, tp[1] + tv[1] * tt, tp[2] + tv[2] * tt
        dx, dy, dz = px - src[0], py - src[1], pz - src[2]
        st = me["stats"]
        pitch = min(max(math.atan2(dz, math.hypot(dx, dy)), st["min_pitch"]), st["max_pitch"])
        return [math.atan2(dy, dx), pitch]

    @staticmethod
    def aligned(actual, want, dist, radius, half_h) -> bool:
        """Is the torso's actual aim within the target's angular size of the wanted aim?"""
        dist = max(dist, 1.0)
        return (abs(wrap(actual[0] - want[0])) <= max(0.01, math.atan2(0.8 * radius, dist))
                and abs(actual[1] - want[1]) <= max(0.01, math.atan2(0.8 * half_h, dist)))

    def combat(self, obs: mf.Observation, me) -> dict:
        weapons = me["weapons"]
        fire = [False, False]
        target = self.pick_target(obs, me) if FLAGS["fight"] else None
        gun = next((w for w in weapons if w and w["kind"] == "gun"), None)

        if target is None:
            for i, w in enumerate(weapons):
                if w and w["kind"] == "laser" and w.get("charge", 0) > 0:
                    fire[i] = True  # hold charge (up to hold_s) if target briefly vanished
            aim = self.idle_aim(obs, me)
            out = {"aim": aim, "fire": fire}
            if gun and gun.get("reload", 0) == 0 and gun.get("ammo", 0) < gun["stats"]["mag"] * 0.5:
                out["reload"] = True  # top up while nobody is around
            self.draw(me, None, "idle")
            return out

        mx, my, mz = me["pos"]
        src = (mx, my, mz + me["stats"]["muzzle_height"])
        tp, radius, half_h = self.body(target)
        dist = math.dist(src, tp)
        has_los = getattr(self, "target_has_los", (self.world is None) or self.world.has_los(src, tp))
        direct = self.aim_at(src, tp, target["vel"], None, me)
        led = self.aim_at(src, tp, target["vel"], gun["stats"]["speed"], me) if gun and FLAGS["lead"] else direct

        # Aim for one primary weapon: a ready/charging laser (hitscan, direct), else the gun (led).
        laser = next((w for w in weapons if w and w["kind"] == "laser"), None)
        laser_primary = laser is not None and (laser.get("charge", 0) > 0 or laser["cooldown"] <= 0) \
            and dist <= laser["stats"]["range"]
        aim = direct if laser_primary or gun is None else led

        ceasefire = False
        if self.healing:
            if (has_los and dist <= 35.0) or (obs["t"] - self.last_damaged_t) <= 0.5:
                ceasefire = False
            else:
                ceasefire = True

        actual = me["aim"]
        for i, w in enumerate(weapons):
            if not w:
                continue
            k, st = w["kind"], w["stats"]
            if not has_los or ceasefire:
                if k == "laser" and w.get("charge", 0) > 0:
                    fire[i] = True
                else:
                    fire[i] = False
                continue
            if k == "gun":
                fire[i] = (w.get("ammo", 0) > 0 and w.get("reload", 0) == 0 and dist <= st["range"]
                           and self.aligned(actual, led, dist, radius, half_h))
            elif k == "lightning":
                fire[i] = (dist <= st["range"] * 0.95 and w["cooldown"] <= 0
                           and me["energy"] >= st.get("energy", 0) + LIGHTNING_ENERGY_RESERVE
                           and abs(wrap(actual[0] - direct[0])) <= math.radians(st["cone_deg"]) * 0.4)
            elif k == "laser":
                charge = w.get("charge", 0)
                if charge == 0:
                    fire[i] = (w["cooldown"] <= 0 and dist <= st["range"] and me["in_zone"]
                               and not self.late and not self.healing)
                elif charge < st["charge"] or dist > st["range"]:
                    fire[i] = True  # keep charging, or hold charge if target stepped out of range
                else:
                    # full and in range: hold until on target, then let go (auto-fires after hold_s anyway)
                    fire[i] = not self.aligned(actual, direct, dist, radius, half_h)
            # mortar: not used yet (needs ballistic aim); never taken while LOOT_PREF excludes it

        out = {"aim": aim, "fire": fire}
        if (not has_los or ceasefire) and gun and gun.get("reload", 0) == 0 and gun.get("ammo", 0) < gun["stats"]["mag"] * 0.5:
            out["reload"] = True
        self.draw(me, tp, f"target {target['id']} {dist:.0f}m")
        return out

    def idle_aim(self, obs: mf.Observation, me):
        """No target: face the last attacker, else recent danger sound, else look where we walk, else sweep."""
        if self.threat is not None and obs["t"] - self.threat[0] <= THREAT_MEMORY_S:
            return [self.threat[1], 0.0]
        if FLAGS.get("sound", True) and self.last_sound is not None and obs["t"] - self.last_sound[0] <= 1.5:
            return [self.last_sound[1], 0.0]
        if self.move_yaw is not None:
            return [self.move_yaw, 0.0]
        return [wrap(me["aim"][0] + 0.35), 0.0]

    # ------------------------------------------------------------------ debug
    def draw(self, me, tp, state: str) -> None:
        if not FLAGS["debug"]:
            return
        x, y, z = me["pos"]
        if tp is not None:
            mf.draw.line([x, y, z + me["stats"]["muzzle_height"]], list(tp), "#ff5040")
        if self.cover_goal is not None:
            mf.draw.point([self.cover_goal[0], self.cover_goal[1], z], "#40ff80", 6)
        mf.draw.label(f"{state} | hp {me['hp']:.0f} e {me['energy']:.0f}"[:80])


if __name__ == "__main__":
    mf.run(Bot)
