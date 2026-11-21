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

# IMPORTANT BENCHMARKING NOTE FOR ALL AGENTS:
# Always run tools/sim.py with 1 copy of Kestrel:
#     python3 tools/sim.py --matches 48 --bots bot:Bot=1,aggro=5,runner=5,example=5
# Running with bot:Bot=4 spawns 4 adjacent copies of Kestrel with 85%+ gun accuracy that kill each
# other at spawn, creating a false "4-copy truce bias" that rewards passive/ceasefire flags which
# lose badly (-1.04 pts) on the official 1v15 server.
FLAGS = {
    "loot": True,             # detour to and take weapon pickups (kinds in LOOT_PREF)
    "fight": True,            # aim at and shoot enemies
    "lead": True,             # lead moving targets by projectile flight time (gun)
    "cover": True,            # seek cover when inside the zone and threatened
    "los_gate": True,         # gate fire[i] on world.has_los(src, tp)
    "cubes": True,            # 20 m cube-seeking
    "early_weapon": True,     # H1: path-verified 85 m unpressured local-floor pickup detour (official server A/B vs 2.94)
    # Keep the flags below False (verified in 48-match 1-copy bot:Bot=1 ablations against e39d414 = 9.25):
    "safe_laser": False,      # gate laser charge on LOS, no cover_goal, hp>140 (-0.23 pts in 1-copy sim: 9.02 vs 9.25)
    "heal": False,            # HEAL_COVER ceasefire (-0.44 pts in 1-copy sim)
    "kite": False,            # open-field kite / 0.60 settle (-1.94 pts; overrides loot_goal)
    "sound": False,           # face recent "near" sounds in idle_aim (0.00 delta)
    "proactive_cover": False, # seek cover on reload/multi-enemy (-0.31 pts in 1-copy sim)
    "combat_strafe": False,   # continuous open-field strafe (-0.48 pts in 1-copy sim)
    "selective_fire": False,  # hold unprovoked fire >58m (-0.23 pts in 1-copy sim; 4-copy truce artifact)
    "debug": True,            # attach debug drawings (never affects the match)
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
        self.bad_pickups = {}   # pickup id -> t_expire blacklist
        self.aggressors = {}    # attacker id -> t it last damaged us
        self.threat = None      # (t, bearing) of the last non-zone damage
        self.threat_id = None
        self.cover_threats = []
        self.cover_exposed = False
        self.last_sound: Optional[Tuple[float, float]] = None
        self.healing = False
        self.last_damaged_t = -99.0
        self.last_move_t = -99.0
        self.late = False
        self.target_has_los = True
        self.last_t = None
        self.dt = 0.1
        self.loot_goal = None   # (x, y) of a pickup to walk to, set by choose_take
        self.loot_goal_id = None
        self.progress_anchor: Optional[Tuple[float, float, float, float, Optional[float]]] = None
        self.cover_goal: Optional[Tuple[float, float]] = None
        self.cover_until = -1.0
        self.cover_retry_at = -1.0
        self.move_yaw = None    # yaw we are walking along, None when holding still
        self.stuck_s = 0.0
        self.unstick_until = -1.0
        self.unstick_side = 1
        self.strafe_flip_t = 0.0
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
                        self.threat_id = e["by"]
        for e in obs["events"]:
            if e.get("type") == "kill":
                target = e["target"]
                self.seen.pop(target, None)
                self.aggressors.pop(target, None)
                if target == self.threat_id:
                    self.threat = None
                    self.threat_id = None
                    self.cover_goal = None
                    self.cover_threats = []
            elif e.get("type") == "take_failed":
                pid = e.get("id")
                if pid is not None:
                    self.bad_pickups[pid] = t + 15.0
        cutoff = t - self.memory_s
        self.seen = {k: v for k, v in self.seen.items() if v[0] >= cutoff}
        self.aggressors = {k: v for k, v in self.aggressors.items() if v >= t - AGGRO_MEMORY_S}
        self.bad_pickups = {k: v for k, v in self.bad_pickups.items() if v > t}

        me = obs["self"]
        if not FLAGS.get("heal", False) or not me["in_zone"]:
            self.healing = False
        elif me["hp"] < 0.65 * me["max_hp"]:
            self.healing = True
        elif me["hp"] >= 0.85 * me["max_hp"]:
            self.healing = False

        if FLAGS.get("sound", False):
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
        if not offer or not offer.get("choices"):
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
        self.loot_goal_id = None
        if not FLAGS.get("loot") or None not in me["weapons"]:
            return {}
        held = {w["kind"] for w in me["weapons"] if w}
        x, y, z = me["pos"]
        zone = obs["zone"]
        cx, cy = zone["next_center"]
        inner = zone["next_radius"] * 0.8
        early_w = FLAGS.get("early_weapon", False)
        max_detour = 85.0 if early_w else LOOT_DETOUR_M
        st = me["stats"]
        mr, mh = st.get("radius"), st.get("height")
        enemy_has_los: Optional[bool] = None

        best, best_d = None, max_detour
        for p in obs["pickups"]:
            pid = p["id"]
            if pid in self.bad_pickups:
                continue
            if p["kind"] not in LOOT_PREF or p["kind"] in held:
                continue
            px, py, pz = p["pos"][0], p["pos"][1], p["pos"][2]
            d = math.hypot(px - x, py - y)
            if d >= best_d:
                continue
            if d <= self.reach:
                if abs(pz - z) > 4.0:
                    continue
            else:
                if early_w and d > LOOT_DETOUR_M:
                    if enemy_has_los is None:
                        me_eye = (x, y, z + st.get("eye_height", 8.0))
                        enemy_has_los = any(
                            m["kind"] == "mech"
                            and (self.world is None or self.world.has_los(me_eye, (m["pos"][0], m["pos"][1], m["pos"][2] + 8.0)))
                            for m in obs["mechs"]
                        )
                    if enemy_has_los:
                        continue
                if early_w and self.world is not None:
                    # Check pickup height against its own local terrain floor (fixes hill rejection bug)
                    if abs(pz - self.world.ground_z(px, py)) > 2.5:
                        continue
                elif abs(pz - z) > 4.0:
                    continue
                if math.hypot(px - cx, py - cy) > inner:
                    continue
                if early_w and self.world is not None:
                    if not (self.world.is_walkable(px, py, pz, radius=mr, height=mh)
                            and self.world.path_clear(me["pos"], (px, py, pz), radius=mr, height=mh)):
                        continue
            best, best_d = p, d
        if best is None:
            return {}
        self.loot_goal = (best["pos"][0], best["pos"][1])
        self.loot_goal_id = best["id"]
        return {"take": best["id"]} if best_d <= self.reach and abs(best["pos"][2] - z) <= 4.0 else {}

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
        if self.threat_id not in by_id and self.threat is not None and obs["t"] - self.threat[0] <= THREAT_MEMORY_S:
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
        t = obs["t"]
        recent_hit = self.threat is not None and t - self.threat[0] < THREAT_MEMORY_S
        pursuing = self.cover_goal is not None and t < self.cover_until

        proactive = False
        if FLAGS.get("proactive_cover", False):
            gun = next((w for w in me["weapons"] if w and w["kind"] == "gun"), None)
            reloading = gun is not None and gun.get("reload", 0) > 0.3
            enemy_charging = any(m.get("charging") for m in obs["mechs"] if m.get("kind") == "mech")
            vis_enemies = [m for m in obs["mechs"] if m.get("kind") == "mech"]
            proactive = self.healing or reloading or enemy_charging or len(vis_enemies) >= 2

        if not (recent_hit or pursuing or proactive):
            self.cover_goal = None
            return None

        threats = self.known_threats(obs, me)
        if pursuing and not threats:
            threats = self.cover_threats
        st = me["stats"]
        eye_h = st.get("eye_height", 8.0)
        mr, mh = st.get("radius"), st.get("height")
        cur_eye = (x, y, z + eye_h)
        exposed_threats = []
        unexposed_threats = []
        for te in threats:
            if self.world.has_los(cur_eye, te):
                exposed_threats.append(te)
            else:
                unexposed_threats.append(te)
        self.cover_exposed = bool(exposed_threats)
        if pursuing:
            gx, gy = self.cover_goal
            gz = self.world.support_z(gx, gy, z)
            if (math.hypot(gx - cx, gy - cy) <= inner
                    and all(not self.world.has_los((gx, gy, gz + eye_h), te)
                            for te in (exposed_threats or threats))
                    and self.world.path_clear(me["pos"], (gx, gy, gz), radius=mr, height=mh)):
                self.cover_threats = threats
                return self.cover_goal
        if not (recent_hit or proactive) or not exposed_threats or (self.cover_goal is None and t < self.cover_retry_at):
            self.cover_goal = None
            return None

        base_angle = math.atan2(cy - y, cx - x)
        radii = (12.0, 24.0, 36.0, 48.0) if FLAGS.get("proactive_cover", False) else (12.0, 24.0, 36.0)
        n_threats = len(threats)
        n_exposed = len(exposed_threats)

        best_pos: Optional[Tuple[float, float]] = None
        best_score = -math.inf
        for r in radii:
            for k in range(12):
                ang = base_angle + TAU * k / 12.0
                px = x + r * math.cos(ang)
                py = y + r * math.sin(ang)
                d_center = math.hypot(px - cx, py - cy)
                if inner > 3.0 and d_center > inner:
                    continue
                if n_threats * 100.0 - r - d_center * 0.25 <= best_score:
                    continue
                pz = self.world.support_z(px, py, z)
                cand_eye = (px, py, pz + eye_h)
                if any(self.world.has_los(cand_eye, t_eye) for t_eye in exposed_threats):
                    continue
                hidden = n_exposed + sum(1 for t_eye in unexposed_threats if not self.world.has_los(cand_eye, t_eye))
                score = hidden * 100.0 - r - d_center * 0.25
                if score <= best_score:
                    continue
                if not self.world.is_walkable(px, py, pz, radius=mr, height=mh):
                    continue
                if not self.world.path_clear(me["pos"], (px, py, pz), radius=mr, height=mh):
                    continue
                best_score = score
                best_pos = (px, py)

        self.cover_goal = best_pos
        self.cover_threats = threats
        if best_pos is not None:
            travel_s = math.dist((x, y), best_pos) / st["walk_speed"]
            self.cover_until = t + travel_s + COVER_HOLD_S
            self.cover_retry_at = -1.0
        else:
            self.cover_until = -1.0
            self.cover_retry_at = t + 0.4
        return self.cover_goal

    def choose_kite(self, obs: mf.Observation, me, cx: float, cy: float, inner: float) -> Tuple[Optional[Tuple[float, float]], float]:
        """Open-field kiting when FLAGS['kite'] is enabled, no building cover is available, and a threat has LOS.
        NOTE: Keep FLAGS['kite'] = False unless decoupled from settle radius and evaluated lower than loot_goal,
        otherwise a single exposed enemy within 85 m aborts secondary weapon pickups at full HP."""
        if not (FLAGS.get("kite", False) and self.world is not None and me["in_zone"]):
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
        kite_dist = 85.0 if FLAGS.get("combat_strafe", False) else 55.0
        if not (td < kite_dist or len(exposed_threats) >= 2 or self.healing):
            return None, td
        tb = math.atan2(nt[1] - y, nt[0] - x)
        step_r = 16.0
        cur_dc = math.hypot(x - cx, y - cy)
        max_dc = max(cur_dc + 6.0, inner) if FLAGS.get("combat_strafe", False) else max(cur_dc, inner * 0.75)

        away = tb + math.pi
        angles = [away, away + 0.5, away - 0.5, away + 1.0, away - 1.0, away + 1.5, away - 1.5]
        best_pt: Optional[Tuple[float, float]] = None
        best_score = -math.inf
        for idx, ang in enumerate(angles):
            kx, ky = x + step_r * math.cos(ang), y + step_r * math.sin(ang)
            d_center = math.hypot(kx - cx, ky - cy)
            if inner > 3.0 and d_center > max_dc:
                continue
            kz = self.world.support_z(kx, ky, z)
            if not self.world.is_walkable(kx, ky, kz, radius=mr, height=mh):
                continue
            if not self.world.path_clear(me["pos"], (kx, ky, kz), radius=mr, height=mh):
                continue
            d_threat = math.hypot(kx - nt[0], ky - nt[1])
            breaks_los = not self.world.has_los((kx, ky, kz + eye_h), nt)
            score = d_threat + (35.0 if breaks_los else 0.0) - 0.35 * d_center - 1.5 * idx
            if score > best_score:
                best_score = score
                best_pt = (kx, ky)
        return best_pt, td

    def choose_combat_strafe(self, obs: mf.Observation, me, cx: float, cy: float, inner: float) -> Tuple[Optional[Tuple[float, float]], bool]:
        """Lateral juking strafe + weapon spacing + charging-laser evade when exposed in combat."""
        if not (FLAGS.get("combat_strafe", False) and me["in_zone"] and inner > 3.0):
            return None, False
        x, y, z = me["pos"]
        t = obs["t"]
        st = me["stats"]
        eye_h = st.get("eye_height", 8.0)
        mr, mh = st.get("radius"), st.get("height")
        me_eye = (x, y, z + eye_h)

        # Find closest visible enemy mech with LOS (or charging laser)
        vis_mechs = []
        charging_threat = None
        for m in obs["mechs"]:
            if m.get("kind") != "mech":
                continue
            ep = m["pos"]
            e_eye = (ep[0], ep[1], ep[2] + 8.0)
            if self.world is None or self.world.has_los(me_eye, e_eye):
                d = math.hypot(ep[0] - x, ep[1] - y)
                if d <= 145.0:
                    vis_mechs.append((d, m))
                    if m.get("charging"):
                        charging_threat = m
        if not vis_mechs:
            return None, False

        vis_mechs.sort(key=lambda item: item[0])
        ed, enemy_m = vis_mechs[0]
        ep = enemy_m["pos"]
        tb = math.atan2(ep[1] - y, ep[0] - x)

        # Periodically flip strafe direction to defeat linear velocity-lead aiming
        if t >= self.strafe_flip_t:
            self.unstick_side = -self.unstick_side
            self.strafe_flip_t = t + 0.55

        # Desired weapon range spacing (kyoto-max style)
        held = {w["kind"]: w for w in me["weapons"] if w}
        enemy_weapons = set(enemy_m.get("weapons") or ())
        if "lightning" in enemy_weapons and "lightning" not in held:
            want_d = 75.0
        elif "lightning" in held and me["energy"] >= 25.0 and ed < 55.0:
            want_d = 24.0
        elif "laser" in held:
            want_d = 95.0
        else:
            want_d = 62.0

        radial = max(-0.65, min(0.65, (ed - want_d) / 25.0))
        for side in (self.unstick_side, -self.unstick_side):
            perp = tb + side * math.pi * 0.5
            ux = math.cos(perp) + radial * math.cos(tb)
            uy = math.sin(perp) + radial * math.sin(tb)
            norm = math.hypot(ux, uy) or 1.0
            dx, dy = ux / norm, uy / norm
            sx, sy = x + 8.0 * dx, y + 8.0 * dy
            if math.hypot(sx - cx, sy - cy) > inner:
                # Bias inward toward next_center near zone edge
                in_yaw = math.atan2(cy - y, cx - x)
                dx = 0.5 * dx + 0.5 * math.cos(in_yaw)
                dy = 0.5 * dy + 0.5 * math.sin(in_yaw)
                norm = math.hypot(dx, dy) or 1.0
                dx, dy = dx / norm, dy / norm
                sx, sy = x + 8.0 * dx, y + 8.0 * dy
            if self.world is not None:
                sz = self.world.support_z(sx, sy, z)
                if not (self.world.is_walkable(sx, sy, sz, radius=mr, height=mh)
                        and self.world.path_clear(me["pos"], (sx, sy, sz), radius=mr, height=mh)):
                    continue
            self.unstick_side = side
            dodge_boost = charging_threat is not None and me["energy"] > 20.0
            return (x + dx, y + dy), dodge_boost
        return None, False

    def movement(self, obs: mf.Observation, me) -> dict:
        x, y = me["pos"][0], me["pos"][1]
        t = obs["t"]
        self.last_move_t = t
        zone = obs["zone"]
        cx, cy = zone["next_center"]
        d = math.hypot(cx - x, cy - y)
        inner = zone["next_radius"] * 0.8     # comfortably inside the next circle
        # NOTE: Do not widen settle to 0.60 * next_radius by default (coupled to FLAGS['kite']);
        # standing still at 60% instead of rotating to 50% early hurt placement (e.g. seed 47).
        if FLAGS.get("kite", False) and zone["next_radius"] > 100.0 and obs.get("alive", 16) > 4:
            settle = max(zone["next_radius"] * 0.60, 3.0)
        else:
            settle = max(zone["next_radius"] * 0.5, 3.0)

        late_r = max(inner, 4.0) if zone["next_radius"] == 0 else inner
        self.late = d > late_r and (d - late_r) > 0.8 * me["stats"]["walk_speed"] * max(zone["shrink_end"] - t, 0.0)
        kite_boost = False
        goal = None
        if not me["in_zone"] or d > inner:
            self.cover_goal = None
            goal = (cx, cy)
        else:
            cover = self.choose_cover(obs, me, cx, cy, inner)
            if cover is not None:
                if math.hypot(cover[0] - x, cover[1] - y) <= COVER_REACH_M and not self.cover_exposed:
                    goal = None
                else:
                    goal = cover
            else:
                kite_goal, kite_td = self.choose_kite(obs, me, cx, cy, inner)
                if kite_goal is not None:
                    goal = kite_goal
                    if (self.healing or kite_td < 45.0) and me["energy"] > 35.0:
                        kite_boost = True
                elif self.threat and t - self.threat[0] < THREAT_MEMORY_S:
                    # Post-hit strafe fallback when no cover building is nearby, checked against walls.
                    bearing = self.threat[1]
                    if FLAGS.get("combat_strafe", False) and t >= self.strafe_flip_t:
                        self.unstick_side = -self.unstick_side
                        self.strafe_flip_t = t + 0.55
                    dx, dy = -math.sin(bearing), math.cos(bearing)
                    if self.unstick_side < 0:
                        dx, dy = -dx, -dy
                    if inner > 3:
                        if self.world is not None:
                            mr, mh = me["stats"].get("radius"), me["stats"].get("height")
                            sx, sy = x + 8.0 * dx, y + 8.0 * dy
                            sz = self.world.support_z(sx, sy, me["pos"][2])
                            if not (self.world.is_walkable(sx, sy, sz, radius=mr, height=mh)
                                    and self.world.path_clear(me["pos"], (sx, sy, sz), radius=mr, height=mh)):
                                dx, dy = -dx, -dy
                                self.unstick_side = -self.unstick_side
                        goal = (x + dx, y + dy)
                    else:
                        goal = (cx, cy)
                else:
                    strafe_goal, strafe_boost = self.choose_combat_strafe(obs, me, cx, cy, inner)
                    if strafe_goal is not None:
                        goal = strafe_goal
                        kite_boost = strafe_boost
                    elif self.loot_goal is not None:
                        goal = self.loot_goal
                    elif FLAGS.get("cubes", True) and me.get("cubes", 0) < self.cfg["cubes"].get("stat_cap", 10):
                        mz = me["pos"][2]
                        best_cd = 20.0
                        for c in obs["cubes"]:
                            cp = c["pos"]
                            if abs(cp[2] - mz) > 4.0:
                                continue
                            cd = math.hypot(cp[0] - x, cp[1] - y)
                            if cd <= best_cd and math.hypot(cp[0] - cx, cp[1] - cy) <= inner:
                                best_cd = cd
                                goal = (cp[0], cp[1])
                if goal is None and cover is None and d > settle:
                    goal = (cx, cy)

        if goal is None:  # hold still: silent, and the gun is steadier
            self.move_yaw = None
            self.stuck_s = 0.0
            self.unstick_until = -1.0
            self.progress_anchor = None
            return {"move": [0.0, 0.0], "boost": False, "jump": False}

        if FLAGS.get("early_weapon", False) and self.loot_goal is not None and goal == self.loot_goal:
            goal_d = math.hypot(goal[0] - x, goal[1] - y)
            if self.progress_anchor is None or self.progress_anchor[4] != self.loot_goal_id:
                self.progress_anchor = (t, x, y, goal_d, self.loot_goal_id)
            elif t - self.progress_anchor[0] >= 2.2:
                net_disp = math.hypot(x - self.progress_anchor[1], y - self.progress_anchor[2])
                goal_prog = self.progress_anchor[3] - goal_d
                if net_disp < 3.0 and goal_prog < 2.0 and goal_d > 4.5 and self.loot_goal_id is not None:
                    self.bad_pickups[self.loot_goal_id] = t + 15.0
                    self.loot_goal = None
                self.progress_anchor = (t, x, y, goal_d, self.loot_goal_id)
        else:
            self.progress_anchor = None

        yaw = math.atan2(goal[1] - y, goal[0] - x)
        # Boost when outside the zone, or when walking won't get us inside before the shrink ends.
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

        speed = 1.0
        if zone["next_radius"] == 0 and t >= self.unstick_until:
            speed = min(1.0, d / 4.0)
            if d < 4.0:
                self.stuck_s = 0.0
                boost = False
        self.move_yaw = yaw
        return {"move": [speed * math.cos(yaw), speed * math.sin(yaw)], "boost": bool(boost), "jump": bool(jump)}

    # ------------------------------------------------------------------ combat
    def pick_target(self, obs: mf.Observation, me):
        """Nearest enemy mech (preferring clear LOS and low-HP when heal/los_gate active); NPCs only if aggressors."""
        mx, my, mz = me["pos"]
        src = (mx, my, mz + me["stats"]["muzzle_height"])
        best, best_s, best_los = None, math.inf, True
        for m in obs["mechs"]:
            if m["kind"] == "npc" and m["id"] not in self.aggressors:
                continue
            tp, _, _ = self.body(m)
            dist = math.dist(src, tp)
            hp_frac = m.get("hp", 1400.0) / max(1.0, m.get("max_hp", 1400.0))
            s = dist * (0.5 + 0.5 * hp_frac)
            if m["kind"] == "npc":
                s += NPC_PENALTY_M
            if s >= best_s:
                continue
            los = (self.world is None) or self.world.has_los(src, tp)
            if not los:
                s += 500.0
            if s < best_s:
                best, best_s, best_los = m, s, los
        self.target_has_los = best_los
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

    def aim_at(self, src, tp, tv, speed, me, clamp=True):
        """[yaw, pitch] from src to tp, led by flight time at `speed` (None or 0 = no lead)."""
        px, py, pz = tp
        if speed:
            for _ in range(3):
                tt = math.dist(src, (px, py, pz)) / speed
                px, py, pz = tp[0] + tv[0] * tt, tp[1] + tv[1] * tt, tp[2] + tv[2] * tt
        dx, dy, dz = px - src[0], py - src[1], pz - src[2]
        st = me["stats"]
        pitch = math.atan2(dz, math.hypot(dx, dy))
        if clamp:
            pitch = min(max(pitch, st["min_pitch"]), st["max_pitch"])
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
        has_los = (not FLAGS.get("los_gate", True)) or self.target_has_los
        direct = self.aim_at(src, tp, target["vel"], None, me, clamp=False)
        led = self.aim_at(src, tp, target["vel"], gun["stats"]["speed"], me, clamp=False) if gun and FLAGS["lead"] else direct

        # Aim for one primary weapon: a ready/charging laser (hitscan, direct), else the gun (led).
        laser = next((w for w in weapons if w and w["kind"] == "laser"), None)
        if FLAGS.get("safe_laser", False):
            can_charge = (laser is not None and laser["cooldown"] <= 0
                          and dist <= laser["stats"]["range"] and me["in_zone"]
                          and has_los and self.cover_goal is None
                          and target.get("hp", 1400.0) > 140.0
                          and not self.late and not self.healing)
        else:
            can_charge = (laser is not None and laser["cooldown"] <= 0
                          and dist <= laser["stats"]["range"] and me["in_zone"]
                          and not self.late and not self.healing)
        laser_primary = laser is not None and dist <= laser["stats"]["range"] and (
            laser.get("charge", 0) > 0 or can_charge)
        yaw, pitch = direct if laser_primary or gun is None else led
        aim = [yaw, min(max(pitch, me["stats"]["min_pitch"]), me["stats"]["max_pitch"])]

        ceasefire = False
        if self.healing:
            if not ((has_los and dist <= 35.0) or (obs["t"] - self.last_damaged_t) <= 0.5):
                ceasefire = True
        if not ceasefire and FLAGS.get("selective_fire", False):
            t_now = obs["t"]
            recent_hit = self.threat is not None and t_now - self.threat[0] < THREAT_MEMORY_S
            provoked = target["id"] in self.aggressors or recent_hit
            finishable = target.get("hp", 1400.0) <= 550.0
            in_full_step = self.last_move_t == t_now
            max_unprovoked_d = 58.0 if (in_full_step and obs.get("alive", 16) > 4) else 108.0
            if not provoked and not finishable and dist > max_unprovoked_d:
                ceasefire = True
            elif not provoked and not finishable and me["hp"] < me["max_hp"] and (t_now - self.last_damaged_t) < 5.2 and dist > 45.0:
                # Let the 5s out-of-combat timer complete when damaged and enemy isn't point-blank
                ceasefire = True

        actual = me["aim"]
        for i, w in enumerate(weapons):
            if not w:
                continue
            k, st = w["kind"], w["stats"]
            if not has_los or ceasefire:
                if k == "laser" and w.get("charge", 0) > 0:
                    fire[i] = True  # hold existing charge rather than dumping into wall
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
                    fire[i] = can_charge
                elif charge < st["charge"] or dist > st["range"]:
                    fire[i] = True  # keep charging, or hold charge if target stepped out of range
                else:
                    # full and in range: hold until on target, then let go (auto-fires after hold_s anyway)
                    fire[i] = not self.aligned(actual, direct, dist, radius, half_h)
            # mortar: not used yet (needs ballistic aim); never taken while LOOT_PREF excludes it

        self.draw(me, tp, f"target {target['id']} {dist:.0f}m")
        return {"aim": aim, "fire": fire}

    def idle_aim(self, obs: mf.Observation, me):
        """No target: face the last attacker, else recent 'near' sound, else look where we walk, else sweep."""
        if self.threat is not None and obs["t"] - self.threat[0] <= THREAT_MEMORY_S:
            return [self.threat[1], 0.0]
        if FLAGS.get("sound", False) and self.last_sound is not None and obs["t"] - self.last_sound[0] <= 1.5:
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
