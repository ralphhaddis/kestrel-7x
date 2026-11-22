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
from typing import Dict, List, Optional, Tuple

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
    "astar_nav": True,        # Precomputed 5 m A* routing around buildings when direct path_clear fails
    "evade": True,            # dodge incoming mortar impacts and charging-laser aim rays
    "farm_crate": True,       # safe opening crate farming when zero enemies are around
    "mortar": True,           # closed-form ballistic mortar solver + over-building lobs
    "weapon_tri": True,       # weapon-triangle target priority + smart slot-1 weapon swap
    "drops": True,            # opportunistic supply drop capture
    "corner_cover": True,     # precomputed exterior building corner cover candidates
    "laser_opt": True,        # drop laser charge when out of zone/evading/late + early-release finisher
    "rear_guard": True,       # face threat/seen/near-sound while idle or retreating without LOS
    "energy_mgmt": True,      # reserve 30 energy for jets unless urgent zap or urgent boost
    "smart_upgrades": True,   # loadout-aware tiered upgrade scoring
    "w4_combat": True,        # W4-1: tick-quantized CDs, gun-during-laser-charge + pre-snap, exact upgrades, 180-deg arc, fast partial cover + boost
    "laser_first": True,      # W4-2: zero-mortar pickup policy (no 504-dmg cookoff), hill-floor fix, safe 75m laser detour & 35m swap
    "farm_npc": True,         # W4-4: safe 2-cube NPC farming (<45m, cubes<6, zero mechs) + 35m uncontested cube sweep
    "early_survival": True,   # W4-5: delayed Phase-1 center convergence (0.78R settle while t < shrink_start-10 and alive >= 10)
    "w5_safe_bound": True,    # W5-1: slack-time dynamic safe boundary so d > inner doesn't disable cover/loot/micro in outer 54% spawns
    "w5_laser_mastery": True, # W5-2: committed lateral laser dodge (no centerline ping-pong) + hold_s no-LOS timeout release
    "w5_nav_mastery": True,   # W5-3: zero-jitter A* caching + stuck waypoint blacklisting + endgame circle mastery
    "w6_crossfire_escape": True, # W6-1: radial 48m cover fallback + boosted pinch escape + anti-rusher backpedal + NPC-fire dodge
    "w6_loot_hunt": True,     # W6-2: safe 62m lightning & 85m unboosted laser detour when slot 1 is empty and zero enemies visible
    "w6_endgame_mastery": True, # W6-3: energy reserve & tight center control in final circles (next_radius <= 35m)
    "w7_squad_survival": True, # W7-1: tree-canopy stealth concealment + 1v2+ proactive squad cover + 4-cube crate farm + mid-sound pre-aim
    "w8_kill_conversion": True, # W8-1: strict focus-fire execute (<35% HP), aggressive chase-to-finish on wounded 1v1 (<45% HP), & aggressive cube/weapon pickup
    "w9_squad_hunter": False, # W9-2: unarmed patrol & 1v1 rush (-0.14 official: 3.95 vs 4.09; keep False)
    "w11_combat_juke": False, # W11-1: 135m supply drop chase (4.00 vs 4.09; keep False, fold 0.45s juke into w12)
    "w12_bullet_evade": False, # W12-1: bullet sidestep + cover-regen hold (3.93 vs 4.09; keep False)
    "w13_squad_endurance": True, # W13-1: distant-enemy-safe weapon hunt (>65m, pickup >50m from enemies), small-zone lightning swap (30m), gun_damage/rate upgrade priority
    # Keep the flags below False (verified in upgraded strong-lobby & 1-copy ablations):
    "aggressive_drops": False, # W4-3: 150m drop chase/contest (-0.24 ± 0.18 SE, 4/8/0 pairs in upgraded Mirror)
    "early_weapon": False,    # 85 m pickup detour (-0.27 in 1-copy sim; -0.21 official: 2.94 -> 2.73)
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
ENERGY_RESERVE = 30.0            # energy reserve for jets under energy_mgmt
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
        self.regen_delay = float(cfg["mech"].get("regen_delay", 5.0))
        self.size_per_cube = cfg["cubes"].get("size_per_cube", 0.0)
        self.crate_r = float(cfg.get("crates", {}).get("radius", 1.4))
        self.crate_h = float(cfg.get("crates", {}).get("height", 2.4))
        self.npc_h = cfg["npc"]["height"]
        self.npc_r = cfg["npc"]["radius"]
        self.reach = cfg["loot"]["pickup_radius"]
        mcfg = cfg.get("weapons", {}).get("mortar", {})
        self.mortar_ttl = float(mcfg.get("ttl", 7.0))
        self.mortar_speed = float(mcfg.get("speed", 72.0))
        self.mortar_gravity = float(mcfg.get("gravity", 22.0))
        self.mortar_g = self.mortar_gravity
        self.mortar_splash = float(mcfg.get("splash_radius", 7.0))
        self.laser_range = float(cfg.get("weapons", {}).get("laser", {}).get("range", 280.0))

        self.world = wm.WorldMap.load(hello)
        self.seen = {}          # actor id -> (t, pos, vel), last time we saw it
        self.seen_hp = {}       # actor id -> (t, hp_frac), last known HP fraction
        self.bad_pickups = {}   # pickup id -> t_expire blacklist
        self.crate_target = None
        self.bad_crates = {}    # crate id -> t_expire blacklist
        self.crate_track = None  # (cid, start_t, start_hp)
        self.aggressors = {}    # attacker id -> t it last damaged us
        self.threat = None      # (t, bearing) of the last non-zone damage
        self.threat_id = None
        self.cover_threats = []
        self.cover_exposed = False
        self.cover_goal_exposed = False
        self.active_drop_holder: Optional[int] = None
        self.npc_farm_target: Optional[dict] = None
        self.last_sound: Optional[Tuple[float, float]] = None
        self.last_near_sound: Optional[Tuple[float, float]] = None
        self.upgrade_req: Dict[str, Optional[str]] = {}
        for u in cfg.get("upgrades", ()):
            uid = u.get("id")
            if not uid:
                continue
            req = u.get("requires")
            if req is None:
                for eff in u.get("effects", ()):
                    path = str(eff.get("path", ""))
                    if path.startswith("weapons."):
                        parts = path.split(".")
                        if len(parts) >= 2 and parts[1]:
                            req = parts[1]
                            break
            self.upgrade_req[uid] = req
        self.healing = False
        self.last_damaged_t = -99.0
        self.last_move_t = -99.0
        self.last_swap_t = -99.0
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
        self.nav_path: Optional[List[Tuple[float, float]]] = None
        self.nav_goal: Optional[Tuple[float, float]] = None
        self.nav_until = -1.0
        self.evading = False
        self.evade_impacts: List[Tuple[float, float, float]] = []
        self.laser_dodge_dir: Dict[int, Tuple[float, float]] = {}  # enemy_id -> (expires_t, side)
        self.laser_no_los_s: float = 0.0
        self.bad_waypoints: Dict[Tuple[int, int], float] = {}      # (wx_cell, wy_cell) -> expires_t
        self.last_wp_pop_t: float = -99.0
        self.patrol_angle: float = self.rng.uniform(0.0, TAU)
        self.patrol_until: float = -1.0
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
            if m.get("kind") == "mech":
                self.seen_hp[m["id"]] = (t, m.get("hp", 1400.0) / max(1.0, m.get("max_hp", 1400.0)))
        for e in obs["events"]:
            if e.get("type") == "damaged":
                self.last_damaged_t = t
                by = e.get("by")
                if by is not None:
                    self.aggressors[by] = t
                    if e.get("bearing") is not None:
                        if (FLAGS.get("w6_crossfire_escape", True)
                                and int(by) >= 100
                                and (any(m.get("kind") == "mech" for m in obs["mechs"])
                                     or any(aid < 16 for aid in self.seen))):
                            pass
                        else:
                            self.threat = (t, e["bearing"])
                            self.threat_id = by
        for e in obs["events"]:
            if e.get("type") == "kill":
                target = e["target"]
                self.seen.pop(target, None)
                self.seen_hp.pop(target, None)
                self.aggressors.pop(target, None)
                if target == self.threat_id:
                    self.threat = None
                    self.threat_id = None
                    self.cover_goal = None
                    self.cover_threats = []
            elif e.get("type") == "took":
                self.last_swap_t = t
            elif e.get("type") == "take_failed":
                pid = e.get("id")
                if pid is not None and e.get("reason") in ("gone", "have", "full"):
                    self.bad_pickups[pid] = t + 15.0
        cutoff = t - self.memory_s
        self.seen = {k: v for k, v in self.seen.items() if v[0] >= cutoff}
        self.seen_hp = {k: v for k, v in self.seen_hp.items() if v[0] >= cutoff}
        self.aggressors = {k: v for k, v in self.aggressors.items() if v >= t - AGGRO_MEMORY_S}
        self.bad_pickups = {k: v for k, v in self.bad_pickups.items() if v > t}
        self.bad_crates = {k: v for k, v in self.bad_crates.items() if v > t}
        self.bad_waypoints = {k: v for k, v in self.bad_waypoints.items() if v > t}
        self.laser_dodge_dir = {k: v for k, v in self.laser_dodge_dir.items() if v[0] > t}

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

        if FLAGS.get("rear_guard", True):
            w7_sq = FLAGS.get("w7_squad_survival", True)
            best_prio = 0
            best_bearing = None
            for s in obs.get("sounds", ()):
                d_s = s.get("dist")
                if d_s != "near" and not (w7_sq and d_s == "mid"):
                    continue
                b = s.get("bearing")
                if b is None:
                    continue
                k = s.get("kind")
                if k in ("gun", "laser", "laser_charge", "lightning", "mortar", "explosion"):
                    prio = 3 if d_s == "near" else 1
                elif k in ("boost", "jet", "step"):
                    prio = 2 if d_s == "near" else 0
                else:
                    prio = 0
                if prio > best_prio:
                    best_prio = prio
                    best_bearing = float(b)
            if best_bearing is not None:
                self.last_near_sound = (t, best_bearing)

    # ------------------------------------------------------------------ upgrades
    def choose_pick(self, obs: mf.Observation) -> dict:
        offer = obs.get("offer")
        if not offer or not offer.get("choices"):
            return {}
        choices = offer["choices"]
        if FLAGS.get("smart_upgrades", True):
            held = {w["kind"] for w in obs["self"]["weapons"] if w}
            if FLAGS.get("w4_combat", True):
                nanites_stacks = sum(1 for u in obs["self"].get("upgrades", ()) if u == "nanites")
                alive = obs.get("alive", 16)
                w13_se = FLAGS.get("w13_squad_endurance", True)
                held_weapon_scores = {
                    "laser_damage": 87 if w13_se else 86,
                    "laser_charge": 86 if w13_se else 85,
                    "gun_damage": 85 if w13_se else 80,
                    "gun_rate": 84 if w13_se else 74,
                    "lightning_range": 83,
                    "gun_mag": 81 if w13_se else 84,
                    "lightning_efficiency": 78 if w13_se else 77,
                    "mortar_reload": 72,
                    "mortar_splash": 70,
                    "gun_velocity": 68,
                    "mortar_velocity": 66,
                    "lightning_chain": 64,
                    "laser_pierce": 62,
                }
                non_weapon_scores = {
                    "armor": 100,
                    "plating": 98,
                    "nanites": 94 if nanites_stacks == 0 else 88,
                    "sensors": 76 if alive > 8 else 58,
                    "servos": 65,
                    "gyros": 56,
                    "reactor": 56 if "lightning" in held else 48,
                    "capacitor": 48 if "lightning" in held else 42,
                    "jets": 40,
                }
            else:
                held_weapon_scores = {
                    "laser_charge": 82,
                    "laser_damage": 80,
                    "mortar_reload": 78,
                    "mortar_splash": 76,
                    "lightning_range": 78,
                    "lightning_efficiency": 76,
                    "gun_rate": 75,
                    "gun_damage": 74,
                    "gun_mag": 70,
                    "gun_velocity": 68,
                    "mortar_velocity": 66,
                    "lightning_chain": 64,
                    "laser_pierce": 62,
                }
                non_weapon_scores = {
                    "armor": 100,
                    "plating": 95,
                    "nanites": 90,
                    "servos": 55,
                    "gyros": 52,
                    "sensors": 40,
                    "reactor": 35,
                    "capacitor": 32,
                    "jets": 30,
                }
            best_idx = 0
            best_score = -1
            for idx, c in enumerate(choices):
                uid = c["id"]
                req = self.upgrade_req.get(uid)
                if req is not None:
                    score = held_weapon_scores.get(uid, 60) if req in held else 10
                else:
                    score = non_weapon_scores.get(uid, 20)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            return {"pick": best_idx}
        ids = [c["id"] for c in choices]
        for want in UPGRADE_PREF:
            if want in ids:
                return {"pick": ids.index(want)}
        return {"pick": 0}

    # ------------------------------------------------------------------ loot
    def weapon_score(self, kind: str, obs) -> float:
        """Context-aware value of a secondary weapon kind."""
        if FLAGS.get("laser_first", True):
            small = obs["zone"]["next_radius"] <= 70.0
            scores = {
                "laser": 30.0 if small else 40.0,
                "lightning": 38.0 if small else 30.0,
                "mortar": 10.0,
            }
            return scores.get(kind, 0.0)
        scores = {"laser": 30.0, "lightning": 22.0, "mortar": 18.0}
        if obs["zone"]["next_radius"] <= 70.0:
            scores["lightning"] += 10.0
            scores["mortar"] -= 6.0
        return scores.get(kind, 0.0)

    def _safe_bound(self, obs: mf.Observation, me) -> float:
        """Dynamic safe radius around `next_center`: expands beyond `0.8 * next_radius` while we have
        ample slack time before `shrink_end`, so spawning or fighting in the outer ring does not
        disable cover seeking, secondary weapon looting, or post-hit evasion."""
        zone = obs["zone"]
        inner = zone["next_radius"] * 0.8
        if not (FLAGS.get("w5_safe_bound", True) and me["in_zone"]):
            return inner
        t = obs["t"]
        rem_s = max(0.0, float(zone.get("shrink_end", t)) - t)
        walk_spd = max(1.0, float(me["stats"].get("walk_speed", 11.7)))
        # Keep a 14s safety cushion before shrink_end
        slack_m = max(0.0, (rem_s - 14.0) * walk_spd * 0.72)
        if slack_m <= 0.0:
            return inner
        cx, cy = zone["next_center"]
        zcx, zcy = zone["center"]
        c_shift = math.hypot(cx - zcx, cy - zcy)
        cur_safe = max(inner, float(zone["radius"]) - c_shift - 12.0)
        # Never expand beyond 1.45 * next_radius so late small circles stay bounded
        return min(cur_safe, inner + slack_m, float(zone["next_radius"]) * 1.45)

    def choose_take(self, obs: mf.Observation, me) -> dict:
        """Pick a pickup to walk to (self.loot_goal) and `take` it once in reach, or swap slot 1."""
        self.loot_goal = None
        self.loot_goal_id = None
        if not FLAGS.get("loot"):
            return {}
        held = {w["kind"] for w in me["weapons"] if w}
        use_lf = FLAGS.get("laser_first", True)
        want_kinds = set(LOOT_PREF)
        if FLAGS.get("mortar", True) and not use_lf:
            want_kinds.add("mortar")
        x, y, z = me["pos"]
        zone = obs["zone"]
        cx, cy = zone["next_center"]
        inner = self._safe_bound(obs, me)
        use_tri = FLAGS.get("weapon_tri", True)

        if None not in me["weapons"]:
            if not use_tri:
                return {}
            w1 = me["weapons"][1]
            if w1 is None or w1["kind"] == "gun":
                return {}
            if w1["kind"] == "laser" and w1.get("charge", 0) > 0:
                return {}
            t = obs["t"]
            if t - self.last_swap_t < 4.0:
                return {}
            vis_enemies = [m for m in obs["mechs"] if m.get("kind") == "mech"]
            min_ed = min((math.hypot(m["pos"][0] - x, m["pos"][1] - y) for m in vis_enemies), default=math.inf)
            active_combat = (self.threat is not None and t - self.threat[0] < THREAT_MEMORY_S) or min_ed < 50.0
            if active_combat:
                if not (me["hp"] > 0.5 * me["max_hp"] and min_ed >= 30.0):
                    return {}
                base_swap_d = self.reach
            else:
                base_swap_d = self.reach if vis_enemies else 14.0

            cur_score = self.weapon_score(w1["kind"], obs)
            best, best_d, best_score = None, math.inf, math.inf
            for p in obs["pickups"]:
                pid = p["id"]
                if pid in self.bad_pickups:
                    continue
                pk = p["kind"]
                if pk not in want_kinds or pk in held:
                    continue
                p_score = self.weapon_score(pk, obs)
                if p_score < cur_score + 6.0:
                    continue
                px, py, pz = p["pos"][0], p["pos"][1], p["pos"][2]
                d = math.hypot(px - x, py - y)
                max_swap_d = (
                    35.0
                    if (use_lf and pk == "laser" and zone["next_radius"] > 70.0 and not vis_enemies and not active_combat)
                    else (
                        30.0
                        if (FLAGS.get("w13_squad_endurance", True) and use_lf and pk == "lightning" and zone["next_radius"] <= 70.0 and not vis_enemies and not active_combat)
                        else base_swap_d
                    )
                )
                if d > max_swap_d:
                    continue
                if d <= self.reach:
                    if abs(pz - z) > 4.0:
                        continue
                else:
                    if use_lf and self.world is not None:
                        if abs(pz - self.world.ground_z(px, py)) > 2.5:
                            continue
                    elif abs(pz - z) > 4.0:
                        continue
                    if math.hypot(px - cx, py - cy) > inner:
                        continue
                rank = d - 0.5 * p_score
                if rank < best_score:
                    best, best_d, best_score = p, d, rank
            if best is None:
                return {}
            self.loot_goal = (best["pos"][0], best["pos"][1])
            self.loot_goal_id = best["id"]
            if best_d <= self.reach and abs(best["pos"][2] - z) <= 4.0:
                self.last_swap_t = t
                return {"take": best["id"], "drop": 1}
            return {}

        early_w = FLAGS.get("early_weapon", False)
        w6_loot = FLAGS.get("w6_loot_hunt", True)
        w8_kc = FLAGS.get("w8_kill_conversion", True)
        max_detour = 85.0 if early_w else (130.0 if (use_lf and w8_kc) else (85.0 if (use_lf and w6_loot) else (75.0 if use_lf else LOOT_DETOUR_M)))
        st = me["stats"]
        mr, mh = st.get("radius"), st.get("height")
        enemy_has_los: Optional[bool] = None
        lf_safe_detour: Optional[bool] = None

        best, best_d, best_rank = None, max_detour, math.inf
        for p in obs["pickups"]:
            pid = p["id"]
            if pid in self.bad_pickups:
                continue
            pk = p["kind"]
            if pk not in want_kinds or pk in held:
                continue
            px, py, pz = p["pos"][0], p["pos"][1], p["pos"][2]
            d = math.hypot(px - x, py - y)
            if early_w:
                if d >= max_detour:
                    continue
            else:
                if use_lf and w8_kc:
                    kind_max_d = 130.0 if pk == "laser" else (95.0 if pk == "lightning" else LOOT_DETOUR_M)
                elif use_lf and w6_loot:
                    kind_max_d = 85.0 if pk == "laser" else (62.0 if pk == "lightning" else LOOT_DETOUR_M)
                else:
                    kind_max_d = 75.0 if (use_lf and pk == "laser") else LOOT_DETOUR_M
                if d > kind_max_d:
                    continue
            rank = (d - 0.5 * self.weapon_score(pk, obs)) if use_tri else d
            if rank >= best_rank:
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
                elif use_lf and d > LOOT_DETOUR_M:
                    if lf_safe_detour is None:
                        t_now = obs["t"]
                        hp_need = 0.65 if w8_kc else 0.85
                        w13_se = FLAGS.get("w13_squad_endurance", True)
                        near_enemy_detour = (
                            any(
                                m["kind"] == "mech" and math.hypot(m["pos"][0] - x, m["pos"][1] - y) <= 65.0
                                for m in obs["mechs"]
                            )
                            if w13_se
                            else any(m["kind"] == "mech" for m in obs["mechs"])
                        )
                        lf_safe_detour = (
                            me["in_zone"]
                            and not self.late
                            and me["hp"] >= hp_need * me["max_hp"]
                            and not near_enemy_detour
                            and (self.threat is None or t_now - self.threat[0] >= THREAT_MEMORY_S)
                        )
                    if not lf_safe_detour:
                        continue
                    if FLAGS.get("w13_squad_endurance", True) and any(
                        m["kind"] == "mech" and math.hypot(m["pos"][0] - px, m["pos"][1] - py) <= 50.0
                        for m in obs["mechs"]
                    ):
                        continue
                if (early_w or use_lf) and self.world is not None:
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
                elif use_lf and d > LOOT_DETOUR_M and self.world is not None:
                    if not self.world.is_walkable(px, py, pz, radius=mr, height=mh):
                        continue
            best, best_d, best_rank = p, d, rank
        if best is None:
            return {}
        self.loot_goal = (best["pos"][0], best["pos"][1])
        self.loot_goal_id = best["id"]
        return {"take": best["id"]} if best_d <= self.reach and abs(best["pos"][2] - z) <= 4.0 else {}

    # ------------------------------------------------------------------ cover & movement
    def known_threats(self, obs: mf.Observation, me) -> List[Tuple[float, float, float]]:
        """Known enemy eye/muzzle 3D positions from visible/remembered mechs or recent damage bearing."""
        w6_esc = FLAGS.get("w6_crossfire_escape", True)
        has_mech_threat = w6_esc and (
            any(m.get("kind") == "mech" for m in obs["mechs"])
            or any(aid < 16 for aid in self.seen)
        )
        by_id = {}
        for m in obs["mechs"]:
            if m["kind"] == "mech" or (m["id"] in self.aggressors and not has_mech_threat):
                p = m["pos"]
                by_id[m["id"]] = (p[0], p[1], p[2] + 8.0)
        for aid, (_, p, _) in self.seen.items():
            if aid not in by_id and (aid < 16 or (aid in self.aggressors and not has_mech_threat)):
                by_id[aid] = (p[0], p[1], p[2] + 8.0)
        threats = list(by_id.values())
        if self.threat_id not in by_id and self.threat is not None and obs["t"] - self.threat[0] <= THREAT_MEMORY_S:
            if not (has_mech_threat and self.threat_id is not None and int(self.threat_id) >= 100):
                x, y, z = me["pos"]
                b = self.threat[1]
                tx = x + 60.0 * math.cos(b)
                ty = y + 60.0 * math.sin(b)
                tz = self.world.ground_z(tx, ty) if self.world is not None else z
                threats.append((tx, ty, tz + 8.0))
        return threats

    @staticmethod
    def _angular_span(px: float, py: float, pts: List[Tuple[float, float]]) -> float:
        """Minimum circular arc (radians) covering all 2D points in `pts` from (px, py)."""
        if len(pts) < 2:
            return 0.0
        angs = sorted(math.atan2(ey - py, ex - px) for ex, ey in pts)
        max_gap = TAU - (angs[-1] - angs[0])
        for i in range(len(angs) - 1):
            g = angs[i + 1] - angs[i]
            if g > max_gap:
                max_gap = g
        return TAU - max_gap

    def choose_cover(self, obs: mf.Observation, me, cx: float, cy: float, inner: float) -> Optional[Tuple[float, float]]:
        """Pick a nearby cover position inside `inner` that breaks LOS to known threats."""
        if not (FLAGS.get("cover") and self.world is not None and me["in_zone"]):
            self.cover_goal = None
            self.cover_goal_exposed = False
            return None
        x, y, z = me["pos"]
        if math.hypot(cx - x, cy - y) > inner:
            self.cover_goal = None
            self.cover_goal_exposed = False
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
        elif FLAGS.get("w7_squad_survival", True):
            vis_enemies = [m for m in obs["mechs"] if m.get("kind") == "mech"]
            if FLAGS.get("w12_bullet_evade", True):
                near_vis = [m for m in vis_enemies if math.hypot(m["pos"][0] - x, m["pos"][1] - y) <= 150.0]
                if len(near_vis) >= 2:
                    min_ef = min(m.get("hp", 1400.0) / max(1.0, m.get("max_hp", 1400.0)) for m in near_vis)
                    proactive = (me["hp"] < 0.96 * me["max_hp"]) and not (min_ef < 0.28 and me["hp"] > 0.45 * me["max_hp"])
                else:
                    proactive = len(vis_enemies) >= 2 and me["hp"] < 0.85 * me["max_hp"]
            else:
                proactive = len(vis_enemies) >= 2 and me["hp"] < 0.85 * me["max_hp"]

        if not (recent_hit or pursuing or proactive):
            self.cover_goal = None
            self.cover_goal_exposed = False
            return None

        threats = self.known_threats(obs, me)
        if pursuing and not threats:
            threats = self.cover_threats
        w4c = FLAGS.get("w4_combat", True)
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
        if w4c:
            if len(exposed_threats) > 1:
                exposed_threats.sort(key=lambda te: (te[0] - x) ** 2 + (te[1] - y) ** 2)
            exposed_threats = exposed_threats[:3]
            if len(unexposed_threats) > 2:
                unexposed_threats.sort(key=lambda te: (te[0] - x) ** 2 + (te[1] - y) ** 2)
                unexposed_threats = unexposed_threats[:2]
        else:
            if len(exposed_threats) > 6:
                exposed_threats.sort(key=lambda te: (te[0] - x) ** 2 + (te[1] - y) ** 2)
                exposed_threats = exposed_threats[:6]
            if len(unexposed_threats) > 6:
                unexposed_threats.sort(key=lambda te: (te[0] - x) ** 2 + (te[1] - y) ** 2)
                unexposed_threats = unexposed_threats[:6]
        self.cover_exposed = bool(exposed_threats)
        if pursuing:
            gx, gy = self.cover_goal
            gz = self.world.support_z(gx, gy, z)
            if math.hypot(gx - cx, gy - cy) <= inner:
                check_t = exposed_threats or threats
                if w4c and self.cover_goal_exposed and exposed_threats:
                    los_ok = any(not self.world.has_los((gx, gy, gz + eye_h), te) for te in exposed_threats)
                else:
                    los_ok = all(not self.world.has_los((gx, gy, gz + eye_h), te) for te in check_t)
                if los_ok and self.world.path_clear(me["pos"], (gx, gy, gz), radius=mr, height=mh):
                    self.cover_threats = threats
                    return self.cover_goal
        if not (recent_hit or proactive) or not exposed_threats or (self.cover_goal is None and t < self.cover_retry_at):
            self.cover_goal = None
            self.cover_goal_exposed = False
            return None

        base_angle = math.atan2(cy - y, cx - x)
        w6_esc = FLAGS.get("w6_crossfire_escape", True)
        radii = (12.0, 24.0, 36.0, 48.0) if (FLAGS.get("proactive_cover", False) or w6_esc) else (12.0, 24.0, 36.0)
        n_exposed = len(exposed_threats)
        n_unexposed = len(unexposed_threats)
        n_threats = n_exposed + n_unexposed

        best_pos: Optional[Tuple[float, float]] = None
        best_score = -math.inf
        best_partial = False

        def eval_cand(px: float, py: float, r: float) -> None:
            nonlocal best_pos, best_score, best_partial
            d_center = math.hypot(px - cx, py - cy)
            if inner > 3.0 and d_center > inner:
                return
            if w4c:
                max_possible = n_exposed * 100.0 + 40.0 + n_unexposed * 15.0 - r - d_center * 0.25
                if max_possible <= best_score:
                    return
                pz = self.world.support_z(px, py, z)
                if not self.world.is_walkable(px, py, pz, radius=mr, height=mh):
                    return
                cand_eye = (px, py, pz + eye_h)
                hides_closest = not self.world.has_los(cand_eye, exposed_threats[0])
                hidden_exp = (1 if hides_closest else 0) + sum(
                    1 for te in exposed_threats[1:] if not self.world.has_los(cand_eye, te)
                )
                if hidden_exp == 0:
                    return
                base_sc = hidden_exp * 100.0 + (40.0 if hides_closest else 0.0) - r - d_center * 0.25
                if base_sc + n_unexposed * 15.0 <= best_score:
                    return
                hidden_unexp = sum(1 for te in unexposed_threats if not self.world.has_los(cand_eye, te))
                score = base_sc + hidden_unexp * 15.0
                if score <= best_score:
                    return
                if not self.world.path_clear(me["pos"], (px, py, pz), radius=mr, height=mh):
                    return
                best_score = score
                best_pos = (px, py)
                best_partial = hidden_exp < n_exposed
            else:
                if n_threats * 100.0 - r - d_center * 0.25 <= best_score:
                    return
                pz = self.world.support_z(px, py, z)
                cand_eye = (px, py, pz + eye_h)
                if any(self.world.has_los(cand_eye, t_eye) for t_eye in exposed_threats):
                    return
                hidden = n_exposed + sum(1 for t_eye in unexposed_threats if not self.world.has_los(cand_eye, t_eye))
                score = hidden * 100.0 - r - d_center * 0.25
                if score <= best_score:
                    return
                if not self.world.is_walkable(px, py, pz, radius=mr, height=mh):
                    return
                if not self.world.path_clear(me["pos"], (px, py, pz), radius=mr, height=mh):
                    return
                best_score = score
                best_pos = (px, py)
                best_partial = False

        if FLAGS.get("corner_cover", True):
            for px, py in self.world.nearby_corners(x, y, max_r=48.0 if w6_esc else 42.0, limit=20):
                eval_cand(px, py, math.hypot(px - x, py - y))

        if best_pos is None:
            for r in radii:
                for k in range(12):
                    ang = base_angle + TAU * k / 12.0
                    eval_cand(x + r * math.cos(ang), y + r * math.sin(ang), r)

        self.cover_goal = best_pos
        self.cover_goal_exposed = best_partial if best_pos is not None else False
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

    def choose_evade(self, obs: mf.Observation, me, cx: float, cy: float, inner: float) -> Tuple[Optional[Tuple[float, float]], bool]:
        """Predict incoming mortar impacts and charging-laser aim rays; return (dodge_goal, should_boost)."""
        self.evade_impacts = []
        if not FLAGS.get("evade", True):
            return None, False

        mortars = [p for p in obs.get("projectiles", ()) if p.get("kind") == "mortar"]
        chargers = [m for m in obs.get("mechs", ()) if m.get("kind") == "mech" and m.get("charging")]
        if not mortars and not chargers:
            return None, False

        x, y, z = me["pos"]
        st = me["stats"]
        mr = st.get("radius", self.mech_r)
        mh = st.get("height", self.mech_h)
        eye_h = st.get("eye_height", 8.0)
        me_center = (x, y, z + mh * 0.5)

        # A. Predict mortar threats: (ix, iy, iz, t_imp, danger_r)
        mortar_threats: List[Tuple[float, float, float, float, float]] = []
        danger_r = self.mortar_splash * 1.3 + mr + 1.5
        g = max(1.0, self.mortar_g)
        for p in mortars:
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            zf = z
            disc = vz * vz + 2.0 * g * (pz - zf)
            if disc < 0.0:
                continue
            t_imp = (vz + math.sqrt(disc)) / g
            if t_imp <= 0.0 or t_imp > 3.8:
                continue
            ix, iy = px + vx * t_imp, py + vy * t_imp
            if self.world is not None:
                zf = self.world.support_z(ix, iy, z)
                disc2 = vz * vz + 2.0 * g * (pz - zf)
                if disc2 >= 0.0:
                    t_imp = (vz + math.sqrt(disc2)) / g
                    if t_imp <= 0.0 or t_imp > 3.8:
                        continue
                    ix, iy = px + vx * t_imp, py + vy * t_imp

            in_splash = math.hypot(ix - x, iy - y) < danger_r
            direct_hit = False
            v2 = vx * vx + vy * vy
            if v2 > 1e-3:
                tc = max(0.0, min(t_imp, ((x - px) * vx + (y - py) * vy) / v2))
                cx_m = px + vx * tc
                cy_m = py + vy * tc
                cz_m = pz + vz * tc - 0.5 * g * tc * tc
                if math.hypot(cx_m - x, cy_m - y) < mr + 2.0 and (z - 1.0) <= cz_m <= (z + mh + 1.5):
                    direct_hit = True
                    ix, iy, zf, t_imp = cx_m, cy_m, z, tc

            if in_splash or direct_hit:
                if not direct_hit and self.world is not None and not self.world.has_los((ix, iy, zf + 0.5), me_center):
                    continue
                mortar_threats.append((ix, iy, zf, t_imp, danger_r))
                self.evade_impacts.append((ix, iy, zf))

        # B. Detect charging-laser threats: (enemy_id, ex, ey, emz, dx, dy, left_x, left_y, perp_dist, pref_side)
        w5_laser = FLAGS.get("w5_laser_mastery", True)
        t_now = obs["t"]
        vx_me, vy_me = me["vel"][0], me["vel"][1]
        laser_threats: List[Tuple[int, float, float, float, float, float, float, float, float, float]] = []
        base_muzzle_h = self.cfg["mech"].get("muzzle_height", 7.0)
        for m in chargers:
            ep = m["pos"]
            aim = m.get("aim")
            if not aim:
                continue
            eyaw, epitch = aim[0], aim[1]
            ek = 1.0 + self.size_per_cube * m.get("cubes", 0)
            emz = ep[2] + base_muzzle_h * ek
            cp = math.cos(epitch)
            ux, uy, uz = cp * math.cos(eyaw), cp * math.sin(eyaw), math.sin(epitch)
            rx, ry, rz = x - ep[0], y - ep[1], (z + mh * 0.5) - emz
            s = rx * ux + ry * uy + rz * uz
            if s <= 0.0 or s > self.laser_range:
                continue
            cx_r, cy_r, cz_r = ep[0] + s * ux, ep[1] + s * uy, emz + s * uz
            perp_xy = math.hypot(x - cx_r, y - cy_r)
            if perp_xy > mr + 5.5 or abs((z + mh * 0.5) - cz_r) > mh * 0.5 + 4.5:
                continue
            if self.world is not None and not self.world.has_los((ep[0], ep[1], emz), me_center):
                continue
            dx, dy = math.cos(eyaw), math.sin(eyaw)
            left_x, left_y = -dy, dx
            left_dot = (x - ep[0]) * left_x + (y - ep[1]) * left_y
            mid = int(m["id"])
            if w5_laser:
                latched = self.laser_dodge_dir.get(mid)
                if latched is not None and latched[0] > t_now:
                    pref_side = latched[1]
                    if self.world is not None:
                        s_pos = (x + 12.0 * pref_side * left_x, y + 12.0 * pref_side * left_y)
                        s_neg = (x - 12.0 * pref_side * left_x, y - 12.0 * pref_side * left_y)
                        z_pos = self.world.support_z(s_pos[0], s_pos[1], z)
                        z_neg = self.world.support_z(s_neg[0], s_neg[1], z)
                        ok_pos = self.world.is_walkable(s_pos[0], s_pos[1], z_pos, radius=mr, height=mh) and self.world.path_clear(me["pos"], (s_pos[0], s_pos[1], z_pos), radius=mr, height=mh)
                        if not ok_pos:
                            ok_neg = self.world.is_walkable(s_neg[0], s_neg[1], z_neg, radius=mr, height=mh) and self.world.path_clear(me["pos"], (s_neg[0], s_neg[1], z_neg), radius=mr, height=mh)
                            if ok_neg:
                                pref_side = -pref_side
                    # Refresh latch while enemy continues charging
                    self.laser_dodge_dir[mid] = (t_now + 0.9, pref_side)
                else:
                    # Pick committed lateral escape direction:
                    # 1. If one perpendicular step breaks LOS or is clear while the other is blocked, prefer it
                    # 2. If already moving laterally (>2.5 m/s), keep momentum so we don't decelerate across beam
                    # 3. Otherwise dodge away from current beam centerline
                    v_dot = vx_me * left_x + vy_me * left_y
                    if abs(left_dot) > 1.8:
                        pref_side = 1.0 if left_dot >= 0.0 else -1.0
                    elif abs(v_dot) > 2.5:
                        pref_side = 1.0 if v_dot >= 0.0 else -1.0
                    else:
                        pref_side = 1.0 if left_dot >= 0.0 else -1.0
                    if self.world is not None:
                        s_pos = (x + 12.0 * pref_side * left_x, y + 12.0 * pref_side * left_y)
                        s_neg = (x - 12.0 * pref_side * left_x, y - 12.0 * pref_side * left_y)
                        z_pos = self.world.support_z(s_pos[0], s_pos[1], z)
                        z_neg = self.world.support_z(s_neg[0], s_neg[1], z)
                        ok_pos = self.world.is_walkable(s_pos[0], s_pos[1], z_pos, radius=mr, height=mh) and self.world.path_clear(me["pos"], (s_pos[0], s_pos[1], z_pos), radius=mr, height=mh)
                        ok_neg = self.world.is_walkable(s_neg[0], s_neg[1], z_neg, radius=mr, height=mh) and self.world.path_clear(me["pos"], (s_neg[0], s_neg[1], z_neg), radius=mr, height=mh)
                        if ok_neg and not ok_pos:
                            pref_side = -pref_side
                        elif ok_pos and ok_neg:
                            hide_pos = not self.world.has_los((ep[0], ep[1], emz), (s_pos[0], s_pos[1], z_pos + eye_h))
                            hide_neg = not self.world.has_los((ep[0], ep[1], emz), (s_neg[0], s_neg[1], z_neg + eye_h))
                            if hide_neg and not hide_pos:
                                pref_side = -pref_side
                    self.laser_dodge_dir[mid] = (t_now + 0.9, pref_side)
            else:
                pref_side = 1.0 if left_dot >= 0.0 else -1.0
            laser_threats.append((mid, ep[0], ep[1], emz, dx, dy, left_x, left_y, perp_xy, pref_side))

        if not mortar_threats and not laser_threats:
            return None, False

        # C. Sample & score dodge candidates
        angles: List[float] = []
        for ix, iy, _, _, _ in mortar_threats:
            if math.hypot(x - ix, y - iy) < 0.5:
                angles.append(math.atan2(cy - y, cx - x))
            else:
                angles.append(math.atan2(y - iy, x - ix))
        for _, ex, ey, _, _, _, left_x, left_y, _, pref_side in laser_threats:
            angles.append(math.atan2(pref_side * left_y, pref_side * left_x))
            angles.append(math.atan2(-pref_side * left_y, -pref_side * left_x))
        for k in range(8):
            angles.append(TAU * k / 8.0)

        step_r = 12.0
        zcx, zcy = obs["zone"]["center"]
        zrad = obs["zone"]["radius"]
        cur_zd = math.hypot(x - zcx, y - zcy)
        cur_dc = math.hypot(x - cx, y - cy)
        mv_x = math.cos(self.move_yaw) if self.move_yaw is not None else 0.0
        mv_y = math.sin(self.move_yaw) if self.move_yaw is not None else 0.0
        best_pt: Optional[Tuple[float, float]] = None
        best_score = -math.inf
        arc_pts: List[Tuple[float, float]] = []
        cur_span = 0.0
        if FLAGS.get("w4_combat", True):
            arc_pts = [
                (m["pos"][0], m["pos"][1])
                for m in obs["mechs"]
                if m.get("kind") == "mech" and math.hypot(m["pos"][0] - x, m["pos"][1] - y) <= 130.0
            ]
            if len(arc_pts) >= 2:
                cur_span = self._angular_span(x, y, arc_pts)

        for ang in angles:
            ca, sa = math.cos(ang), math.sin(ang)
            kx, ky = x + step_r * ca, y + step_r * sa
            cand_zd = math.hypot(kx - zcx, ky - zcy)
            if me["in_zone"]:
                if zrad > 6.0 and cand_zd > zrad - 3.0:
                    continue
                if self.late and math.hypot(kx - cx, ky - cy) > cur_dc + 2.0:
                    continue
            else:
                if cand_zd >= cur_zd:
                    continue

            if self.world is not None:
                kz = self.world.support_z(kx, ky, z)
                if not (self.world.is_walkable(kx, ky, kz, radius=mr, height=mh)
                        and self.world.path_clear(me["pos"], (kx, ky, kz), radius=mr, height=mh)):
                    continue
            else:
                kz = z

            score = 0.0
            for ix, iy, _, _, dr in mortar_threats:
                md = math.hypot(kx - ix, ky - iy)
                if md >= dr:
                    score += 120.0 + min(md - dr, 10.0)
                else:
                    score += md * 6.0
            for _, ex, ey, emz, _, _, left_x, left_y, _, pref_side in laser_threats:
                cur_dot = (x - ex) * left_x + (y - ey) * left_y
                cand_dot = (kx - ex) * left_x + (ky - ey) * left_y
                if w5_laser:
                    # Reward moving along the latched escape side; strongly penalize reversing across beam
                    latched_prog = (cand_dot - cur_dot) * pref_side
                    score += latched_prog * 9.0 + min(abs(cand_dot), 18.0) * 4.0
                    if latched_prog < -1.0:
                        score -= 95.0
                else:
                    score += min(abs(cand_dot), 18.0) * 8.0
                    if (cur_dot > 0.5 and cand_dot < 0.0) or (cur_dot < -0.5 and cand_dot > 0.0):
                        score -= 60.0
                if self.world is not None and not self.world.has_los((ex, ey, emz), (kx, ky, kz + eye_h)):
                    score += 110.0 if w5_laser else 90.0
            if cur_span > 2.618:
                cand_span = self._angular_span(kx, ky, arc_pts)
                score += (cur_span - cand_span) * 60.0
            d_center = math.hypot(kx - cx, ky - cy)
            if inner > 3.0 and d_center > inner:
                score -= 15.0
            score -= d_center * 0.05
            if self.move_yaw is not None:
                score += 4.0 * (ca * mv_x + sa * mv_y)

            if score > best_score:
                best_score = score
                best_pt = (kx, ky)

        if best_pt is None:
            return None, False

        imminent_mortar = any(t_imp <= 1.8 for _, _, _, t_imp, _ in mortar_threats)
        tight_laser = any(pd <= mr + (4.5 if w5_laser else 3.5) for *_, pd, _ in laser_threats)
        should_boost = bool(me.get("grounded", True)) and (me["energy"] > 15.0) and (imminent_mortar or tight_laser)
        return best_pt, should_boost

    def choose_bullet_dodge(self, obs: mf.Observation, me, cx: float, cy: float, inner: float) -> Optional[Tuple[float, float]]:
        """Sidestep imminent incoming gun/NPC projectiles perpendicular to flight vector without dropping laser charge."""
        if not (FLAGS.get("w12_bullet_evade", True) and me["in_zone"] and inner > 3.0):
            return None
        projs = obs.get("projectiles", ())
        if not projs:
            return None
        x, y, z = me["pos"]
        st = me["stats"]
        mr = st.get("radius", self.mech_r)
        mh = st.get("height", self.mech_h)
        me_center = (x, y, z + mh * 0.5)

        has_mech_t = (
            any(m.get("kind") == "mech" for m in obs["mechs"])
            or any(aid < 16 for aid in self.seen)
        )
        soonest = None
        best_tc = 0.72
        for p in projs:
            pk = p.get("kind")
            if pk not in ("gun", "npc") or p.get("owner") == me["id"]:
                continue
            if pk == "npc" and has_mech_t:
                continue
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            v2 = vx * vx + vy * vy
            if v2 < 100.0:
                continue
            tc = ((x - px) * vx + (y - py) * vy) / v2
            if tc < 0.03 or tc >= best_tc:
                continue
            cx_p = px + vx * tc
            cy_p = py + vy * tc
            cz_p = pz + vz * tc
            if math.hypot(x - cx_p, y - cy_p) > mr + 1.5:
                continue
            if not (z - 0.8 <= cz_p <= z + mh + 0.8):
                continue
            if self.world is not None and not self.world.has_los((px, py, pz), me_center):
                continue
            best_tc = tc
            soonest = (vx, vy, cx_p, cy_p, math.sqrt(v2))

        if soonest is None:
            return None
        vx, vy, cx_p, cy_p, v_xy = soonest
        lx, ly = -vy / v_xy, vx / v_xy
        dot = (x - cx_p) * lx + (y - cy_p) * ly
        if abs(dot) < 0.6:
            pref = 1.0 if self.unstick_side >= 0 else -1.0
        else:
            pref = 1.0 if dot >= 0.0 else -1.0
        for side in (pref, -pref):
            sx = x + 8.0 * side * lx
            sy = y + 8.0 * side * ly
            if math.hypot(sx - cx, sy - cy) > inner:
                continue
            if self.world is not None:
                sz = self.world.support_z(sx, sy, z)
                if not (self.world.is_walkable(sx, sy, sz, radius=mr, height=mh)
                        and self.world.path_clear(me["pos"], (sx, sy, sz), radius=mr, height=mh)):
                    continue
            return (sx, sy)
        return None

    def choose_crate(self, obs: mf.Observation, me, cx: float, cy: float, inner: float) -> Optional[dict]:
        """Select a nearby ground crate to farm when zero enemies are around."""
        if not (FLAGS.get("farm_crate", True) and me["in_zone"] and not self.late
                and not self.evading and self.cover_goal is None
                and me["hp"] >= me["max_hp"] - 1.0):
            return None
        t = obs["t"]
        cubes = me.get("cubes", 0)
        w7_sq = FLAGS.get("w7_squad_survival", True)
        if w7_sq:
            if FLAGS.get("w8_kill_conversion", True):
                if not (cubes == 0 or (cubes < 6 and t < 110.0) or (None in me["weapons"] and cubes < 8)):
                    return None
            elif not (cubes == 0 or (cubes < 4 and t < 90.0)):
                return None
        else:
            if not (cubes == 0 or (None in me["weapons"] and cubes < 2 and t < 90.0)):
                return None
        if any(m.get("kind") == "mech" for m in obs["mechs"]):
            return None
        if any(aid < 16 for aid in self.seen):
            return None
        if self.threat is not None and t - self.threat[0] < THREAT_MEMORY_S:
            return None
        if self.aggressors:
            return None

        x, y, z = me["pos"]
        npcs = [m for m in obs["mechs"] if m.get("kind") == "npc"]
        for m in npcs:
            if math.hypot(m["pos"][0] - x, m["pos"][1] - y) <= 48.0:
                return None

        st = me["stats"]
        mr, mh = st.get("radius"), st.get("height")
        best = None
        best_d = 55.0
        for c in obs.get("crates", ()):
            cid = c["id"]
            if cid in self.bad_crates:
                continue
            px, py, pz = c["pos"][0], c["pos"][1], c["pos"][2]
            d_xy = math.hypot(px - x, py - y)
            if d_xy > best_d:
                continue
            if math.hypot(px - cx, py - cy) > inner:
                continue
            if abs(pz - z) > 4.0:
                continue
            if self.world is not None:
                if abs(pz - self.world.ground_z(px, py)) > 2.0:
                    continue
                if not self.world.is_walkable(px, py, pz, radius=mr, height=mh):
                    continue
            if any(math.hypot(m["pos"][0] - px, m["pos"][1] - py) <= 45.0 for m in npcs):
                continue
            best = c
            best_d = d_xy

        if best is None:
            self.crate_track = None
            return None

        cid = best["id"]
        chp = best.get("hp", 300.0)
        if self.crate_track is None or self.crate_track[0] != cid:
            self.crate_track = (cid, t, chp)
        else:
            _, start_t, start_hp = self.crate_track
            if chp < start_hp:
                self.crate_track = (cid, t, chp)
            elif t - start_t >= 3.5:
                self.bad_crates[cid] = t + 30.0
                self.crate_track = None
                return None
        return best

    def choose_drop(self, obs: mf.Observation, me, cx: float, cy: float, inner: float) -> Optional[Tuple[dict, float]]:
        """Select a nearby, safe, winnable supply drop inside `inner`."""
        if not (FLAGS.get("drops", True) and me["in_zone"] and not self.late):
            return None
        agg = FLAGS.get("aggressive_drops", True)
        x, y, _ = me["pos"]
        walk_speed = max(1.0, me["stats"]["walk_speed"])
        boost_speed = max(1.0, me["stats"].get("boost_speed", walk_speed * 1.65))
        vis_mechs = [m for m in obs["mechs"] if m.get("kind") == "mech"]
        vis_ids = {m["id"] for m in vis_mechs}

        best: Optional[Tuple[dict, float]] = None
        best_d = math.inf
        for dr in obs.get("drops", ()):
            dx, dy, _ = dr["pos"]
            rad = float(dr.get("radius", 8.0))
            lands_in = float(dr.get("lands_in", 0.0))
            dist = math.hypot(dx - x, dy - y)
            if agg:
                if dist >= best_d or dist > 150.0:
                    continue
                if math.hypot(dx - cx, dy - cy) > inner:
                    continue
                eta_walk = max(0.0, dist - max(2.0, rad - 2.0)) / walk_speed
                if lands_in > eta_walk + 2.0:
                    continue
                if dr.get("holder") not in (None, me["id"]):
                    rem_cap = max(0.0, float(dr.get("capture_s", 4.0)) - float(dr.get("progress", 0.0)))
                    eta_boost = max(0.0, dist - rad) / boost_speed
                    if rem_cap <= eta_boost + 0.3:
                        continue
            else:
                w11_j = FLAGS.get("w11_combat_juke", True)
                max_dr_d = 135.0 if w11_j else 85.0
                max_dr_t = 12.0 if w11_j else 8.0
                if dist >= best_d or dist > max_dr_d or lands_in > max_dr_t:
                    continue
                if math.hypot(dx - cx, dy - cy) > inner:
                    continue
                if dr.get("holder") not in (None, me["id"]):
                    rem_cap = max(0.0, float(dr.get("capture_s", 4.0)) - float(dr.get("progress", 0.0)))
                    eta = max(0.0, dist - rad) / walk_speed
                    if rem_cap <= eta + 0.3:
                        continue
            near_ids = set()
            for m in vis_mechs:
                if math.hypot(m["pos"][0] - dx, m["pos"][1] - dy) <= 35.0:
                    near_ids.add(m["id"])
            for aid, (_, spos, _) in self.seen.items():
                if aid < 16 and aid not in vis_ids:
                    if math.hypot(spos[0] - dx, spos[1] - dy) <= 35.0:
                        near_ids.add(aid)
            enemies_near = len(near_ids)
            finishing_own = (
                agg
                and dr.get("holder") == me["id"]
                and (float(dr.get("capture_s", 4.0)) - float(dr.get("progress", 0.0))) <= 2.5
                and me["hp"] > 350.0
            )
            if not finishing_own:
                if enemies_near > 1:
                    continue
                hp_req = 0.45 if agg else 0.35
                if enemies_near == 1 and me["hp"] < hp_req * me["max_hp"]:
                    continue
            best = (dr, dist)
            best_d = dist
        return best

    def choose_npc_farm(self, obs: mf.Observation, me, cx: float, cy: float, inner: float) -> Optional[dict]:
        """Select a nearby ground-level NPC to farm for cubes when completely safe from enemy mechs."""
        if not (FLAGS.get("farm_npc", True) and me["in_zone"] and not self.late
                and not self.evading and self.cover_goal is None
                and me["hp"] >= 0.80 * me["max_hp"] and me.get("cubes", 0) < 6):
            return None
        t = obs["t"]
        if any(m.get("kind") == "mech" for m in obs["mechs"]):
            return None
        if any(aid < 16 for aid in self.seen):
            return None
        if self.threat is not None and t - self.threat[0] < THREAT_MEMORY_S:
            return None
        if any(aid < 16 for aid in self.aggressors):
            return None
        x, y, z = me["pos"]
        src = (x, y, z + me["stats"]["muzzle_height"])
        best = None
        best_d = 45.0
        for m in obs["mechs"]:
            if m.get("kind") != "npc":
                continue
            nx, ny, nz = m["pos"]
            d_xy = math.hypot(nx - x, ny - y)
            if d_xy > best_d:
                continue
            if math.hypot(nx - cx, ny - cy) > inner:
                continue
            if self.world is not None:
                if abs(nz - self.world.ground_z(nx, ny)) > 2.5:
                    continue
                tp, _, _ = self.body(m)
                if not self.world.has_los(src, tp):
                    continue
            elif abs(nz - z) > 4.0:
                continue
            best = m
            best_d = d_xy
        return best

    def movement(self, obs: mf.Observation, me) -> dict:
        x, y = me["pos"][0], me["pos"][1]
        t = obs["t"]
        self.last_move_t = t
        zone = obs["zone"]
        cx, cy = zone["next_center"]
        d = math.hypot(cx - x, cy - y)
        strict_inner = zone["next_radius"] * 0.8     # comfortably inside the next circle
        inner = self._safe_bound(obs, me)
        # NOTE: Do not widen settle to 0.60 * next_radius by default (coupled to FLAGS['kite']);
        # standing still at 60% instead of rotating to 50% early hurt placement (e.g. seed 47).
        if FLAGS.get("kite", False) and zone["next_radius"] > 100.0 and obs.get("alive", 16) > 4:
            settle = max(zone["next_radius"] * 0.60, 3.0)
        elif (FLAGS.get("early_survival", True)
              and me["in_zone"]
              and t < zone.get("shrink_start", 0.0) - 10.0
              and obs.get("alive", 16) >= 10):
            settle = max(zone["next_radius"] * 0.78, 3.0)
        else:
            settle = max(zone["next_radius"] * 0.5, 3.0)

        w4c = FLAGS.get("w4_combat", True)
        w5_nav = FLAGS.get("w5_nav_mastery", True)
        w6_esc = FLAGS.get("w6_crossfire_escape", True)
        w6_loot = FLAGS.get("w6_loot_hunt", True)
        w6_end = FLAGS.get("w6_endgame_mastery", True)
        w8_kc = FLAGS.get("w8_kill_conversion", True)
        w9_sh = FLAGS.get("w9_squad_hunter", False)
        if w6_end and zone["next_radius"] <= 35.0:
            settle = max(zone["next_radius"] * 0.32, 2.0)
        agg_drops = FLAGS.get("aggressive_drops", True)
        late_r = max(strict_inner, 4.0) if zone["next_radius"] == 0 else strict_inner
        self.late = d > late_r and (d - late_r) > 0.8 * me["stats"]["walk_speed"] * max(zone["shrink_end"] - t, 0.0)
        self.evading = False
        self.crate_target = None
        self.npc_farm_target = None
        self.active_drop_holder = None
        kite_boost = False
        cover_boost = False
        drop_boost = False
        loot_boost = False
        is_arc_step = False
        goal = None
        evade_goal, evade_boost = self.choose_evade(obs, me, cx, cy, inner)
        if evade_goal is not None:
            self.evading = True
            goal = evade_goal
            kite_boost = evade_boost
        elif not me["in_zone"] or self.late or d > inner:
            self.cover_goal = None
            self.cover_goal_exposed = False
            goal = (cx, cy)
        else:
            drop_pick = self.choose_drop(obs, me, cx, cy, inner)
            holding_drop = (
                drop_pick is not None
                and drop_pick[1] <= float(drop_pick[0].get("radius", 8.0)) - 1.5
                and float(drop_pick[0].get("lands_in", 0.0)) <= 0.5
                and drop_pick[0].get("holder") == me["id"]
                and float(drop_pick[0].get("progress", 0.0)) >= 1.0
                and me["hp"] > 0.35 * me["max_hp"]
            )
            if (agg_drops and drop_pick is not None
                    and drop_pick[0].get("holder") == me["id"]
                    and (float(drop_pick[0].get("capture_s", 4.0)) - float(drop_pick[0].get("progress", 0.0))) <= 2.5
                    and me["hp"] > 350.0):
                holding_drop = True
            if holding_drop:
                self.cover_goal = None
                self.cover_goal_exposed = False
                goal = None
            else:
                cover = self.choose_cover(obs, me, cx, cy, inner)
                if cover is not None:
                    dist_c = math.hypot(cover[0] - x, cover[1] - y)
                    if ((w9_sh or FLAGS.get("w12_bullet_evade", True)) and self.cover_goal is not None
                            and not self.cover_exposed
                            and me["hp"] < (0.72 if FLAGS.get("w12_bullet_evade", True) else 0.55) * me["max_hp"]
                            and (t - self.last_damaged_t) >= 0.8
                            and not any(m.get("kind") == "mech" and math.hypot(m["pos"][0] - x, m["pos"][1] - y) < 32.0 for m in obs["mechs"])):
                        # Behind cover with low HP: extend cover_until so we stay behind the wall
                        # long enough to trigger 5s out-of-combat HP regeneration (+45..72 HP/s)!
                        self.cover_until = max(self.cover_until, t + 1.5)
                    if dist_c <= COVER_REACH_M and (not self.cover_exposed or (w4c and self.cover_goal_exposed)):
                        goal = None
                    else:
                        goal = cover
                        if (w4c and dist_c > COVER_REACH_M
                                and (self.threat is not None and t - self.threat[0] < THREAT_MEMORY_S)
                                and self.cover_exposed
                                and (me["hp"] < 0.70 * me["max_hp"] or (w6_esc and len([m for m in obs["mechs"] if m.get("kind") == "mech"]) >= 2))
                                and me["energy"] > 45.0):
                            cover_boost = True
                else:
                    hold_drop_ring = False
                    kite_goal, kite_td = self.choose_kite(obs, me, cx, cy, inner)
                    if kite_goal is not None:
                        goal = kite_goal
                        if (self.healing or kite_td < 45.0) and me["energy"] > 35.0:
                            kite_boost = True
                    elif (self.threat and t - self.threat[0] < THREAT_MEMORY_S
                          and not (w6_esc and self.threat_id is not None and int(self.threat_id) >= 100
                                   and (any(m.get("kind") == "mech" for m in obs["mechs"])
                                        or any(aid < 16 for aid in self.seen)))):
                        # Post-hit strafe / pinch escape / anti-rusher backpedal when no cover building is nearby.
                        bearing = self.threat[1]
                        if (FLAGS.get("combat_strafe", False) or FLAGS.get("w11_combat_juke", False) or FLAGS.get("w12_bullet_evade", True)) and t >= self.strafe_flip_t:
                            self.unstick_side = -self.unstick_side
                            self.strafe_flip_t = t + 0.45
                        arc_goal = None
                        if w4c and inner > 3:
                            arc_pts = [
                                (m["pos"][0], m["pos"][1])
                                for m in obs["mechs"]
                                if m.get("kind") == "mech" and math.hypot(m["pos"][0] - x, m["pos"][1] - y) <= 130.0
                            ]
                            if len(arc_pts) >= 2:
                                cur_span = self._angular_span(x, y, arc_pts)
                                if cur_span > (2.10 if w6_esc else 2.618):
                                    mr, mh = me["stats"].get("radius"), me["stats"].get("height")
                                    step_arc = 14.0 if w6_esc else 8.0
                                    best_span = cur_span
                                    for k_ang in range(12 if w6_esc else 8):
                                        ang = TAU * k_ang / (12.0 if w6_esc else 8.0)
                                        sx, sy = x + step_arc * math.cos(ang), y + step_arc * math.sin(ang)
                                        if math.hypot(sx - cx, sy - cy) > inner:
                                            continue
                                        if self.world is not None:
                                            sz = self.world.support_z(sx, sy, me["pos"][2])
                                            if not (self.world.is_walkable(sx, sy, sz, radius=mr, height=mh)
                                                    and self.world.path_clear(me["pos"], (sx, sy, sz), radius=mr, height=mh)):
                                                continue
                                        sp = self._angular_span(sx, sy, arc_pts)
                                        if sp < best_span:
                                            best_span = sp
                                            arc_goal = (sx, sy)
                                    if arc_goal is not None and w6_esc and me["energy"] > 45.0:
                                        cover_boost = True
                            elif w6_esc and len(arc_pts) == 1:
                                # 1v1 close-quarters rusher (< 42 m) when we do not hold lightning: diagonal backpedal
                                # (Skip backpedaling if w8_kc and enemy is critically low HP and we have HP advantage!)
                                vis_1v1 = next((m for m in obs["mechs"] if m.get("kind") == "mech" and math.hypot(m["pos"][0] - x, m["pos"][1] - y) <= 130.0), None)
                                enemy_low_1v1 = (
                                    w8_kc
                                    and vis_1v1 is not None
                                    and vis_1v1.get("hp", 1400.0) < 0.45 * vis_1v1.get("max_hp", 1400.0)
                                    and me["hp"] > max(0.35 * me["max_hp"], vis_1v1.get("hp", 1400.0))
                                )
                                held_k = {w["kind"] for w in me["weapons"] if w}
                                ed_1v1 = math.hypot(arc_pts[0][0] - x, arc_pts[0][1] - y)
                                if not enemy_low_1v1 and ed_1v1 < 42.0 and "lightning" not in held_k:
                                    away_a = math.atan2(y - arc_pts[0][1], x - arc_pts[0][0])
                                    mr, mh = me["stats"].get("radius"), me["stats"].get("height")
                                    for da in (0.55 * self.unstick_side, -0.55 * self.unstick_side, 0.0, 1.0 * self.unstick_side, -1.0 * self.unstick_side):
                                        ang = away_a + da
                                        sx, sy = x + 12.0 * math.cos(ang), y + 12.0 * math.sin(ang)
                                        if math.hypot(sx - cx, sy - cy) > inner:
                                            continue
                                        if self.world is not None:
                                            sz = self.world.support_z(sx, sy, me["pos"][2])
                                            if not (self.world.is_walkable(sx, sy, sz, radius=mr, height=mh)
                                                    and self.world.path_clear(me["pos"], (sx, sy, sz), radius=mr, height=mh)):
                                                continue
                                        arc_goal = (sx, sy)
                                        if ed_1v1 < 30.0 and me["energy"] > 50.0:
                                            cover_boost = True
                                        break
                        if arc_goal is not None:
                            goal = arc_goal
                            is_arc_step = True
                        else:
                            bullet_goal = self.choose_bullet_dodge(obs, me, cx, cy, inner)
                            if bullet_goal is not None:
                                goal = bullet_goal
                                is_arc_step = True
                            else:
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
                        bullet_goal = self.choose_bullet_dodge(obs, me, cx, cy, inner)
                        strafe_goal, strafe_boost = (None, False) if bullet_goal is not None else self.choose_combat_strafe(obs, me, cx, cy, inner)
                        if bullet_goal is not None:
                            goal = bullet_goal
                            is_arc_step = True
                        elif strafe_goal is not None:
                            goal = strafe_goal
                            kite_boost = strafe_boost
                        elif drop_pick is not None:
                            dr, drop_d = drop_pick
                            dr_rad = float(dr.get("radius", 8.0))
                            holder = dr.get("holder")
                            if agg_drops and holder not in (None, me["id"]):
                                self.active_drop_holder = holder
                            if drop_d <= dr_rad - 2.0:
                                goal = None
                                hold_drop_ring = True
                            else:
                                goal = (dr["pos"][0], dr["pos"][1])
                                if agg_drops and drop_d > dr_rad and me["energy"] >= 35.0:
                                    if holder not in (None, me["id"]):
                                        if me["hp"] >= 0.50 * me["max_hp"]:
                                            drop_boost = True
                                    else:
                                        lands_in = float(dr.get("lands_in", 0.0))
                                        walk_speed = max(1.0, me["stats"]["walk_speed"])
                                        eta_walk = max(0.0, drop_d - max(2.0, dr_rad - 2.0)) / walk_speed
                                        if lands_in <= 1.0 and eta_walk > 0.8:
                                            dx_dr, dy_dr = dr["pos"][0], dr["pos"][1]
                                            vis_m = [m for m in obs["mechs"] if m.get("kind") == "mech"]
                                            vis_ids_dr = {m["id"] for m in vis_m}
                                            en_near = any(
                                                math.hypot(m["pos"][0] - dx_dr, m["pos"][1] - dy_dr) <= 35.0
                                                for m in vis_m
                                            ) or any(
                                                aid < 16 and aid not in vis_ids_dr
                                                and math.hypot(spos[0] - dx_dr, spos[1] - dy_dr) <= 35.0
                                                for aid, (_, spos, _) in self.seen.items()
                                            )
                                            if not en_near:
                                                drop_boost = True
                        elif self.loot_goal is not None:
                            goal = self.loot_goal
                        elif FLAGS.get("cubes", True) and me.get("cubes", 0) < self.cfg["cubes"].get("stat_cap", 10):
                            mz = me["pos"][2]
                            uncontested_cubes = (
                                FLAGS.get("farm_npc", True)
                                and not any(m.get("kind") == "mech" for m in obs["mechs"])
                                and not any(aid < 16 for aid in self.seen)
                            )
                            best_cd = (50.0 if w8_kc else 35.0) if uncontested_cubes else (26.0 if w8_kc else 20.0)
                            for c in obs["cubes"]:
                                cp = c["pos"]
                                if self.world is not None and w8_kc:
                                    if abs(cp[2] - self.world.ground_z(cp[0], cp[1])) > 2.5:
                                        continue
                                elif abs(cp[2] - mz) > 4.0:
                                    continue
                                cd = math.hypot(cp[0] - x, cp[1] - y)
                                if cd <= best_cd and math.hypot(cp[0] - cx, cp[1] - cy) <= inner:
                                    best_cd = cd
                                    goal = (cp[0], cp[1])
                    if (w8_kc and goal is None and cover is None and not hold_drop_ring
                            and me["hp"] >= 0.45 * me["max_hp"]):
                        # Kill-conversion chase: if a single wounded enemy mech (< 55% HP) is within 110 m
                        # (and no other enemy mech is in self.seen), press forward to finish the kill before
                        # it breaks LOS and regenerates!
                        vis_m_kc = [m for m in obs["mechs"] if m.get("kind") == "mech"]
                        seen_mech_ids = [aid for aid in self.seen if aid < 16]
                        if len(vis_m_kc) == 1 and len(seen_mech_ids) <= 1:
                            em_kc = vis_m_kc[0]
                            ehp_kc = em_kc.get("hp", 1400.0)
                            ehp_f = ehp_kc / max(1.0, em_kc.get("max_hp", 1400.0))
                            ed_kc = math.hypot(em_kc["pos"][0] - x, em_kc["pos"][1] - y)
                            has_lt_kc = any(w and w["kind"] == "lightning" for w in me["weapons"])
                            has_ls_kc = any(w and w["kind"] == "laser" for w in me["weapons"])
                            want_close = 24.0 if has_lt_kc else 42.0
                            ep_kc = em_kc["pos"]
                            elev_ok = (self.world is None or abs(ep_kc[2] - self.world.ground_z(ep_kc[0], ep_kc[1])) <= 2.5)
                            # Under w9_squad_hunter: if we hold lightning and have >= 70% HP & comparable/better HP,
                            # rush an isolated 1v1 enemy within 90 m (not actively charging a laser) to get inside
                            # 42 m lightning range (240 DPS + laser short-circuit!) and melt it!
                            lt_rush = (w9_sh and has_lt_kc and me["hp"] >= max(0.70 * me["max_hp"], ehp_kc * 0.85)
                                       and not em_kc.get("charging") and 24.0 < ed_kc <= 90.0)
                            # Also under w9_squad_hunter: if we hold laser + HP advantage, maintain pressure on < 70% HP
                            hp_thresh = 0.70 if (w9_sh and has_ls_kc) else 0.55
                            if ((lt_rush or (ehp_f < hp_thresh and me["hp"] > ehp_kc)) and elev_ok
                                    and want_close < ed_kc <= 115.0
                                    and math.hypot(ep_kc[0] - cx, ep_kc[1] - cy) <= inner):
                                goal = (ep_kc[0], ep_kc[1])
                                if (ehp_f < 0.35 or (lt_rush and ed_kc > 42.0)) and me["energy"] > 50.0 and me.get("grounded", True) and (has_lt_kc or ed_kc <= 75.0):
                                    cover_boost = True
                        elif not vis_m_kc and len(seen_mech_ids) == 1 and me["hp"] >= 0.60 * me["max_hp"]:
                            # If a single recently wounded enemy (< 55% HP under w9_sh, else < 45%) ducked behind a corner
                            # into self.seen (< 4.5s ago), chase its last seen position to finish the kill before it regenerates!
                            aid = seen_mech_ids[0]
                            st_t, spos, _ = self.seen[aid]
                            hp_info = self.seen_hp.get(aid)
                            mem_hp_th = 0.55 if w9_sh else 0.45
                            if ((t - st_t) <= 4.5 and hp_info is not None and (t - hp_info[0]) <= 4.5
                                    and hp_info[1] < mem_hp_th):
                                wd = math.hypot(spos[0] - x, spos[1] - y)
                                elev_ok = (self.world is None or abs(spos[2] - self.world.ground_z(spos[0], spos[1])) <= 2.5)
                                if elev_ok and 8.0 < wd < 95.0 and math.hypot(spos[0] - cx, spos[1] - cy) <= inner:
                                    goal = (spos[0], spos[1])
                    if goal is None and cover is None and not hold_drop_ring:
                        crate = self.choose_crate(obs, me, cx, cy, inner)
                        if crate is not None:
                            self.crate_target = crate
                            cpx, cpy = crate["pos"][0], crate["pos"][1]
                            if math.hypot(cpx - x, cpy - y) > 5.0:
                                goal = (cpx, cpy)
                            else:
                                goal = None
                        else:
                            npc_t = self.choose_npc_farm(obs, me, cx, cy, inner)
                            if npc_t is not None:
                                self.npc_farm_target = npc_t
                                nx, ny = npc_t["pos"][0], npc_t["pos"][1]
                                if math.hypot(nx - x, ny - y) > 22.0:
                                    goal = (nx, ny)
                    if (goal is None and cover is None and not hold_drop_ring
                            and self.crate_target is None and self.npc_farm_target is None):
                        if w6_esc and not any(m.get("kind") == "mech" for m in obs["mechs"]) and any(aid >= 100 for aid in self.aggressors):
                            # Being chipped by an aggroed NPC while not farming it: step away toward next_center
                            goal = (cx, cy)
                        elif d > settle:
                            goal = (cx, cy)
                        elif (w9_sh and None in me["weapons"] and t >= 12.0 and t < 150.0
                              and me["in_zone"] and not self.late
                              and me["hp"] >= 0.60 * me["max_hp"]
                              and not any(m.get("kind") == "mech" for m in obs["mechs"])
                              and not any(aid < 16 for aid in self.seen)
                              and (self.last_near_sound is None or t - self.last_near_sound[0] > 2.5)
                              and (self.threat is None or t - self.threat[0] >= THREAT_MEMORY_S)):
                            # Unarmed (gun-only): actively patrol walkable points inside settle to discover floor weapons & crates!
                            pr_p = min(settle * 0.65, max(25.0, zone["next_radius"] * 0.45))
                            if pr_p >= 20.0:
                                reach_p = min(18.0, pr_p * 0.6)
                                px_p = cx + pr_p * math.cos(self.patrol_angle)
                                py_p = cy + pr_p * math.sin(self.patrol_angle)
                                need_new = (t >= self.patrol_until or math.hypot(x - px_p, y - py_p) < reach_p
                                            or (self.world is not None and not self.world.is_walkable(px_p, py_p)))
                                if need_new:
                                    for _ in range(5):
                                        self.patrol_angle = wrap(self.patrol_angle + 1.35)
                                        px_p = cx + pr_p * math.cos(self.patrol_angle)
                                        py_p = cy + pr_p * math.sin(self.patrol_angle)
                                        if self.world is None or self.world.is_walkable(px_p, py_p):
                                            self.patrol_until = t + 7.0
                                            break
                                if self.world is None or self.world.is_walkable(px_p, py_p):
                                    goal = (px_p, py_p)
                        elif (FLAGS.get("w7_squad_survival", True)
                              and self.world is not None
                              and hasattr(self.world, "nearby_canopies")
                              and not any(m.get("kind") == "mech" for m in obs["mechs"])
                              and (any(aid < 16 for aid in self.seen)
                                   or (self.last_near_sound is not None and t - self.last_near_sound[0] <= 2.5))
                              and (self.threat is None or t - self.threat[0] >= THREAT_MEMORY_S)):
                            mr_c, mh_c = me["stats"].get("radius"), me["stats"].get("height")
                            mz_c = me["pos"][2]
                            for cnx, cny in self.world.nearby_canopies(x, y, max_r=22.0, limit=4):
                                if math.hypot(cnx - cx, cny - cy) <= settle:
                                    cnz = self.world.ground_z(cnx, cny)
                                    if abs(mz_c - cnz) > 2.5:
                                        continue
                                    dc_can = math.hypot(cnx - x, cny - y)
                                    if dc_can <= 2.2:
                                        break
                                    if self.world.path_clear(me["pos"], (cnx, cny, cnz), radius=mr_c, height=mh_c):
                                        goal = (cnx, cny)
                                        break

        if goal is None:  # hold still: silent, and the gun is steadier
            self.move_yaw = None
            self.stuck_s = 0.0
            self.unstick_until = -1.0
            self.progress_anchor = None
            self.nav_path = None
            self.nav_goal = None
            return {"move": [0.0, 0.0], "boost": False, "jump": False}

        if (FLAGS.get("early_weapon", False) or w8_kc) and self.loot_goal is not None and goal == self.loot_goal:
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

        steer_target = goal
        if (FLAGS.get("astar_nav", False) and self.world is not None
                and not self.evading
                and not is_arc_step
                and t >= self.unstick_until
                and math.hypot(goal[0] - x, goal[1] - y) > 10.0):
            if self.world.nav_line_clear((x, y), goal):
                self.nav_path = None
                self.nav_goal = None
            else:
                if w5_nav and self.stuck_s >= 0.45 and self.nav_path and (t - self.last_wp_pop_t) >= 0.9:
                    self.last_wp_pop_t = t
                    wx0, wy0 = self.nav_path[0]
                    self.bad_waypoints[(int(wx0 // 6.0), int(wy0 // 6.0))] = t + 12.0
                    self.nav_path.pop(0)
                    if not self.nav_path:
                        self.nav_until = -1.0
                replan = (
                    self.nav_goal is None
                    or math.hypot(goal[0] - self.nav_goal[0], goal[1] - self.nav_goal[1]) > 12.0
                    or (not self.nav_path and t >= self.nav_until)
                    or (not w5_nav and t >= self.nav_until)
                )
                if replan:
                    raw_path = self.world.find_path((x, y), goal, max_expansions=800)
                    if w5_nav and raw_path and len(raw_path) > 1 and (int(raw_path[0][0] // 6.0), int(raw_path[0][1] // 6.0)) in self.bad_waypoints:
                        self.nav_path = raw_path[1:]
                    else:
                        self.nav_path = raw_path
                    self.nav_goal = goal
                    self.nav_until = t + 1.5
                if self.nav_path:
                    while len(self.nav_path) > 1 and math.hypot(self.nav_path[0][0] - x, self.nav_path[0][1] - y) < 6.0:
                        self.nav_path.pop(0)
                    if w5_nav and (t - self.last_wp_pop_t) < 1.2:
                        steer_target = self.nav_path[0]
                    else:
                        steer_target = self.world.next_waypoint(me["pos"], self.nav_path)

        yaw = math.atan2(steer_target[1] - y, steer_target[0] - x)
        # Boost when outside the zone, or when walking won't get us inside before the shrink ends.
        reserve_e = 38.0 if (w6_end and zone["next_radius"] <= 35.0) else ENERGY_RESERVE
        if FLAGS.get("energy_mgmt", True):
            urgent_boost = (not me["in_zone"]) or self.late or (self.evading and kite_boost) or cover_boost or drop_boost
            non_urgent_boost = ((not self.evading) and kite_boost and (me["energy"] >= reserve_e + 15.0)) or loot_boost
            boost = urgent_boost or non_urgent_boost
        else:
            boost = not me["in_zone"] or self.late or kite_boost or cover_boost or drop_boost or loot_boost
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
        if zone["next_radius"] == 0 and t >= self.unstick_until and not self.evading:
            speed = min(1.0, d / 4.0)
            if d < 4.0:
                self.stuck_s = 0.0
                boost = False
        self.move_yaw = yaw
        return {"move": [speed * math.cos(yaw), speed * math.sin(yaw)], "boost": bool(boost), "jump": bool(jump)}

    # ------------------------------------------------------------------ combat
    def pick_target(self, obs: mf.Observation, me):
        """Nearest enemy mech (preferring clear LOS and low-HP when heal/los_gate active); NPCs only if aggressors or farmed."""
        mx, my, mz = me["pos"]
        src = (mx, my, mz + me["stats"]["muzzle_height"])
        use_tri = FLAGS.get("weapon_tri", True)
        my_kinds = {w["kind"] for w in me["weapons"] if w} if use_tri else ()
        max_w_range = max((w["stats"].get("range", 0.0) for w in me["weapons"] if w), default=0.0)
        agg_drops = FLAGS.get("aggressive_drops", True)
        farm_npc_id = (
            self.npc_farm_target["id"]
            if (FLAGS.get("farm_npc", True) and self.npc_farm_target is not None)
            else None
        )
        best, best_s, best_los = None, math.inf, True
        for m in obs["mechs"]:
            if m["kind"] == "npc" and m["id"] not in self.aggressors and m["id"] != farm_npc_id:
                continue
            tp, _, _ = self.body(m)
            dist = math.dist(src, tp)
            hp_frac = m.get("hp", 1400.0) / max(1.0, m.get("max_hp", 1400.0))
            if FLAGS.get("w8_kill_conversion", True) and m["kind"] == "mech":
                # Continuous focus-fire execute curve: strongly favors finishing wounded targets without
                # discontinuities or sniping 170 m targets over point-blank rushers.
                s = dist * (0.30 + 0.70 * hp_frac)
            else:
                s = dist * (0.5 + 0.5 * hp_frac)
            if m["kind"] == "npc":
                if m["id"] != farm_npc_id:
                    s += NPC_PENALTY_M
            else:
                if use_tri:
                    ew = set(m.get("weapons") or ())
                    if "laser" in my_kinds and "mortar" in ew and dist <= self.laser_range:
                        s *= 0.65
                    if "lightning" in my_kinds and "laser" in ew and dist <= 50.0:
                        s *= 0.55 if m.get("charging") else 0.72
                    if "mortar" in my_kinds and "lightning" in ew and 16.0 <= dist <= 160.0:
                        s *= 0.75
                    if "mortar" in my_kinds and "laser" in ew and m.get("charging"):
                        s *= 0.80
                    if FLAGS.get("w12_bullet_evade", True) and m.get("charging") and dist <= min(self.laser_range, max_w_range):
                        s *= 0.75
                if agg_drops and self.active_drop_holder is not None and m["id"] == self.active_drop_holder:
                    s *= 0.55
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

    @staticmethod
    def solve_mortar_pitch(
        R: float,
        dz: float,
        v: float,
        g: float,
        high_arc: bool,
        min_p: float,
        max_p: float,
        ttl: float,
        r_off: float = 3.0,
    ) -> Optional[Tuple[float, float]]:
        """Closed-form ballistic pitch and flight time with 1-step barrel-offset refinement."""
        if R <= 1.0 or v <= 0.0 or g <= 0.0:
            return None
        v2 = v * v
        D = v2 * v2 - g * (g * R * R + 2.0 * dz * v2)
        if D < 0.0:
            return None
        s = math.sqrt(D)
        theta = math.atan2(v2 + s, g * R) if high_arc else math.atan2(v2 - s, g * R)
        R1 = R - r_off * math.cos(theta)
        dz1 = dz - r_off * math.sin(theta)
        if R1 > 1.0:
            D1 = v2 * v2 - g * (g * R1 * R1 + 2.0 * dz1 * v2)
            if D1 < 0.0:
                return None
            s1 = math.sqrt(D1)
            theta = math.atan2(v2 + s1, g * R1) if high_arc else math.atan2(v2 - s1, g * R1)
            R_eff = R - r_off * math.cos(theta)
        else:
            R_eff = R
        cos_t = math.cos(theta)
        if cos_t <= 1e-6 or R_eff <= 0.0:
            return None
        t_flight = R_eff / (v * cos_t)
        if not (min_p <= theta <= max_p) or not (0.0 < t_flight <= ttl - 0.1):
            return None
        return (theta, t_flight)

    def mortar_arc_clear(
        self,
        mx: float,
        my: float,
        mortar_z: float,
        r_off: float,
        yaw: float,
        pitch: float,
        v: float,
        g: float,
        t_flight: float,
    ) -> bool:
        """Verify mortar parabolic trajectory clears terrain and building prisms."""
        if self.world is None:
            return True
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        ux = cp * math.cos(yaw)
        uy = cp * math.sin(yaw)
        uz = sp
        sx = mx + r_off * ux
        sy = my + r_off * uy
        sz = mortar_z + r_off * uz

        world = self.world
        cell = world.cell
        grid = world.grid
        max_gz = world.max_ground_z

        def hits_world(ax: float, ay: float, az: float) -> bool:
            if az <= max_gz and az < world.ground_z(ax, ay):
                return True
            bucket = grid.get((int(ax // cell), int(ay // cell)))
            if bucket:
                for p in bucket:
                    if p["base"] <= az <= p["top"] and world._in_prism_xy(ax, ay, p):
                        return True
            return False

        if hits_world(sx, sy, sz):
            return False

        N = max(8, int(math.ceil(t_flight / 0.12)))
        vx, vy, vz = v * ux, v * uy, v * uz
        for k in range(1, N):
            tau = t_flight * k / N
            ax = sx + vx * tau
            ay = sy + vy * tau
            az = sz + vz * tau - 0.5 * g * tau * tau
            if hits_world(ax, ay, az):
                if tau < 0.85 * t_flight or math.hypot(ax - mx, ay - my) < 15.0:
                    return False
                break
        return True

    def solve_mortar(
        self,
        me,
        tpos,
        tvel,
        mortar_w,
        prefer_high: bool = False,
    ) -> Optional[Tuple[float, float, float]]:
        """Solve damped-lead mortar (yaw, pitch, t_flight) with self-splash and arc clearance checks."""
        mx, my, mz = me["pos"]
        if math.hypot(tpos[0] - mx, tpos[1] - my) < 15.0:
            return None
        st = me["stats"]
        mortar_z = mz + st.get("height", self.mech_h) + 0.5
        r_off = st.get("radius", self.mech_r) + 0.5
        min_p, max_p = st["min_pitch"], st["max_pitch"]
        wst = mortar_w.get("stats", {})
        v = float(wst.get("speed", self.mortar_speed))
        g = float(wst.get("gravity", self.mortar_gravity))
        ttl = float(wst.get("ttl", self.mortar_ttl))

        arcs = (True,) if prefer_high else (False, True)
        for high_arc in arcs:
            px, py = tpos[0], tpos[1]
            pz = (self.world.support_z(px, py, tpos[2]) if self.world is not None else tpos[2]) + 0.5
            sol = None
            for it in range(2):
                R = math.hypot(px - mx, py - my)
                if R < 15.0:
                    sol = None
                    break
                res = self.solve_mortar_pitch(R, pz - mortar_z, v, g, high_arc, min_p, max_p, ttl, r_off=r_off)
                if res is None:
                    sol = None
                    break
                pitch, tf = res
                if it == 0:
                    lead_scale = 1.0 if tf <= 1.2 else max(0.35, 1.2 / tf)
                    px = tpos[0] + tvel[0] * tf * lead_scale
                    py = tpos[1] + tvel[1] * tf * lead_scale
                    pz = (self.world.support_z(px, py, tpos[2]) if self.world is not None else tpos[2]) + 0.5
                else:
                    sol = (math.atan2(py - my, px - mx), pitch, tf)
            if sol is not None:
                yaw, pitch, tf = sol
                if self.mortar_arc_clear(mx, my, mortar_z, r_off, yaw, pitch, v, g, tf):
                    return (yaw, pitch, tf)
        return None

    def combat(self, obs: mf.Observation, me) -> dict:
        weapons = me["weapons"]
        fire = [False, False]
        w4c = FLAGS.get("w4_combat", True)
        cd_fast = 0.10 if w4c else 0.0
        cd_mortar = 0.06 if w4c else 0.0
        target = self.pick_target(obs, me) if FLAGS["fight"] else None
        gun = next((w for w in weapons if w and w["kind"] == "gun"), None)
        laser = next((w for w in weapons if w and w["kind"] == "laser"), None)
        mortar = next((w for w in weapons if w and w["kind"] == "mortar"), None)
        w5_laser = FLAGS.get("w5_laser_mastery", True)
        if laser is not None and laser.get("charge", 0.0) > 0.0:
            if target is None or not self.target_has_los:
                self.laser_no_los_s += self.dt
            else:
                self.laser_no_los_s = 0.0
        else:
            self.laser_no_los_s = 0.0
        drop_occluded_charge = (
            w5_laser
            and laser is not None
            and (
                (laser.get("charge", 0.0) >= laser["stats"].get("charge", 0.75) and self.laser_no_los_s >= 0.45)
                or (self.laser_no_los_s >= 0.30 and me["hp"] < 0.60 * me["max_hp"] and (obs["t"] - self.last_damaged_t) < 0.6)
            )
        )
        hold_ok = ((not FLAGS.get("laser_opt", True)) or (me["in_zone"] and not self.evading and not self.late)) and not drop_occluded_charge

        if target is None:
            for i, w in enumerate(weapons):
                if w and w["kind"] == "laser" and w.get("charge", 0) > 0:
                    fire[i] = hold_ok  # hold charge (up to hold_s) if target briefly vanished, unless dropping for speed

            if (FLAGS.get("fight", True) and FLAGS.get("mortar", True)
                    and mortar is not None and mortar["cooldown"] <= cd_mortar
                    and not self.healing
                    and not (laser is not None and laser.get("charge", 0) > 0)):
                mx, my, _ = me["pos"]
                t_now = obs["t"]
                best_mem = None
                best_md = math.inf
                for aid, (st_t, spos, svel) in self.seen.items():
                    if aid >= 16 or (t_now - st_t) > 3.5:
                        continue
                    md = math.hypot(spos[0] - mx, spos[1] - my)
                    if 18.0 <= md <= 190.0 and md < best_md:
                        best_md = md
                        best_mem = (aid, spos, svel)
                if best_mem is not None:
                    aid, spos, svel = best_mem
                    mortar_sol = self.solve_mortar(me, spos, svel, mortar, prefer_high=True)
                    if mortar_sol is not None:
                        m_yaw, m_pitch, _ = mortar_sol
                        aim = [m_yaw, min(max(m_pitch, me["stats"]["min_pitch"]), me["stats"]["max_pitch"])]
                        actual = me["aim"]
                        if abs(wrap(actual[0] - m_yaw)) <= 0.05 and abs(actual[1] - m_pitch) <= 0.04:
                            for i, w in enumerate(weapons):
                                if w and w["kind"] == "mortar":
                                    fire[i] = True
                        self.draw(me, spos, f"mortar mem {aid} {best_md:.0f}m")
                        return {"aim": aim, "fire": fire}

            if self.crate_target is not None and gun is not None:
                mx, my, mz = me["pos"]
                src = (mx, my, mz + me["stats"]["muzzle_height"])
                cpos = self.crate_target["pos"]
                cp = (cpos[0], cpos[1], cpos[2] + self.crate_h * 0.5)
                dist = math.dist(src, cp)
                direct = self.aim_at(src, cp, (0.0, 0.0, 0.0), None, me, clamp=False)
                aim = [direct[0], min(max(direct[1], me["stats"]["min_pitch"]), me["stats"]["max_pitch"])]
                actual = me["aim"]
                w7_sq = FLAGS.get("w7_squad_survival", True)
                if (self.world is None or self.world.has_los(src, cp)):
                    for i, w in enumerate(weapons):
                        if not w:
                            continue
                        if w["kind"] == "gun":
                            fire[i] = (w.get("ammo", 0) > 0 and w.get("reload", 0) == 0
                                       and dist <= w["stats"]["range"]
                                       and self.aligned(actual, direct, dist, self.crate_r, self.crate_h * 0.5))
                        elif w7_sq and w["kind"] == "lightning":
                            cost_c = w["stats"].get("energy", 3)
                            fire[i] = (w["cooldown"] <= cd_fast and dist <= w["stats"]["range"] * 0.95
                                       and me["energy"] >= cost_c + ENERGY_RESERVE
                                       and abs(wrap(actual[0] - direct[0])) <= math.radians(w["stats"]["cone_deg"]) * 0.4)
                self.draw(me, cp, f"crate {self.crate_target['id']} {dist:.0f}m")
                return {"aim": aim, "fire": fire}

            aim = self.idle_aim(obs, me)
            out = {"aim": aim, "fire": fire}
            if gun and gun.get("reload", 0) == 0:
                mag = gun["stats"]["mag"]
                ammo = gun.get("ammo", 0)
                t_idle = obs["t"]
                calm_reload = (
                    FLAGS.get("w12_bullet_evade", True)
                    and ammo < mag
                    and not any(aid < 16 for aid in self.seen)
                    and (self.threat is None or t_idle - self.threat[0] >= THREAT_MEMORY_S)
                    and (self.last_near_sound is None or t_idle - self.last_near_sound[0] > 2.5)
                )
                if ammo < mag * 0.5 or calm_reload:
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

        ceasefire = False
        if self.healing:
            if not ((has_los and dist <= 35.0) or (obs["t"] - self.last_damaged_t) <= 0.5):
                ceasefire = True
        elif (FLAGS.get("w9_squad_hunter", False) and not self.evading and self.cover_goal is not None
              and not self.cover_exposed and me["hp"] < 0.55 * me["max_hp"]
              and (obs["t"] - self.last_damaged_t) >= 0.8
              and target.get("hp", 1400.0) > 250.0 and dist > 38.0):
            # Hold fire while safely behind cover at <55% HP so 5s out-of-combat regen triggers!
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

        # Aim priority: laser_primary > mortar_primary > gun (led) / direct.
        if FLAGS.get("safe_laser", False):
            can_charge = (laser is not None and laser["cooldown"] <= cd_fast
                          and dist <= laser["stats"]["range"] and me["in_zone"]
                          and has_los and self.cover_goal is None
                          and target.get("hp", 1400.0) > 140.0
                          and not self.late and not self.healing and not self.evading)
        else:
            can_charge = (laser is not None and laser["cooldown"] <= cd_fast
                          and dist <= laser["stats"]["range"] and me["in_zone"]
                          and not self.late and not self.healing and not self.evading)
        laser_primary = laser is not None and dist <= laser["stats"]["range"] and (
            laser.get("charge", 0) > 0 or can_charge)

        laser_charge = float(laser.get("charge", 0.0)) if laser is not None else 0.0
        laser_full = max(1e-6, float(laser["stats"].get("charge", 0.75))) if laser is not None else 0.75
        laser_min_charge = float(laser["stats"].get("min_charge", 0.15)) if laser is not None else 0.15
        laser_est_dmg = 0.0
        laser_next_dmg = 0.0
        laser_early_finish = False
        laser_next_finish = False
        if laser is not None:
            lst = laser["stats"]
            min_frac = float(lst.get("min_frac", 0.35))
            cubes_cfg = self.cfg.get("cubes", {})
            cube_scale = 1.0 + float(cubes_cfg.get("damage_per_cube", 0.04)) * min(
                me.get("cubes", 0), int(cubes_cfg.get("stat_cap", 10))
            )
            cookoff_mult = (1.0 + float(lst.get("cookoff", 0.6))) if "mortar" in (target.get("weapons") or ()) else 1.0
            laser_est_dmg = (
                lst["damage"]
                * (min_frac + (1.0 - min_frac) * min(1.0, laser_charge / laser_full))
                * cube_scale
                * cookoff_mult
            )
            next_charge = min(laser_full, laser_charge + 0.15)
            laser_next_dmg = (
                lst["damage"]
                * (min_frac + (1.0 - min_frac) * min(1.0, next_charge / laser_full))
                * cube_scale
                * cookoff_mult
            )
            thp = target.get("hp", 1400.0)
            if FLAGS.get("laser_opt", True) and dist <= lst["range"]:
                laser_early_finish = laser_charge >= laser_min_charge and thp <= laser_est_dmg
                laser_next_finish = next_charge >= laser_min_charge and thp <= laser_next_dmg

        gun_ready = (
            gun is not None
            and gun.get("ammo", 0) > 0
            and gun.get("reload", 0) == 0
            and gun.get("cooldown", 0.0) <= 0.10
            and dist <= gun["stats"]["range"]
            and has_los
        )

        mortar_sol = None
        if (FLAGS.get("mortar", True) and mortar is not None and mortar["cooldown"] <= cd_mortar
                and not ceasefire and not (laser is not None and laser.get("charge", 0) > 0)):
            dist_xy = math.hypot(target["pos"][0] - mx, target["pos"][1] - my)
            if has_los and dist_xy >= 16.0:
                mortar_sol = self.solve_mortar(me, target["pos"], target["vel"], mortar, prefer_high=False)
            elif (not has_los) and dist_xy >= 18.0:
                mortar_sol = self.solve_mortar(me, target["pos"], target["vel"], mortar, prefer_high=True)
        mortar_primary = (not laser_primary) and (mortar_sol is not None)

        if laser_primary:
            if (w4c and gun_ready and laser_charge < laser_full - 0.16
                    and not laser_early_finish and not laser_next_finish):
                yaw, pitch = led
            else:
                yaw, pitch = direct
        elif mortar_primary:
            yaw, pitch = mortar_sol[0], mortar_sol[1]
        elif gun is None:
            yaw, pitch = direct
        else:
            yaw, pitch = led
        aim = [yaw, min(max(pitch, me["stats"]["min_pitch"]), me["stats"]["max_pitch"])]

        actual = me["aim"]
        for i, w in enumerate(weapons):
            if not w:
                continue
            k, st = w["kind"], w["stats"]
            if k == "mortar":
                if mortar_primary and not ceasefire:
                    m_yaw, m_pitch, _ = mortar_sol
                    fire[i] = abs(wrap(actual[0] - m_yaw)) <= 0.05 and abs(actual[1] - m_pitch) <= 0.04
                continue
            if not has_los or ceasefire:
                if k == "laser" and w.get("charge", 0) > 0:
                    fire[i] = hold_ok  # hold existing charge rather than dumping into wall
                else:
                    fire[i] = False
                continue
            if k == "gun":
                fire[i] = (w.get("ammo", 0) > 0 and w.get("reload", 0) == 0 and dist <= st["range"]
                           and self.aligned(actual, led, dist, radius, half_h))
            elif k == "lightning":
                cost = st.get("energy", 0)
                if FLAGS.get("energy_mgmt", True):
                    urgent_zap = (target.get("hp", 1400.0) < 180.0 or dist < 25.0 or bool(target.get("charging")))
                    reserve_e = 38.0 if (FLAGS.get("w6_endgame_mastery", True) and obs["zone"]["next_radius"] <= 35.0) else ENERGY_RESERVE
                    min_energy = cost if urgent_zap else (cost + reserve_e)
                else:
                    min_energy = cost + LIGHTNING_ENERGY_RESERVE
                fire[i] = (dist <= st["range"] * 0.95 and w["cooldown"] <= cd_fast
                           and me["energy"] >= min_energy
                           and abs(wrap(actual[0] - direct[0])) <= math.radians(st["cone_deg"]) * 0.4)
            elif k == "laser":
                charge = w.get("charge", 0)
                if charge == 0:
                    fire[i] = can_charge
                elif not hold_ok:
                    fire[i] = False
                else:
                    is_aligned = self.aligned(actual, direct, dist, radius, half_h)
                    if laser_early_finish and is_aligned:
                        fire[i] = False  # early release finisher
                    elif charge < st["charge"] or dist > st["range"]:
                        fire[i] = True  # keep charging, or hold charge if target stepped out of range
                    else:
                        # full and in range: hold until on target, then let go (auto-fires after hold_s anyway)
                        fire[i] = not is_aligned

        if (FLAGS.get("rear_guard", True) and not any(fire) and not has_los and not mortar_primary
                and not (laser is not None and laser.get("charge", 0) > 0)):
            zone = obs["zone"]
            cx, cy = zone["next_center"]
            d_next = math.hypot(cx - mx, cy - my)
            inner = zone["next_radius"] * 0.8
            if not me["in_zone"] or d_next > inner or self.cover_goal is not None or self.evading:
                rg_yaw = self.rear_guard_yaw(obs, me)
                if rg_yaw is not None:
                    aim = [rg_yaw, 0.0]

        self.draw(me, tp, f"target {target['id']} {dist:.0f}m")
        return {"aim": aim, "fire": fire}

    def rear_guard_yaw(self, obs: mf.Observation, me) -> Optional[float]:
        """Priority yaw for rear-guard awareness: 1. recent damaged bearing, 2. closest seen enemy, 3. near sound."""
        t = obs["t"]
        w6_esc = FLAGS.get("w6_crossfire_escape", True)
        has_mech_threat = w6_esc and (
            any(m.get("kind") == "mech" for m in obs["mechs"])
            or any(aid < 16 for aid in self.seen)
        )
        if (self.threat is not None and t - self.threat[0] <= THREAT_MEMORY_S
                and not (has_mech_threat and self.threat_id is not None and int(self.threat_id) >= 100)):
            return float(self.threat[1])
        mx, my = me["pos"][0], me["pos"][1]
        best_d = 220.0
        best_yaw: Optional[float] = None
        for aid, (_, spos, _) in self.seen.items():
            if aid < 16 or (aid in self.aggressors and not has_mech_threat):
                d = math.hypot(spos[0] - mx, spos[1] - my)
                if 0.1 < d <= best_d:
                    best_d = d
                    best_yaw = math.atan2(spos[1] - my, spos[0] - mx)
        if best_yaw is not None:
            return best_yaw
        if self.last_near_sound is not None and t - self.last_near_sound[0] <= 2.0:
            return float(self.last_near_sound[1])
        return None

    def idle_aim(self, obs: mf.Observation, me):
        """No target: face rear-guard threat/seen/near-sound, else look where we walk, else sweep."""
        if FLAGS.get("rear_guard", True):
            rg_yaw = self.rear_guard_yaw(obs, me)
            if rg_yaw is not None:
                return [rg_yaw, 0.0]
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
        for ix, iy, iz in self.evade_impacts:
            mf.draw.point([ix, iy, iz], "#ffaa00", 5)
        mf.draw.label(f"{state} | hp {me['hp']:.0f} e {me['energy']:.0f}"[:80])


if __name__ == "__main__":
    mf.run(Bot)
