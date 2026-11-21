"""Unit tests (stdlib only): python3 -m unittest discover -s tests -q

The wrapper swallows exceptions from act() and silently re-sends the last move/aim, so check.py can
PASS while act() crashes every tick. test_replay_stream runs Bot.act directly to catch that.
"""
import copy
import gzip
import json
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "kit", "check"))

import bot as B  # noqa: E402
import world_map as wm  # noqa: E402
from check import Schema  # noqa: E402

with gzip.open(os.path.join(ROOT, "kit", "check", "stream.jsonl.gz"), "rt") as f:
    MSGS = [json.loads(line) for line in f if line.strip()]
HELLO = MSGS[0]
HELLO.setdefault("map", {})["dir"] = os.path.join(ROOT, "kit", "arena")
OBS = [m for m in MSGS if m.get("type") == "obs"]
SCHEMA = Schema(os.path.join(ROOT, "kit", "mechfight.schema.json"))
WORLD = wm.WorldMap.load(HELLO)


def make_obs(**self_over):
    o = copy.deepcopy(OBS[0])
    o["self"].update(self_over)
    return o


def enemy(id_, pos, vel=(0, 0, 0), kind="mech"):
    m = {"id": id_, "kind": kind, "pos": list(pos), "vel": list(vel), "aim": [0, 0], "hp": 1400, "max_hp": 1400}
    if kind == "mech":
        m.update(cubes=0, weapons=["gun"], charging=False)
    return m


class Replay(unittest.TestCase):
    def run_stream(self, flags):
        old = dict(B.FLAGS)
        B.FLAGS.update(flags)
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            for o in OBS:
                B.mf.draw.clear()
                a = bot.act(copy.deepcopy(o))
                self.assertEqual(a["tick"], o["tick"])
                errs = SCHEMA.errors(a, "Action")
                self.assertFalse(errs, f"tick {o['tick']}: {errs}")
                self.assertLessEqual(math.hypot(*a["move"]), 1.0 + 1e-9)
                self.assertTrue(all(math.isfinite(v) for v in a["move"] + a["aim"]))
                self.assertEqual(len(a["fire"]), 2)
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_replay_stream_default(self):
        self.run_stream({})

    def test_replay_stream_all_flags(self):
        self.run_stream({"loot": True, "fight": True, "lead": False, "debug": False, "cover": True})
        self.run_stream({"loot": True, "fight": True, "lead": False, "debug": False, "cover": False})

    def test_replay_stream_no_fight(self):
        self.run_stream({"fight": False})


class WorldMapTests(unittest.TestCase):
    def test_ground_z_range_and_bilinear_smoothness(self):
        for x in range(-450, 500, 95):
            for y in range(-300, 350, 80):
                gz = WORLD.ground_z(x, y)
                self.assertIsInstance(gz, float)
                self.assertTrue(math.isfinite(gz))
                self.assertGreater(gz, -20.0)
                self.assertLess(gz, 60.0)
        # Bilinear interpolation inside a single 1 m cell should be linear along midlines
        x0, y0 = 10.0, 20.0
        z00 = WORLD.ground_z(x0, y0)
        z10 = WORLD.ground_z(x0 + 1.0, y0)
        z01 = WORLD.ground_z(x0, y0 + 1.0)
        z11 = WORLD.ground_z(x0 + 1.0, y0 + 1.0)
        self.assertAlmostEqual(WORLD.ground_z(x0 + 0.5, y0), 0.5 * (z00 + z10), places=5)
        self.assertAlmostEqual(WORLD.ground_z(x0 + 0.5, y0 + 0.5), 0.25 * (z00 + z10 + z01 + z11), places=5)

    def test_is_walkable_spawn_and_out_of_bounds(self):
        sx, sy, sz = OBS[0]["self"]["pos"]
        self.assertTrue(WORLD.is_walkable(sx, sy, sz))
        self.assertTrue(WORLD.is_walkable(sx, sy))
        self.assertFalse(WORLD.is_walkable(-9999.0, 0.0))
        self.assertFalse(WORLD.is_walkable(0.0, 9999.0))

    def test_is_walkable_building_and_bridge(self):
        # Solid building without holes: centroid at base + 1.0 is not walkable
        bldg = (-37.7, -292.75)  # block 0 prism: base=-5.64, top=24.27, ring around (-37.7, -292.75)
        self.assertFalse(WORLD.is_walkable(bldg[0], bldg[1], -4.64))
        self.assertFalse(WORLD.is_walkable(bldg[0], bldg[1]))
        # High bridge prism: ring around (86.9, 193.7), base=8.91, top=10.91
        bx, by, b_base = 86.9, 193.7, 8.91
        self.assertTrue(WORLD.is_walkable(bx, by, b_base - 10.0))
        if b_base - WORLD.ground_z(bx, by) >= 9.0:
            self.assertTrue(WORLD.is_walkable(bx, by))

    def test_path_clear(self):
        sx, sy, sz = OBS[0]["self"]["pos"]
        self.assertTrue(WORLD.path_clear((sx, sy, sz), (sx + 2.0, sy, WORLD.ground_z(sx + 2.0, sy))))
        # Path cutting across building prism [-136.6..-83.6] x [-234.3..-215.6], base=-5.64, top=8.61
        self.assertFalse(WORLD.path_clear((-145.0, -225.0, 0.0), (-75.0, -225.0, 0.0)))

    def test_has_los(self):
        # Two points high in open air above buildings have LOS
        self.assertTrue(WORLD.has_los((-100.0, -100.0, 120.0), (100.0, 100.0, 120.0)))
        # Segment passing straight through building prism at mid-height (base=-5.64, top=24.27 -> mid=9.315)
        mid_z = (-5.64 + 24.27) / 2.0
        self.assertFalse(WORLD.has_los((-50.0, -292.5, mid_z), (-25.0, -292.5, mid_z)))

    def test_astar_find_path_routes_around_building(self):
        # Start east and goal west of building prism 64 (-9.2..21.5 x -96..-78)
        start = (29.5, -87.0)
        goal = (-29.2, -87.0)
        self.assertFalse(WORLD.nav_line_clear(start, goal))
        path = WORLD.find_path(start, goal)
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 2)
        for wx, wy in path[:-1]:
            self.assertTrue(WORLD.is_walkable(wx, wy), f"waypoint ({wx}, {wy}) not walkable")
        self.assertLess(math.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1]), 6.0)


class Geometry(unittest.TestCase):
    def setUp(self):
        self.bot = B.Bot(copy.deepcopy(HELLO))

    def test_wrap(self):
        for a, w in [(0, 0), (math.pi / 2, math.pi / 2), (3 * math.pi / 2, -math.pi / 2), (-3 * math.pi / 2, math.pi / 2)]:
            self.assertAlmostEqual(B.wrap(a), w)

    def test_aim_direct(self):
        me = make_obs()["self"]
        yaw, pitch = self.bot.aim_at((0, 0, 0), (0, 100, 0), (0, 0, 0), None, me)
        self.assertAlmostEqual(yaw, math.pi / 2)  # north
        self.assertAlmostEqual(pitch, 0)
        yaw, pitch = self.bot.aim_at((0, 0, 0), (100, 0, 100), (0, 0, 0), None, me)
        self.assertAlmostEqual(pitch, math.pi / 4)

    def test_aim_lead_converges(self):
        me = make_obs()["self"]
        speed, v = 252.0, 10.0
        yaw, _ = self.bot.aim_at((0, 0, 0), (200, 0, 0), (0, v, 0), speed, me)
        # exact intercept: (200)^2 + (v t)^2 = (speed t)^2
        t = 200 / math.sqrt(speed ** 2 - v ** 2)
        self.assertAlmostEqual(yaw, math.atan2(v * t, 200), places=4)

    def test_pitch_clamped(self):
        me = make_obs()["self"]
        _, pitch = self.bot.aim_at((0, 0, 0), (1, 0, -100), (0, 0, 0), None, me)
        self.assertAlmostEqual(pitch, me["stats"]["min_pitch"])

    def test_aligned(self):
        self.assertTrue(B.Bot.aligned([0.0, 0.0], [0.005, 0.0], 100, 2.5, 4.5))
        self.assertFalse(B.Bot.aligned([0.0, 0.0], [0.1, 0.0], 100, 2.5, 4.5))
        self.assertTrue(B.Bot.aligned([math.pi - 0.001, 0], [-math.pi + 0.001, 0], 100, 2.5, 4.5))  # across the seam


class Decisions(unittest.TestCase):
    def setUp(self):
        B.mf.draw.clear()
        self.bot = B.Bot(copy.deepcopy(HELLO))

    def test_pick_preference(self):
        o = make_obs()
        o["offer"] = {"id": 1, "expires": 9, "choices": [{"id": "sensors", "name": "", "desc": ""},
                                                         {"id": "gun_rate", "name": "", "desc": ""},
                                                         {"id": "armor", "name": "", "desc": ""}]}
        self.assertEqual(self.bot.choose_pick(o), {"pick": 2})
        o["offer"]["choices"] = [{"id": "zzz", "name": "", "desc": ""}] * 3
        self.assertEqual(self.bot.choose_pick(o), {"pick": 0})
        o["offer"] = None
        self.assertEqual(self.bot.choose_pick(o), {})

    def test_gun_fires_only_when_aligned(self):
        o = make_obs()
        x, y, z = o["self"]["pos"]
        o["mechs"] = [enemy(5, (x + 100, y, z))]
        o["self"]["aim"] = [0.0, 0.0]
        self.bot.update_memory(o)
        a = self.bot.combat(o, o["self"])
        self.assertTrue(a["fire"][0])
        o["self"]["aim"] = [0.5, 0.0]
        self.assertFalse(self.bot.combat(o, o["self"])["fire"][0])

    def test_gun_no_fire_out_of_range_or_empty(self):
        o = make_obs()
        x, y, z = o["self"]["pos"]
        o["self"]["aim"] = [0.0, 0.0]
        o["mechs"] = [enemy(5, (x + 400, y, z))]
        self.assertFalse(self.bot.combat(o, o["self"])["fire"][0])
        o["mechs"] = [enemy(5, (x + 100, y, z))]
        o["self"]["weapons"][0]["ammo"] = 0
        self.assertFalse(self.bot.combat(o, o["self"])["fire"][0])

    def test_npc_ignored_unless_aggressor(self):
        o = make_obs()
        x, y, z = o["self"]["pos"]
        o["mechs"] = [enemy(104, (x + 20, y, z), kind="npc")]
        self.bot.update_memory(o)
        self.assertIsNone(self.bot.pick_target(o, o["self"]))
        o["events"] = [{"type": "damaged", "amount": 25, "weapon": "npc", "bearing": 0.0, "by": 104}]
        self.bot.update_memory(o)
        self.assertEqual(self.bot.pick_target(o, o["self"])["id"], 104)

    def test_laser_charges_then_releases_on_target(self):
        o = make_obs()
        x, y, z = o["self"]["pos"]
        o["mechs"] = [enemy(5, (x + 100, y, z))]
        laser = {"kind": "laser", "cooldown": 0, "charge": 0,
                 "stats": dict(HELLO["config"]["weapons"]["laser"])}
        o["self"]["weapons"] = [None, laser]
        o["self"]["aim"] = [0.0, 0.0]
        self.assertTrue(self.bot.combat(o, o["self"])["fire"][1])       # start charging
        laser["charge"] = 0.4
        self.assertTrue(self.bot.combat(o, o["self"])["fire"][1])       # keep charging
        laser["charge"] = laser["stats"]["charge"]
        self.assertFalse(self.bot.combat(o, o["self"])["fire"][1])      # full + aligned: let go
        o["self"]["aim"] = [0.5, 0.0]
        self.assertTrue(self.bot.combat(o, o["self"])["fire"][1])       # full, off target: hold
        o["mechs"] = [enemy(5, (x + 400, y, z))]
        self.assertTrue(self.bot.combat(o, o["self"])["fire"][1])       # full, out of range: hold charge, don't waste
        o["mechs"] = []
        self.assertTrue(self.bot.combat(o, o["self"])["fire"][1])       # target lost mid-charge: hold charge, don't dump
        laser["charge"] = 0
        self.assertFalse(self.bot.combat(o, o["self"])["fire"][1])      # not charging + no target: don't start charging

    def test_moves_toward_zone_when_outside(self):
        o = make_obs(in_zone=False)
        a = self.bot.movement(o, o["self"])
        cx, cy = o["zone"]["next_center"]
        x, y, _ = o["self"]["pos"]
        want = math.atan2(cy - y, cx - x)
        got = math.atan2(a["move"][1], a["move"][0])
        self.assertLess(abs(B.wrap(got - want)), 0.5)
        self.assertTrue(a["boost"])

    def test_holds_still_at_center(self):
        o = make_obs()
        cx, cy = o["zone"]["next_center"]
        o["self"]["pos"] = [cx, cy, 0.0]
        self.assertEqual(self.bot.movement(o, o["self"])["move"], [0.0, 0.0])

    def test_cover_selection_and_hysteresis(self):
        old_cover = B.FLAGS.get("cover", True)
        try:
            B.FLAGS["cover"] = True
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True)
            # Place mech in open ground near building prism 64 (-9.2..-78.0 to 21.5..-96.0), exposed to enemy
            mx, my = 29.5, -74.0
            mz = WORLD.ground_z(mx, my)
            o["self"]["pos"] = [mx, my, mz]
            o["zone"]["next_center"] = [mx, my]
            o["zone"]["next_radius"] = 200.0
            # Enemy in direct LOS just north of the building so stepping behind the building breaks LOS
            ex, ey = -29.2, -74.0
            ez = WORLD.ground_z(ex, ey)
            muzzle_h = o["self"]["stats"]["muzzle_height"]
            self.assertTrue(WORLD.has_los((mx, my, mz + muzzle_h), (ex, ey, ez + muzzle_h)))
            bearing = math.atan2(ey - my, ex - mx)
            o["mechs"] = [enemy(7, (ex, ey, ez))]
            o["events"] = [{"type": "damaged", "amount": 25, "weapon": "gun", "bearing": bearing, "by": 7}]
            bot.update_memory(o)
            a1 = bot.movement(o, o["self"])
            goal1 = getattr(bot, "cover_goal", None)
            self.assertIsNotNone(goal1, "Expected cover_goal to be selected when FLAGS['cover'] is True")
            gx, gy = goal1[0], goal1[1]
            gz = WORLD.ground_z(gx, gy)
            self.assertTrue(WORLD.is_walkable(gx, gy, gz))
            muzzle_h = o["self"]["stats"]["muzzle_height"]
            self.assertFalse(WORLD.has_los((gx, gy, gz + muzzle_h), (ex, ey, ez + muzzle_h)))
            self.assertGreater(math.hypot(*a1["move"]), 0.0)
            # Hysteresis: 0.1s later within COVER_HOLD_S, cover_goal remains committed
            o2 = copy.deepcopy(o)
            o2["t"] = o["t"] + 0.1
            o2["tick"] = o["tick"] + 1
            o2["events"] = []
            bot.update_memory(o2)
            bot.movement(o2, o2["self"])
            self.assertEqual(bot.cover_goal, goal1)
        finally:
            B.FLAGS["cover"] = old_cover

    def test_cover_route_survives_hit_timeout_and_checks_actual_position(self):
        o = make_obs(in_zone=True)
        me = o["self"]
        def place(x, y):
            me["pos"] = [x, y, WORLD.ground_z(x, y)]
        place(18, -70)
        o["zone"].update(next_center=[18, -70], next_radius=200)
        o["mechs"] = [enemy(7, (-29.2, -74, WORLD.ground_z(-29.2, -74)))]
        o["events"] = [dict(type="damaged", weapon="gun", by=7, bearing=math.pi)]
        self.bot.update_memory(o)
        # The cover point lies just beyond the building corner.
        self.bot.cover_goal = (24, -80.3923)
        self.bot.cover_until = o["t"] + 4
        place(22.8, -78.3138)
        o["events"] = []
        o["t"] += 2.1
        self.bot.update_memory(o)
        self.assertGreater(math.hypot(*self.bot.movement(o, me)["move"]), 0)
        self.assertEqual(self.bot.cover_goal, (24, -80.3923))
        place(*self.bot.cover_goal)
        self.assertEqual(self.bot.movement(o, me)["move"], [0, 0])

    def test_roof_path_and_corner_ray(self):
        self.assertTrue(WORLD.path_clear((-113, -195, 5.82), (-112, -195, 5.82)))
        a, b = (-87.54, -216.37, 1), (-82.74, -217.05, 1)
        self.assertFalse(WORLD.has_los(a, b))
        self.assertFalse(WORLD.has_los(b, a))

    def test_laser_cannot_steal_gun_lead_outside_zone(self):
        o = make_obs(in_zone=False)
        me = o["self"]
        x, y, z = me["pos"]
        self.bot.world = None
        me["weapons"][1] = dict(kind="laser", cooldown=0, charge=0,
                                stats=HELLO["config"]["weapons"]["laser"])
        o["mechs"] = [enemy(7, (x + 100, y, z), (0, 25, 0))]
        a = self.bot.combat(o, me)
        self.assertGreater(a["aim"][0], 0.08)
        me["aim"] = a["aim"]
        self.assertEqual(self.bot.combat(o, me)["fire"], [True, False])

    def test_unreachable_elevation(self):
        o = make_obs()
        me = o["self"]
        self.bot.world = None
        x, y, z = me["pos"]
        o["mechs"] = [enemy(7, (x + 4, y, z + 55))]
        me["aim"] = [0, me["stats"]["max_pitch"]]
        a = self.bot.combat(o, me)
        self.assertEqual(a["aim"], me["aim"])
        self.assertFalse(a["fire"][0])

    def test_pickup_within_server_radius_outside_destination_margin(self):
        o = make_obs()
        me = o["self"]
        x, y, z = me["pos"]
        o["zone"].update(next_center=[x - 100, y], next_radius=5)
        o["pickups"] = [dict(id=8111, kind="laser", pos=[x + 3.9, y, z])]
        self.assertEqual(self.bot.choose_take(o, me), {"take": 8111})

    def test_known_threats_include_unseen_attacker_and_remove_kill(self):
        o = make_obs()
        o["mechs"] = [enemy(7, (30, 0, 0))]
        o["events"] = [dict(type="damaged", by=4, bearing=1.2)]
        self.bot.update_memory(o)
        self.assertEqual(len(self.bot.known_threats(o, o["self"])), 2)
        o["mechs"] = []
        o["events"] = [dict(type="kill", target=7), dict(type="kill", target=4)]
        self.bot.update_memory(o)
        self.assertEqual(self.bot.known_threats(o, o["self"]), [])

    def test_cover_disabled_flag(self):
        old_cover = B.FLAGS.get("cover", True)
        try:
            B.FLAGS["cover"] = False
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True)
            mx, my = -24.0, -292.5
            mz = WORLD.ground_z(mx, my)
            o["self"]["pos"] = [mx, my, mz]
            o["zone"]["next_center"] = [mx, my]
            o["zone"]["next_radius"] = 200.0
            ex, ey = -55.0, -292.5
            o["mechs"] = [enemy(7, (ex, ey, WORLD.ground_z(ex, ey)))]
            o["events"] = [{"type": "damaged", "amount": 25, "weapon": "gun", "bearing": math.pi, "by": 7}]
            bot.update_memory(o)
            bot.movement(o, o["self"])
            self.assertIsNone(getattr(bot, "cover_goal", None))
        finally:
            B.FLAGS["cover"] = old_cover

    def test_los_fire_gate_keeps_aim_behind_wall(self):
        o = make_obs(in_zone=True)
        mx, my = 29.5, -87.0
        mz = WORLD.ground_z(mx, my)
        ex, ey = -29.2, -87.0
        ez = WORLD.ground_z(ex, ey)
        o["self"]["pos"] = [mx, my, mz]
        o["self"]["aim"] = [math.pi, 0.0]
        o["mechs"] = [enemy(8, (ex, ey, ez))]
        muzzle_h = o["self"]["stats"]["muzzle_height"]
        self.assertFalse(WORLD.has_los((mx, my, mz + muzzle_h), (ex, ey, ez + muzzle_h)))
        self.bot.update_memory(o)
        a = self.bot.combat(o, o["self"])
        self.assertLess(abs(B.wrap(a["aim"][0] - math.pi)), 0.2)
        self.assertEqual(a["fire"], [False, False])
        # If laser already has charge > 0, hold charge rather than releasing into a wall
        laser = {"kind": "laser", "cooldown": 0, "charge": 0.4,
                 "stats": dict(HELLO["config"]["weapons"]["laser"])}
        o["self"]["weapons"] = [o["self"]["weapons"][0], laser]
        a2 = self.bot.combat(o, o["self"])
        self.assertEqual(a2["fire"], [False, True])

    def test_target_priority_los_and_low_hp(self):
        o = make_obs(in_zone=True)
        # Two enemies in clear LOS: 25m full HP (1400) vs 35m low HP (100)
        mx, my, mz = o["self"]["pos"]
        e_full = enemy(10, (mx + 25.0, my, WORLD.ground_z(mx + 25.0, my)))
        e_low = enemy(11, (mx + 35.0, my, WORLD.ground_z(mx + 35.0, my)))
        e_low["hp"] = 100
        muzzle_h = o["self"]["stats"]["muzzle_height"]
        self.assertTrue(WORLD.has_los((mx, my, mz + muzzle_h), (mx + 25.0, my, e_full["pos"][2] + 4.5)))
        self.assertTrue(WORLD.has_los((mx, my, mz + muzzle_h), (mx + 35.0, my, e_low["pos"][2] + 4.5)))
        o["mechs"] = [e_full, e_low]
        self.bot.update_memory(o)
        self.assertEqual(self.bot.pick_target(o, o["self"])["id"], 11)

        # Enemy behind wall at ~40m (has_los == False) vs enemy in clear LOS at 58m (has_los == True)
        mx2, my2 = 29.5, -74.0
        mz2 = WORLD.ground_z(mx2, my2)
        o["self"]["pos"] = [mx2, my2, mz2]
        bx, by = -10.5, -90.0  # blocked by building prism 64 (-9.2..21.5 x -96..-78)
        bz = WORLD.ground_z(bx, by)
        cx, cy = -29.2, -74.0  # 58.7m west in clear LOS just north of prism 64
        cz = WORLD.ground_z(cx, cy)
        self.assertFalse(WORLD.has_los((mx2, my2, mz2 + muzzle_h), (bx, by, bz + 4.5)))
        self.assertTrue(WORLD.has_los((mx2, my2, mz2 + muzzle_h), (cx, cy, cz + 4.5)))
        o["mechs"] = [enemy(12, (bx, by, bz)), enemy(13, (cx, cy, cz))]
        self.bot.update_memory(o)
        self.assertEqual(self.bot.pick_target(o, o["self"])["id"], 13)

    def test_heal_cover_ceasefire_and_break_rules(self):
        old_heal = B.FLAGS.get("heal", True)
        try:
            B.FLAGS["heal"] = True
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True, hp=500, max_hp=1400)
            o["t"] = 10.0
            o["events"] = []
            x, y, z = o["self"]["pos"]
            o["self"]["aim"] = [0.0, 0.0]
            o["mechs"] = [enemy(15, (x + 100.0, y, z))]
            bot.update_memory(o)
            a = bot.combat(o, o["self"])
            self.assertEqual(a["fire"], [False, False])

            # Close-range LOS threat (25m <= 35m) breaks ceasefire
            o["mechs"] = [enemy(15, (x + 25.0, y, z))]
            bot.update_memory(o)
            a_close = bot.combat(o, o["self"])
            self.assertTrue(a_close["fire"][0])

            # Recovery to hp >= 0.85 * max_hp (1250 >= 1190) ends ceasefire at 100m
            o["self"]["hp"] = 1250
            o["mechs"] = [enemy(15, (x + 100.0, y, z))]
            bot.update_memory(o)
            a_healed = bot.combat(o, o["self"])
            self.assertTrue(a_healed["fire"][0])

            # Laser speed-trap guard: when charge == 0, do not start charging while healing
            o["self"]["hp"] = 500
            laser = {"kind": "laser", "cooldown": 0, "charge": 0,
                     "stats": dict(HELLO["config"]["weapons"]["laser"])}
            o["self"]["weapons"] = [o["self"]["weapons"][0], laser]
            bot.update_memory(o)
            a_laser_guard = bot.combat(o, o["self"])
            self.assertFalse(a_laser_guard["fire"][1])
        finally:
            B.FLAGS["heal"] = old_heal

    def test_open_field_kiting_and_perimeter_settle(self):
        old_kite = B.FLAGS.get("kite", True)
        try:
            B.FLAGS["kite"] = True
            bot = B.Bot(copy.deepcopy(HELLO))
            # Perimeter settle: next_radius=300, alive=10 -> settle = 0.60 * 300 = 180m; at d=165m holds still
            o = make_obs(in_zone=True)
            cx, cy = o["zone"]["next_center"]
            o["zone"]["next_radius"] = 300.0
            o["alive"] = 10
            o["mechs"] = []
            o["pods"] = []
            o["cubes"] = []
            o["events"] = []
            px, py = cx + 165.0, cy
            o["self"]["pos"] = [px, py, WORLD.ground_z(px, py)]
            bot.update_memory(o)
            a_settle = bot.movement(o, o["self"])
            self.assertEqual(a_settle["move"], [0.0, 0.0])

            # Open-field kiting: use open ground near spawn with no cover building blocking LOS to enemy 35m away
            ox, oy, oz = OBS[0]["self"]["pos"]
            ex, ey = ox + 35.0, oy
            ez = WORLD.ground_z(ex, ey)
            o2 = make_obs(in_zone=True)
            o2["self"]["pos"] = [ox, oy, oz]
            o2["zone"]["next_center"] = [ox, oy]
            o2["zone"]["next_radius"] = 300.0
            o2["pods"] = []
            o2["cubes"] = []
            o2["mechs"] = [enemy(20, (ex, ey, ez))]
            bot.cover_goal = None
            bot.update_memory(o2)
            # Ensure no cover is available at this open location, or force choose_cover None if needed
            if hasattr(bot, "choose_cover") and bot.choose_cover(o2, o2["self"], ox, oy, 240.0) is not None:
                bot.choose_cover = lambda *args, **kwargs: None
            a_kite = bot.movement(o2, o2["self"])
            mx_v, my_v = a_kite["move"]
            self.assertGreater(math.hypot(mx_v, my_v), 0.1)
            # Dot product of move vector with vector to enemy (35, 0) must be < 0 (kiting away)
            self.assertLess(mx_v * (ex - ox) + my_v * (ey - oy), 0.0)
        finally:
            B.FLAGS["kite"] = old_kite

    def test_cube_pickup_and_near_sound_idle_aim(self):
        old_cubes = B.FLAGS.get("cubes", True)
        old_sound = B.FLAGS.get("sound", True)
        old_heal = B.FLAGS.get("heal", True)
        try:
            B.FLAGS["cubes"] = True
            B.FLAGS["sound"] = True
            B.FLAGS["heal"] = True
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True, hp=500, max_hp=1400)
            o["t"] = 20.0
            sx, sy, sz = o["self"]["pos"]
            o["zone"]["next_center"] = [sx, sy]
            o["zone"]["next_radius"] = 300.0
            o["mechs"] = []
            o["pods"] = []
            o["events"] = []
            cx, cy = sx + 10.0, sy
            cz = WORLD.ground_z(cx, cy)
            o["cubes"] = [{"id": 99, "pos": [cx, cy, cz], "amount": 1}]
            o["sounds"] = [{"kind": "gun", "bearing": 1.25, "dist": "near", "age": 0.05}]
            bot.update_memory(o)
            a_move = bot.movement(o, o["self"])
            self.assertGreater(a_move["move"][0], 0.9)
            a_combat = bot.combat(o, o["self"])
            self.assertAlmostEqual(a_combat["aim"][0], 1.25, places=5)
        finally:
            B.FLAGS["cubes"] = old_cubes
            B.FLAGS["sound"] = old_sound
            B.FLAGS["heal"] = old_heal

    def test_proactive_cover_combat_strafe_and_selective_fire(self):
        old = dict(B.FLAGS)
        B.FLAGS.update(proactive_cover=True, combat_strafe=True, selective_fire=True)
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            # 1. Selective fire: during a full act() step with alive > 4, unprovoked full-HP enemy at 120m holds fire
            o = make_obs(in_zone=True, hp=1400, max_hp=1400)
            o["t"] = 15.0
            o["alive"] = 12
            o["events"] = []
            x, y, z = o["self"]["pos"]
            o["self"]["aim"] = [0.0, 0.0]
            o["mechs"] = [enemy(14, (x + 120.0, y, z))]
            a_unprov = bot.act(copy.deepcopy(o))
            self.assertEqual(a_unprov["fire"], [False, False])

            # Finishable low-HP enemy at 120m breaks selective_fire ceasefire
            o_low = copy.deepcopy(o)
            o_low["mechs"][0]["hp"] = 300
            a_finish = bot.act(o_low)
            self.assertTrue(a_finish["fire"][0])

            # 2. Combat strafe: when exposed to an enemy at 80m in open ground with no cover, never stands still
            bot2 = B.Bot(copy.deepcopy(HELLO))
            bot2.world = None
            o_strafe = make_obs(in_zone=True, hp=1400, max_hp=1400)
            cx, cy = o_strafe["zone"]["next_center"]
            o_strafe["self"]["pos"] = [cx, cy, 0.0]
            o_strafe["zone"]["next_radius"] = 300.0
            o_strafe["mechs"] = [enemy(9, (cx + 80.0, cy, 0.0))]
            o_strafe["events"] = []
            a_st = bot2.act(o_strafe)
            self.assertGreater(math.hypot(*a_st["move"]), 0.5)
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_early_weapon_local_floor_and_detour(self):
        old = dict(B.FLAGS)
        B.FLAGS["early_weapon"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True, hp=1400, max_hp=1400)
            x, y, z = o["self"]["pos"]
            o["zone"].update(next_center=[x, y], next_radius=300.0, shrink_end=o["t"] + 80.0)
            # Place a laser 75m away (> 40m baseline, <= 85m cap) on ground level at its own coordinates
            px, py = x + 75.0, y
            pz = WORLD.ground_z(px, py)
            self.assertTrue(WORLD.is_walkable(px, py, pz))
            self.assertTrue(WORLD.path_clear((x, y, z), (px, py, pz)))
            o["pickups"] = [{"id": 8999, "kind": "laser", "pos": [px, py, pz]}]
            o["mechs"] = []
            bot.update_memory(o)
            bot.choose_take(o, o["self"])
            self.assertEqual(bot.loot_goal, (px, py))

            # Rejected when an enemy mech has LOS (mid-range >40m detour requires no enemy LOS)
            ex, ey = x + 60.0, y
            ez = WORLD.ground_z(ex, ey)
            self.assertTrue(WORLD.has_los((x, y, z + 8.0), (ex, ey, ez + 8.0)))
            o_los = copy.deepcopy(o)
            o_los["mechs"] = [enemy(12, (ex, ey, ez))]
            bot.update_memory(o_los)
            bot.choose_take(o_los, o_los["self"])
            self.assertIsNone(bot.loot_goal)

            # Rejected when outside inner (inner = 0.8 * next_radius = 60m < 75m)
            o_out = copy.deepcopy(o)
            o_out["zone"].update(next_center=[x, y], next_radius=75.0)
            bot.update_memory(o_out)
            bot.choose_take(o_out, o_out["self"])
            self.assertIsNone(bot.loot_goal)

            # Stall after 2.3s with zero displacement blacklists pickup 8999
            bot.act(copy.deepcopy(o))
            o_stall = copy.deepcopy(o)
            o_stall["t"] += 2.3
            bot.act(o_stall)
            self.assertIn(8999, bot.bad_pickups)
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_safe_laser_gates_cover_goal_and_low_hp(self):
        old = dict(B.FLAGS)
        B.FLAGS["safe_laser"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True)
            x, y, z = o["self"]["pos"]
            laser = {"kind": "laser", "cooldown": 0, "charge": 0,
                     "stats": dict(HELLO["config"]["weapons"]["laser"])}
            o["self"]["weapons"] = [o["self"]["weapons"][0], laser]
            o["self"]["aim"] = [0.0, 0.0]

            # Normal healthy enemy in LOS with no cover_goal: laser starts charging
            o["mechs"] = [enemy(5, (x + 100, y, z))]
            bot.cover_goal = None
            bot.update_memory(o)
            self.assertTrue(bot.combat(o, o["self"])["fire"][1])

            # While cover_goal is set: laser does not start charging
            bot.cover_goal = (x - 10.0, y)
            self.assertFalse(bot.combat(o, o["self"])["fire"][1])

            # When target HP <= 140: laser does not start charging (let gun finish)
            bot.cover_goal = None
            o["mechs"][0]["hp"] = 140.0
            self.assertFalse(bot.combat(o, o["self"])["fire"][1])
            o["mechs"][0]["hp"] = 100.0
            self.assertFalse(bot.combat(o, o["self"])["fire"][1])
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_evade_incoming_mortar_and_ignore_distant(self):
        old = dict(B.FLAGS)
        B.FLAGS["evade"] = True
        try:
            o = make_obs(in_zone=True, energy=80.0)
            x, y, z = o["self"]["pos"]
            o["zone"].update(center=[x, y], radius=400.0, next_center=[x, y], next_radius=200.0)
            o["mechs"] = []
            o["cubes"] = []
            o["pickups"] = []
            o["events"] = []

            # Incoming mortar shell landing near (x, y) in ~1.4 s under g=22
            o["projectiles"] = [
                {"id": 501, "kind": "mortar", "pos": [x - 60.0, y, z + 15.0], "vel": [60.0, 0.0, 5.0]}
            ]
            a = self.bot.movement(o, o["self"])
            self.assertGreater(math.hypot(*a["move"]), 0.5)
            self.assertTrue(a["boost"])
            self.assertTrue(self.bot.evading)

            # Harmless distant mortar shell landing ~40 m away: hold still
            o["projectiles"] = [
                {"id": 502, "kind": "mortar", "pos": [x - 100.0, y + 40.0, z + 15.0], "vel": [60.0, 0.0, 5.0]}
            ]
            a2 = self.bot.movement(o, o["self"])
            self.assertEqual(a2["move"], [0.0, 0.0])
            self.assertFalse(self.bot.evading)
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_evade_charging_laser_away_from_centerline(self):
        old = dict(B.FLAGS)
        B.FLAGS["evade"] = True
        try:
            o = make_obs(in_zone=True, energy=80.0)
            x, y, z = o["self"]["pos"]
            o["zone"].update(center=[x, y], radius=400.0, next_center=[x, y], next_radius=200.0)
            o["cubes"] = []
            o["pickups"] = []
            o["projectiles"] = []
            o["events"] = []

            # Enemy 100 m west aiming east; our mech is 2 m NORTH of the beam centerline
            e = enemy(7, (x - 100.0, y - 2.0, z))
            e["aim"] = [0.0, 0.0]
            e["charging"] = True
            e["weapons"] = ["gun", "laser"]
            o["mechs"] = [e]
            a = self.bot.movement(o, o["self"])
            self.assertGreater(a["move"][1], 0.5)
            self.assertTrue(a["boost"])
            self.assertTrue(self.bot.evading)

            # Not charging: no laser evasion
            e["charging"] = False
            self.bot.movement(o, o["self"])
            self.assertFalse(self.bot.evading)

            # Charging but aimed 45 deg away from us: no laser evasion
            e["charging"] = True
            e["aim"] = [0.8, 0.0]
            self.bot.movement(o, o["self"])
            self.assertFalse(self.bot.evading)
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_evade_wall_blocked_flips_direction(self):
        old = dict(B.FLAGS)
        B.FLAGS["evade"] = True
        try:
            o = make_obs(in_zone=True, energy=80.0)
            mx, my = 5.0, -74.0
            gz = WORLD.ground_z(mx, my)
            o["self"]["pos"] = [mx, my, gz]
            o["zone"].update(center=[mx, my], radius=400.0, next_center=[mx, my], next_radius=200.0)
            o["cubes"] = []
            o["pickups"] = []
            o["projectiles"] = []
            o["events"] = []

            # Enemy 34 m west aiming east with clear LOS; preferred offset is slightly south into building prism 64
            ex, ey = -29.2, -73.0
            ez = WORLD.ground_z(ex, ey)
            e = enemy(8, (ex, ey, ez))
            e["aim"] = [0.0, 0.0]
            e["charging"] = True
            e["weapons"] = ["gun", "laser"]
            o["mechs"] = [e]
            a = self.bot.movement(o, o["self"])
            self.assertTrue(self.bot.evading)
            self.assertGreater(a["move"][1], 0.3)
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_evade_zero_gun_dps_cost(self):
        old = dict(B.FLAGS)
        B.FLAGS["evade"] = True
        try:
            o = make_obs(in_zone=True, energy=80.0, hp=1400, max_hp=1400)
            o["t"] = 10.0
            x, y, z = o["self"]["pos"]
            o["self"]["aim"] = [0.0, 0.0]
            o["zone"].update(center=[x, y], radius=400.0, next_center=[x, y], next_radius=200.0)
            o["cubes"] = []
            o["pickups"] = []
            o["projectiles"] = []
            o["events"] = []

            e = enemy(9, (x + 50.0, y, z))
            e["aim"] = [math.pi, 0.0]
            e["charging"] = True
            e["weapons"] = ["gun", "laser"]
            o["mechs"] = [e]
            a = self.bot.act(o)
            self.assertTrue(self.bot.evading)
            self.assertGreater(math.hypot(*a["move"]), 0.5)
            self.assertTrue(a["fire"][0])
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_evade_disabled_flag(self):
        old = dict(B.FLAGS)
        B.FLAGS["evade"] = False
        try:
            o = make_obs(in_zone=True, energy=80.0)
            x, y, z = o["self"]["pos"]
            o["zone"].update(center=[x, y], radius=400.0, next_center=[x, y], next_radius=200.0)
            o["mechs"] = []
            o["cubes"] = []
            o["pickups"] = []
            o["events"] = []
            o["projectiles"] = [
                {"id": 503, "kind": "mortar", "pos": [x - 60.0, y, z + 15.0], "vel": [60.0, 0.0, 5.0]}
            ]
            a = self.bot.movement(o, o["self"])
            self.assertFalse(self.bot.evading)
            self.assertEqual(a["move"], [0.0, 0.0])
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_take_failed_reason_blacklist(self):
        o = make_obs(in_zone=True)
        o["t"] = 10.0
        o["events"] = [
            {"type": "take_failed", "id": 101, "reason": "gone"},
            {"type": "take_failed", "id": 102, "reason": "have"},
            {"type": "take_failed", "id": 103, "reason": "full"},
            {"type": "take_failed", "id": 104, "reason": "busy"},
            {"type": "take_failed", "id": 105, "reason": "range"},
        ]
        self.bot.update_memory(o)
        self.assertIn(101, self.bot.bad_pickups)
        self.assertIn(102, self.bot.bad_pickups)
        self.assertIn(103, self.bot.bad_pickups)
        self.assertNotIn(104, self.bot.bad_pickups)
        self.assertNotIn(105, self.bot.bad_pickups)

    def test_farm_crate_selects_and_shoots_safe_ground_crate(self):
        old = dict(B.FLAGS)
        B.FLAGS["farm_crate"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True, hp=1400, max_hp=1400, cubes=0)
            o["t"] = 12.0
            x, y, z = o["self"]["pos"]
            o["zone"].update(center=[x, y], radius=400.0, next_center=[x, y], next_radius=200.0)
            o["mechs"] = []
            o["cubes"] = []
            o["pickups"] = []
            o["events"] = []
            cx_c, cy_c = x + 20.0, y
            cz_c = WORLD.ground_z(cx_c, cy_c)
            o["crates"] = [{"id": 2001, "pos": [cx_c, cy_c, cz_c], "hp": 300}]

            # Walks toward crate when d_xy > 5.0 m
            bot.update_memory(o)
            am = bot.movement(o, o["self"])
            self.assertIsNotNone(bot.crate_target)
            self.assertEqual(bot.crate_target["id"], 2001)
            self.assertGreater(am["move"][0], 0.8)

            # Aims at crate center and fires when aligned
            src = (x, y, z + o["self"]["stats"]["muzzle_height"])
            cp = (cx_c, cy_c, cz_c + bot.crate_h * 0.5)
            want = bot.aim_at(src, cp, (0.0, 0.0, 0.0), None, o["self"], clamp=True)
            o["self"]["aim"] = want
            ac = bot.combat(o, o["self"])
            self.assertTrue(ac["fire"][0])
            self.assertAlmostEqual(ac["aim"][0], want[0], places=4)

            # Off-aim: does not fire
            o["self"]["aim"] = [B.wrap(want[0] + 0.5), want[1]]
            self.assertFalse(bot.combat(o, o["self"])["fire"][0])

            # Within 5.0 m: holds still (goal = None) even if d > settle
            o["zone"]["next_center"] = [x + 150.0, y]
            o["crates"] = [{"id": 2001, "pos": [x + 4.0, y, WORLD.ground_z(x + 4.0, y)], "hp": 300}]
            am_close = bot.movement(o, o["self"])
            self.assertEqual(am_close["move"], [0.0, 0.0])
            self.assertIsNotNone(bot.crate_target)
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_farm_crate_gates_and_filters(self):
        old = dict(B.FLAGS)
        B.FLAGS["farm_crate"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True, hp=1400, max_hp=1400, cubes=0)
            o["t"] = 20.0
            x, y, z = o["self"]["pos"]
            o["zone"].update(center=[x, y], radius=400.0, next_center=[x, y], next_radius=200.0)
            o["mechs"] = []
            o["cubes"] = []
            o["pickups"] = []
            o["events"] = []
            cx_c, cy_c = x + 15.0, y
            cz_c = WORLD.ground_z(cx_c, cy_c)
            o["crates"] = [{"id": 2010, "pos": [cx_c, cy_c, cz_c], "hp": 300}]

            # Baseline selects crate
            self.assertIsNotNone(bot.choose_crate(o, o["self"], x, y, 160.0))

            # Damaged gate
            o["self"]["hp"] = 1350
            self.assertIsNone(bot.choose_crate(o, o["self"], x, y, 160.0))
            o["self"]["hp"] = 1400

            # Visible enemy mech gate
            o["mechs"] = [enemy(4, (x + 100.0, y, z))]
            self.assertIsNone(bot.choose_crate(o, o["self"], x, y, 160.0))
            o["mechs"] = []

            # Remembered enemy mech gate (aid < 16)
            bot.seen[5] = (o["t"] - 1.0, [x + 90.0, y, z], [0.0, 0.0, 0.0])
            self.assertIsNone(bot.choose_crate(o, o["self"], x, y, 160.0))
            bot.seen.clear()

            # NPC within 48m of self rejected; NPC at 60m from self & 46m from crate allowed
            o["mechs"] = [enemy(101, (x - 40.0, y, z), kind="npc")]
            self.assertIsNone(bot.choose_crate(o, o["self"], x, y, 160.0))
            o["mechs"] = [enemy(101, (x + 50.0, y, z), kind="npc")]  # 35m from crate -> rejected
            self.assertIsNone(bot.choose_crate(o, o["self"], x, y, 160.0))
            o["mechs"] = [enemy(101, (x - 60.0, y, z), kind="npc")]  # 60m from self, 75m from crate -> allowed
            self.assertIsNotNone(bot.choose_crate(o, o["self"], x, y, 160.0))
            o["mechs"] = []

            # Roof crate (pz - ground_z > 2.0) rejected
            o["crates"] = [{"id": 2011, "pos": [cx_c, cy_c, cz_c + 6.0], "hp": 300}]
            self.assertIsNone(bot.choose_crate(o, o["self"], x, y, 160.0))

            # Goal window: cubes == 1 with empty weapon slot & t < 90 allowed, but t >= 90 rejected
            o["crates"] = [{"id": 2010, "pos": [cx_c, cy_c, cz_c], "hp": 300}]
            o["self"]["cubes"] = 1
            self.assertIsNotNone(bot.choose_crate(o, o["self"], x, y, 160.0))
            o["t"] = 95.0
            self.assertIsNone(bot.choose_crate(o, o["self"], x, y, 160.0))

            # Flag disabled
            o["t"] = 20.0
            o["self"]["cubes"] = 0
            B.FLAGS["farm_crate"] = False
            self.assertIsNone(bot.choose_crate(o, o["self"], x, y, 160.0))
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_farm_crate_watchdog_blacklists_stuck_crate(self):
        old = dict(B.FLAGS)
        B.FLAGS["farm_crate"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True, hp=1400, max_hp=1400, cubes=0)
            o["t"] = 10.0
            x, y, _ = o["self"]["pos"]
            o["mechs"] = []
            cx_c, cy_c = x + 15.0, y
            cz_c = WORLD.ground_z(cx_c, cy_c)
            o["crates"] = [{"id": 2099, "pos": [cx_c, cy_c, cz_c], "hp": 300}]

            self.assertIsNotNone(bot.choose_crate(o, o["self"], x, y, 160.0))
            o["t"] = 13.6
            self.assertIsNone(bot.choose_crate(o, o["self"], x, y, 160.0))
            self.assertIn(2099, bot.bad_crates)

            # Pruned after 30s expiry
            o["t"] = 44.0
            bot.update_memory(o)
            self.assertNotIn(2099, bot.bad_crates)
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_mortar_loot_pickup_flag(self):
        old = dict(B.FLAGS)
        try:
            o = make_obs(in_zone=True)
            x, y, z = o["self"]["pos"]
            o["zone"].update(next_center=[x, y], next_radius=200.0)
            o["pickups"] = [{"id": 7771, "kind": "mortar", "pos": [x + 15.0, y, z]}]

            B.FLAGS["mortar"] = True
            self.bot.choose_take(o, o["self"])
            self.assertEqual(self.bot.loot_goal, (x + 15.0, y))

            B.FLAGS["mortar"] = False
            self.bot.choose_take(o, o["self"])
            self.assertIsNone(self.bot.loot_goal)
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_mortar_solve_pitch_exactness_and_bounds(self):
        v, g, ttl, r_off = 72.0, 22.0, 7.0, 3.0
        min_p, max_p = -0.7, 1.3
        for R, dz, high in [(80.0, 0.0, False), (120.0, 8.0, False), (140.0, -5.0, True)]:
            sol = B.Bot.solve_mortar_pitch(R, dz, v, g, high, min_p, max_p, ttl, r_off=r_off)
            self.assertIsNotNone(sol, f"Expected solution for R={R}, dz={dz}, high={high}")
            theta, tf = sol
            self.assertGreaterEqual(theta, min_p)
            self.assertLessEqual(theta, max_p)
            self.assertGreater(tf, 0.0)
            self.assertLessEqual(tf, ttl - 0.1)
            # Forward kinematics check
            hit_R = r_off * math.cos(theta) + v * math.cos(theta) * tf
            hit_dz = r_off * math.sin(theta) + v * math.sin(theta) * tf - 0.5 * g * tf * tf
            self.assertAlmostEqual(hit_R, R, delta=0.1)
            self.assertAlmostEqual(hit_dz, dz, delta=0.1)

        # Out-of-range and degenerate R
        self.assertIsNone(B.Bot.solve_mortar_pitch(500.0, 0.0, v, g, False, min_p, max_p, ttl))
        self.assertIsNone(B.Bot.solve_mortar_pitch(0.5, 0.0, v, g, False, min_p, max_p, ttl))

    def test_mortar_self_splash_guard_and_damped_lead(self):
        bot = B.Bot(copy.deepcopy(HELLO))
        bot.world = None
        o = make_obs(in_zone=True)
        me = o["self"]
        x, y, z = me["pos"]
        mortar_w = {"kind": "mortar", "cooldown": 0.0, "stats": dict(HELLO["config"]["weapons"]["mortar"])}

        # Raw distance < 15 m rejected
        self.assertIsNone(bot.solve_mortar(me, [x + 12.0, y, z], [0.0, 0.0, 0.0], mortar_w))
        # Target at 18 m rushing toward us -> predicted < 15 m rejected
        self.assertIsNone(bot.solve_mortar(me, [x + 18.0, y, z], [-20.0, 0.0, 0.0], mortar_w))

        # Moving target at 80 m: yaw leads north (+y)
        sol = bot.solve_mortar(me, [x + 80.0, y, z], [0.0, 12.0, 0.0], mortar_w)
        self.assertIsNotNone(sol)
        self.assertGreater(sol[0], 0.05)

    def test_mortar_over_building_lob_and_arc_clear(self):
        bot = B.Bot(copy.deepcopy(HELLO))
        o = make_obs(in_zone=True)
        me = o["self"]
        mortar_w = {"kind": "mortar", "cooldown": 0.0, "stats": dict(HELLO["config"]["weapons"]["mortar"])}
        me["weapons"] = [me["weapons"][0], mortar_w]

        # Position east of building prism 64 (-9.2..21.5 x -96..-78), enemy west at 140m (no LOS)
        mx, my = 45.0, -87.0
        mz = WORLD.ground_z(mx, my)
        ex, ey = -95.0, -87.0
        ez = WORLD.ground_z(ex, ey)
        me["pos"] = [mx, my, mz]
        muzzle_h = me["stats"]["muzzle_height"]
        self.assertFalse(WORLD.has_los((mx, my, mz + muzzle_h), (ex, ey, ez + 4.5)))

        # Flat arc blocked by building prism 64, high arc clears over it
        sol_high = bot.solve_mortar(me, [ex, ey, ez], [0.0, 0.0, 0.0], mortar_w, prefer_high=True)
        self.assertIsNotNone(sol_high)
        m_yaw, m_pitch, _ = sol_high
        self.assertGreater(m_pitch, 0.8)

        # In combat(), enemy without LOS still gets aimed and fired at by mortar when aligned
        o["mechs"] = [enemy(7, (ex, ey, ez))]
        bot.update_memory(o)
        me["aim"] = [m_yaw, m_pitch]
        ac = bot.combat(o, me)
        self.assertEqual(ac["fire"], [False, True])

    def test_mortar_combat_priority_cooldown_and_memory_lob(self):
        old = dict(B.FLAGS)
        B.FLAGS["mortar"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            bot.world = None
            o = make_obs(in_zone=True)
            o["t"] = 30.0
            me = o["self"]
            x, y, z = me["pos"]
            mortar_w = {"kind": "mortar", "cooldown": 0.0, "stats": dict(HELLO["config"]["weapons"]["mortar"])}
            me["weapons"] = [me["weapons"][0], mortar_w]

            # Ready mortar aims ballistic pitch (> gun pitch) and fires when aligned
            o["mechs"] = [enemy(7, (x + 100.0, y, z))]
            a1 = bot.combat(o, me)
            self.assertGreater(a1["aim"][1], 0.08)
            me["aim"] = a1["aim"]
            a2 = bot.combat(o, me)
            self.assertTrue(a2["fire"][1])

            # On cooldown: never steals aim from gun
            mortar_w["cooldown"] = 1.5
            a_cd = bot.combat(o, me)
            self.assertLess(a_cd["aim"][1], 0.0)
            me["aim"] = a_cd["aim"]
            a_cd2 = bot.combat(o, me)
            self.assertEqual(a_cd2["fire"], [True, False])

            # Remembered enemy lob when target is None (age <= 3.5s, 18 <= dist <= 190)
            mortar_w["cooldown"] = 0.0
            o["mechs"] = []
            bot.seen = {7: (o["t"] - 1.5, [x + 140.0, y, z], [0.0, 0.0, 0.0])}
            a_mem = bot.combat(o, me)
            self.assertGreater(a_mem["aim"][1], 0.8)
            me["aim"] = a_mem["aim"]
            a_mem2 = bot.combat(o, me)
            self.assertEqual(a_mem2["fire"], [False, True])

            # Stale memory (> 3.5s) does not fire
            bot.seen = {7: (o["t"] - 4.0, [x + 140.0, y, z], [0.0, 0.0, 0.0])}
            a_stale = bot.combat(o, me)
            self.assertEqual(a_stale["fire"], [False, False])

            # Disabled flag: mortar does not fire
            B.FLAGS["mortar"] = False
            o["mechs"] = [enemy(7, (x + 80.0, y, z))]
            me["aim"] = [0.0, 0.0]
            a_off = bot.combat(o, me)
            self.assertFalse(a_off["fire"][1])
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_weapon_tri_priority(self):
        old = dict(B.FLAGS)
        B.FLAGS["weapon_tri"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            bot.world = None
            o = make_obs(in_zone=True)
            me = o["self"]
            x, y, z = me["pos"]

            # 1. Laser prioritizes mortar enemy (0.65x) over closer gun-only enemy
            me["weapons"] = [me["weapons"][0], {"kind": "laser", "cooldown": 0, "charge": 0,
                                                "stats": dict(HELLO["config"]["weapons"]["laser"])}]
            e_gun = enemy(1, (x + 80.0, y, z))
            e_mortar = enemy(2, (x + 100.0, y, z))
            e_mortar["weapons"] = ["gun", "mortar"]
            o["mechs"] = [e_gun, e_mortar]
            self.assertEqual(bot.pick_target(o, me)["id"], 2)

            # 2. Lightning prioritizes charging laser (0.55x) and non-charging laser (0.72x) within 50m
            me["weapons"] = [me["weapons"][0], {"kind": "lightning", "cooldown": 0,
                                                "stats": dict(HELLO["config"]["weapons"]["lightning"])}]
            e_close_gun = enemy(3, (x + 30.0, y, z))
            e_laser = enemy(4, (x + 38.0, y, z))
            e_laser["weapons"] = ["gun", "laser"]
            e_laser["charging"] = False
            o["mechs"] = [e_close_gun, e_laser]
            self.assertEqual(bot.pick_target(o, me)["id"], 4)
            e_laser["charging"] = True
            e_laser["pos"] = [x + 48.0, y, z]
            self.assertEqual(bot.pick_target(o, me)["id"], 4)

            # 3. Mortar prioritizes lightning (0.75x in 16..160m) and charging laser (0.80x)
            me["weapons"] = [me["weapons"][0], {"kind": "mortar", "cooldown": 0,
                                                "stats": dict(HELLO["config"]["weapons"]["mortar"])}]
            e_gun2 = enemy(5, (x + 70.0, y, z))
            e_light = enemy(6, (x + 85.0, y, z))
            e_light["weapons"] = ["gun", "lightning"]
            o["mechs"] = [e_gun2, e_light]
            self.assertEqual(bot.pick_target(o, me)["id"], 6)

            e_ch_laser = enemy(7, (x + 82.0, y, z))
            e_ch_laser["weapons"] = ["gun", "laser"]
            e_ch_laser["charging"] = True
            o["mechs"] = [e_gun2, e_ch_laser]
            self.assertEqual(bot.pick_target(o, me)["id"], 7)

            # 4. Flag disabled: closer gun-only enemy wins
            B.FLAGS["weapon_tri"] = False
            self.assertEqual(bot.pick_target(o, me)["id"], 5)
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_swap_slot1_sends_take_and_drop(self):
        old = dict(B.FLAGS)
        B.FLAGS["weapon_tri"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True, hp=1400, max_hp=1400)
            o["t"] = 20.0
            me = o["self"]
            x, y, z = me["pos"]
            o["zone"].update(next_center=[x, y], next_radius=200.0)
            o["mechs"] = []
            mortar_w = {"kind": "mortar", "cooldown": 0.0, "stats": dict(HELLO["config"]["weapons"]["mortar"])}
            me["weapons"] = [me["weapons"][0], mortar_w]

            # Mortar (18) -> Laser (30) at feet: swaps slot 1
            o["pickups"] = [{"id": 9101, "kind": "laser", "pos": [x + 2.0, y, z]}]
            self.assertEqual(bot.choose_take(o, me), {"take": 9101, "drop": 1})

            # Mortar (18) -> Lightning (22) in open zone (delta 4 < 6): no swap
            bot.last_swap_t = -99.0
            o["pickups"] = [{"id": 9102, "kind": "lightning", "pos": [x + 2.0, y, z]}]
            self.assertEqual(bot.choose_take(o, me), {})

            # Small late zone (next_radius=60): Mortar (12) -> Lightning (32) detours up to 14m, swaps at feet
            o["zone"]["next_radius"] = 60.0
            o["pickups"] = [{"id": 9103, "kind": "lightning", "pos": [x + 10.0, y, z]}]
            self.assertEqual(bot.choose_take(o, me), {})
            self.assertEqual(bot.loot_goal, (x + 10.0, y))

            o["pickups"] = [{"id": 9104, "kind": "lightning", "pos": [x + 2.0, y, z]}]
            self.assertEqual(bot.choose_take(o, me), {"take": 9104, "drop": 1})

            # Active combat guard: enemy at 25m blocks even feet-swap; enemy at 40m allows feet-swap only when HP > 50%
            bot.last_swap_t = -99.0
            o["mechs"] = [enemy(8, (x + 25.0, y, z))]
            self.assertEqual(bot.choose_take(o, me), {})
            o["mechs"] = [enemy(8, (x + 40.0, y, z))]
            me["hp"] = 600.0  # < 50%
            self.assertEqual(bot.choose_take(o, me), {})
            me["hp"] = 1000.0  # > 50%
            self.assertEqual(bot.choose_take(o, me), {"take": 9104, "drop": 1})
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_no_swap_while_laser_charging(self):
        old = dict(B.FLAGS)
        B.FLAGS["weapon_tri"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            # Force a large score advantage for lightning so only the laser-charging gate blocks swap
            bot.weapon_score = lambda kind, obs: 40.0 if kind == "lightning" else 30.0
            o = make_obs(in_zone=True, hp=1400, max_hp=1400)
            o["t"] = 25.0
            me = o["self"]
            x, y, z = me["pos"]
            o["zone"].update(next_center=[x, y], next_radius=60.0)
            o["mechs"] = []
            laser_w = {"kind": "laser", "cooldown": 0.0, "charge": 0.5,
                       "stats": dict(HELLO["config"]["weapons"]["laser"])}
            me["weapons"] = [me["weapons"][0], laser_w]
            o["pickups"] = [{"id": 9201, "kind": "lightning", "pos": [x + 2.0, y, z]}]

            self.assertEqual(bot.choose_take(o, me), {})
            self.assertIsNone(bot.loot_goal)

            laser_w["charge"] = 0.0
            self.assertEqual(bot.choose_take(o, me), {"take": 9201, "drop": 1})
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_no_swap_ping_pong(self):
        old = dict(B.FLAGS)
        B.FLAGS["weapon_tri"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True, hp=1400, max_hp=1400)
            o["t"] = 10.0
            me = o["self"]
            x, y, z = me["pos"]
            o["zone"].update(next_center=[x, y], next_radius=200.0)
            o["mechs"] = []
            mortar_w = {"kind": "mortar", "cooldown": 0.0, "stats": dict(HELLO["config"]["weapons"]["mortar"])}
            me["weapons"] = [me["weapons"][0], mortar_w]
            o["pickups"] = [{"id": 9301, "kind": "laser", "pos": [x + 2.0, y, z]}]

            # Simulate recent "took" event at t=10.0
            o["events"] = [{"type": "took", "id": 9300, "slot": 1}]
            bot.update_memory(o)
            o["events"] = []

            # Within 4.0s cooldown: blocked
            o["t"] = 12.5
            self.assertEqual(bot.choose_take(o, me), {})
            self.assertIsNone(bot.loot_goal)

            # After 4.0s cooldown: allowed
            o["t"] = 14.5
            self.assertEqual(bot.choose_take(o, me), {"take": 9301, "drop": 1})
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_drop_ignored_when_far_early_or_contested(self):
        old = dict(B.FLAGS)
        B.FLAGS["drops"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True, hp=1400, max_hp=1400)
            o["t"] = 30.0
            me = o["self"]
            x, y, z = me["pos"]
            o["mechs"] = []
            cx, cy, inner = x, y, 160.0

            # Far (> 85m)
            o["drops"] = [{"id": 1, "pos": [x + 90.0, y, z], "radius": 8.0, "lands_in": 2.0, "holder": None, "progress": 0.0, "capture_s": 4.0}]
            self.assertIsNone(bot.choose_drop(o, me, cx, cy, inner))

            # Too early (lands_in > 8.0s)
            o["drops"] = [{"id": 2, "pos": [x + 30.0, y, z], "radius": 8.0, "lands_in": 9.5, "holder": None, "progress": 0.0, "capture_s": 4.0}]
            self.assertIsNone(bot.choose_drop(o, me, cx, cy, inner))

            # Outside inner
            o["drops"] = [{"id": 3, "pos": [x + 40.0, y, z], "radius": 8.0, "lands_in": 1.0, "holder": None, "progress": 0.0, "capture_s": 4.0}]
            self.assertIsNone(bot.choose_drop(o, me, cx, cy, 30.0))

            # Unwinnable race against another holder (rem_cap <= eta + 0.3)
            o["drops"] = [{"id": 4, "pos": [x + 50.0, y, z], "radius": 8.0, "lands_in": 0.0, "holder": 7, "progress": 3.2, "capture_s": 4.0}]
            self.assertIsNone(bot.choose_drop(o, me, cx, cy, inner))

            # Contested by 2 unique enemies within 35m (1 visible + 1 in seen)
            o["drops"] = [{"id": 5, "pos": [x + 30.0, y, z], "radius": 8.0, "lands_in": 1.0, "holder": None, "progress": 0.0, "capture_s": 4.0}]
            o["mechs"] = [enemy(7, (x + 40.0, y, z))]
            bot.seen[8] = (o["t"] - 1.0, [x + 20.0, y, z], [0.0, 0.0, 0.0])
            self.assertIsNone(bot.choose_drop(o, me, cx, cy, inner))

            # 1 enemy near: rejected if low HP (< 35%), accepted if healthy
            bot.seen.clear()
            me["hp"] = 400.0
            self.assertIsNone(bot.choose_drop(o, me, cx, cy, inner))
            me["hp"] = 1000.0
            pick = bot.choose_drop(o, me, cx, cy, inner)
            self.assertIsNotNone(pick)
            self.assertEqual(pick[0]["id"], 5)

            # Flag disabled
            B.FLAGS["drops"] = False
            self.assertIsNone(bot.choose_drop(o, me, cx, cy, inner))
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_drop_routes_with_astar_and_holds_in_ring(self):
        old = dict(B.FLAGS)
        B.FLAGS["drops"] = True
        B.FLAGS["astar_nav"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True, hp=1400, max_hp=1400)
            o["t"] = 40.0
            me = o["self"]
            # Start east of building prism 64, drop west of prism 64 (dist ~58.7m <= 85m)
            mx, my = 29.5, -87.0
            mz = WORLD.ground_z(mx, my)
            dx, dy = -29.2, -87.0
            dz = WORLD.ground_z(dx, dy)
            me["pos"] = [mx, my, mz]
            o["zone"].update(center=[0.0, -87.0], radius=300.0, next_center=[0.0, -87.0], next_radius=200.0)
            o["mechs"] = []
            o["cubes"] = []
            o["pickups"] = []
            o["crates"] = []
            o["events"] = []
            o["drops"] = [{"id": 50, "pos": [dx, dy, dz], "radius": 8.0, "lands_in": 2.0, "holder": None, "progress": 0.0, "capture_s": 4.0}]

            a = bot.movement(o, me)
            self.assertGreater(math.hypot(*a["move"]), 0.5)
            self.assertFalse(a["boost"])
            self.assertIsNotNone(bot.nav_path)

            # Inside ring (dist <= radius - 2.0): holds still even if d > settle
            me["pos"] = [dx + 3.0, dy, dz]
            o["zone"].update(next_center=[dx + 120.0, dy], next_radius=200.0)
            a_hold = bot.movement(o, me)
            self.assertEqual(a_hold["move"], [0.0, 0.0])

            # Active capture hold (holding_drop): skips choose_cover even when threatened
            o["drops"] = [{"id": 50, "pos": [dx, dy, dz], "radius": 8.0, "lands_in": 0.0,
                           "holder": me["id"], "progress": 1.5, "capture_s": 4.0}]
            bot.threat = (o["t"], 0.0)
            a_cap = bot.movement(o, me)
            self.assertEqual(a_cap["move"], [0.0, 0.0])
            self.assertIsNone(bot.cover_goal)
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_drop_evade_still_preempts_ring_hold(self):
        old = dict(B.FLAGS)
        B.FLAGS["drops"] = True
        B.FLAGS["evade"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True, hp=1400, max_hp=1400, energy=80.0)
            o["t"] = 45.0
            me = o["self"]
            x, y, z = me["pos"]
            o["zone"].update(center=[x, y], radius=400.0, next_center=[x, y], next_radius=200.0)
            o["mechs"] = []
            o["cubes"] = []
            o["pickups"] = []
            o["events"] = []
            o["drops"] = [{"id": 60, "pos": [x + 1.0, y, z], "radius": 8.0, "lands_in": 0.0,
                           "holder": me["id"], "progress": 2.5, "capture_s": 4.0}]
            # Incoming mortar shell landing right on us
            o["projectiles"] = [
                {"id": 601, "kind": "mortar", "pos": [x - 60.0, y, z + 15.0], "vel": [60.0, 0.0, 5.0]}
            ]
            a = bot.movement(o, me)
            self.assertTrue(bot.evading)
            self.assertGreater(math.hypot(*a["move"]), 0.5)
            self.assertTrue(a["boost"])
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_corner_cover_precompute_and_query(self):
        old = dict(B.FLAGS)
        B.FLAGS.update(cover=True, corner_cover=True)
        try:
            self.assertGreater(len(WORLD.corner_nodes), 50)
            for cx, cy in WORLD.corner_nodes[:50]:
                self.assertTrue(WORLD.is_walkable(cx, cy))

            mx, my = 29.5, -74.0
            near = WORLD.nearby_corners(mx, my, max_r=42.0, limit=20)
            self.assertGreater(len(near), 0)
            self.assertLessEqual(len(near), 20)
            dists = [math.hypot(px - mx, py - my) for px, py in near]
            self.assertEqual(dists, sorted(dists))
            for d in dists:
                self.assertGreaterEqual(d, 2.0 - 1e-6)
                self.assertLessEqual(d, 42.0 + 1e-6)

            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True)
            mz = WORLD.ground_z(mx, my)
            o["self"]["pos"] = [mx, my, mz]
            o["zone"].update(next_center=[mx, my], next_radius=200.0)
            ex, ey = -29.2, -74.0
            ez = WORLD.ground_z(ex, ey)
            o["mechs"] = [enemy(7, (ex, ey, ez))]
            o["events"] = [{"type": "damaged", "amount": 25, "weapon": "gun",
                            "bearing": math.atan2(ey - my, ex - mx), "by": 7}]
            bot.update_memory(o)
            goal = bot.choose_cover(o, o["self"], mx, my, 160.0)
            self.assertIsNotNone(goal)
            self.assertIn(goal, near)
            gz = WORLD.support_z(goal[0], goal[1], mz)
            eye_h = o["self"]["stats"]["eye_height"]
            self.assertFalse(WORLD.has_los((goal[0], goal[1], gz + eye_h), (ex, ey, ez + 8.0)))
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_laser_opt_early_release_and_zone_drop(self):
        old = dict(B.FLAGS)
        B.FLAGS["laser_opt"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            bot.world = None
            o = make_obs(in_zone=True, cubes=0)
            x, y, z = o["self"]["pos"]
            laser = {"kind": "laser", "cooldown": 0.0, "charge": 0.4,
                     "stats": dict(HELLO["config"]["weapons"]["laser"])}
            o["self"]["weapons"] = [None, laser]
            o["self"]["aim"] = [0.0, 0.0]

            # Early release finisher: charge=0.4 -> est_dmg ~ 219 >= 180 HP -> release (fire=False)
            e = enemy(5, (x + 100.0, y, z))
            e["hp"] = 180.0
            o["mechs"] = [e]
            self.assertFalse(bot.combat(o, o["self"])["fire"][1])

            # High HP (500 > est_dmg): keeps charging (fire=True)
            e["hp"] = 500.0
            self.assertTrue(bot.combat(o, o["self"])["fire"][1])

            # Out of zone: drops charge (fire=False) both with target and without target
            o["self"]["in_zone"] = False
            self.assertFalse(bot.combat(o, o["self"])["fire"][1])
            o["mechs"] = []
            self.assertFalse(bot.combat(o, o["self"])["fire"][1])

            # Evading or late inside zone: drops charge
            o["self"]["in_zone"] = True
            o["mechs"] = [e]
            bot.evading = True
            self.assertFalse(bot.combat(o, o["self"])["fire"][1])
            bot.evading = False
            bot.late = True
            self.assertFalse(bot.combat(o, o["self"])["fire"][1])
            bot.late = False

            # Flag disabled: restores baseline hold behavior
            B.FLAGS["laser_opt"] = False
            e["hp"] = 180.0
            self.assertTrue(bot.combat(o, o["self"])["fire"][1])
            o["self"]["in_zone"] = False
            o["mechs"] = []
            self.assertTrue(bot.combat(o, o["self"])["fire"][1])
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_rear_guard_priority_and_retreat(self):
        old = dict(B.FLAGS)
        B.FLAGS["rear_guard"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True)
            o["t"] = 20.0
            x, y, z = o["self"]["pos"]
            o["mechs"] = []
            o["events"] = []
            # Priority 3: near sounds (weapon prio 3 beats step prio 2)
            o["sounds"] = [
                {"kind": "step", "bearing": 0.5, "dist": "near", "age": 0.05},
                {"kind": "gun", "bearing": -1.2, "dist": "near", "age": 0.05},
            ]
            bot.update_memory(o)
            self.assertAlmostEqual(bot.rear_guard_yaw(o, o["self"]), -1.2, places=5)
            self.assertAlmostEqual(bot.idle_aim(o, o["self"])[0], -1.2, places=5)

            # Priority 2: closest seen enemy beats near sound
            bot.seen[4] = (o["t"] - 0.5, [x, y + 60.0, z], [0.0, 0.0, 0.0])  # north = pi/2
            bot.seen[5] = (o["t"] - 0.5, [x + 120.0, y, z], [0.0, 0.0, 0.0])  # east = 0
            self.assertAlmostEqual(bot.rear_guard_yaw(o, o["self"]), math.pi / 2, places=4)

            # Priority 1: recent damaged bearing beats seen enemy
            bot.threat = (o["t"] - 0.2, -2.5)
            self.assertAlmostEqual(bot.rear_guard_yaw(o, o["self"]), -2.5, places=5)

            # Retreat without LOS in combat(): locks aim to rear_guard_yaw
            mx, my = 29.5, -87.0
            mz = WORLD.ground_z(mx, my)
            ex, ey = -29.2, -87.0
            ez = WORLD.ground_z(ex, ey)
            o["self"]["pos"] = [mx, my, mz]
            o["self"]["in_zone"] = False
            o["mechs"] = [enemy(8, (ex, ey, ez))]
            bot.threat = (o["t"] - 0.1, 0.75)
            ac = bot.combat(o, o["self"])
            self.assertEqual(ac["fire"], [False, False])
            self.assertAlmostEqual(ac["aim"][0], 0.75, places=5)
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_energy_mgmt_lightning_and_boost(self):
        old = dict(B.FLAGS)
        B.FLAGS["energy_mgmt"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            bot.world = None
            o = make_obs(in_zone=True)
            me = o["self"]
            x, y, z = me["pos"]
            l_stats = dict(HELLO["config"]["weapons"]["lightning"])
            cost = l_stats["energy"]
            me["weapons"] = [None, {"kind": "lightning", "cooldown": 0.0, "stats": l_stats}]
            me["aim"] = [0.0, 0.0]

            # Non-urgent target at 35m, 1000 HP, not charging: requires cost + 30 energy
            e = enemy(6, (x + 35.0, y, z))
            e["hp"] = 1000.0
            o["mechs"] = [e]
            me["energy"] = cost + 25.0
            self.assertFalse(bot.combat(o, me)["fire"][1])
            me["energy"] = cost + 30.0
            self.assertTrue(bot.combat(o, me)["fire"][1])

            # Urgent zap (low HP < 180, or dist < 25m, or charging): fires at cost + 2
            me["energy"] = cost + 2.0
            e["hp"] = 150.0
            self.assertTrue(bot.combat(o, me)["fire"][1])
            e["hp"] = 1000.0
            e["charging"] = True
            self.assertTrue(bot.combat(o, me)["fire"][1])
            e["charging"] = False
            e["pos"] = [x + 20.0, y, z]
            self.assertTrue(bot.combat(o, me)["fire"][1])

            # Boost energy reserve: non-urgent kite boost needs energy >= ENERGY_RESERVE + 15 (45.0)
            B.FLAGS["kite"] = True
            bot2 = B.Bot(copy.deepcopy(HELLO))
            o2 = make_obs(in_zone=True, energy=40.0)
            ox, oy, oz = o2["self"]["pos"]
            o2["zone"].update(center=[ox, oy], radius=400.0, next_center=[ox, oy], next_radius=300.0)
            o2["cubes"] = []
            o2["pickups"] = []
            o2["mechs"] = [enemy(9, (ox + 35.0, oy, oz))]
            bot2.choose_cover = lambda *args, **kwargs: None
            self.assertFalse(bot2.movement(o2, o2["self"])["boost"])
            o2["self"]["energy"] = 50.0
            self.assertTrue(bot2.movement(o2, o2["self"])["boost"])

            # Urgent boost (outside zone): still boosts at low energy
            o2["self"]["in_zone"] = False
            o2["self"]["energy"] = 20.0
            self.assertTrue(bot2.movement(o2, o2["self"])["boost"])
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_smart_upgrades_loadout_aware(self):
        old = dict(B.FLAGS)
        B.FLAGS["smart_upgrades"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs()
            def offer(*ids):
                o["offer"] = {"id": 1, "expires": 9, "choices": [{"id": i, "name": "", "desc": ""} for i in ids]}
                return bot.choose_pick(o)["pick"]

            # Gun only: unheld laser_charge (10) loses to servos (55) and jets (30) and unknown (20)
            self.assertEqual(offer("laser_charge", "servos", "jets"), 1)
            self.assertEqual(offer("laser_charge", "zzz", "mortar_reload"), 1)

            # Holding laser: laser_charge (82) beats gun_rate (75), but armor (100) beats laser_charge
            o["self"]["weapons"] = [o["self"]["weapons"][0], {"kind": "laser", "cooldown": 0, "charge": 0,
                                                              "stats": dict(HELLO["config"]["weapons"]["laser"])}]
            self.assertEqual(offer("gun_rate", "laser_charge", "servos"), 1)
            self.assertEqual(offer("laser_charge", "armor", "plating"), 1)

            # Holding mortar: mortar_reload (78) beats gun_rate (75) and unheld lightning_range (10)
            o["self"]["weapons"] = [o["self"]["weapons"][0], {"kind": "mortar", "cooldown": 0,
                                                              "stats": dict(HELLO["config"]["weapons"]["mortar"])}]
            self.assertEqual(offer("lightning_range", "gun_rate", "mortar_reload"), 2)

            # Holding lightning: lightning_range (78) beats gun_rate (75)
            o["self"]["weapons"] = [o["self"]["weapons"][0], {"kind": "lightning", "cooldown": 0,
                                                              "stats": dict(HELLO["config"]["weapons"]["lightning"])}]
            self.assertEqual(offer("gun_rate", "lightning_range", "servos"), 1)

            # Flag disabled: falls back to UPGRADE_PREF (gun_rate beats lightning_range)
            B.FLAGS["smart_upgrades"] = False
            self.assertEqual(offer("lightning_range", "gun_rate", "servos"), 1)
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_rear_guard_preserves_laser_charge_aim(self):
        old = dict(B.FLAGS)
        B.FLAGS["rear_guard"] = True
        B.FLAGS["laser_opt"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=False, hp=1400.0, max_hp=1400.0)
            mx, my, mz = o["self"]["pos"]
            laser_st = dict(HELLO["config"]["weapons"]["laser"])
            o["self"]["weapons"] = [
                o["self"]["weapons"][0],
                {"kind": "laser", "cooldown": 0.0, "charge": 0.45, "stats": laser_st},
            ]
            # Target directly east (+X), occluded, with a rear threat directly west (-X)
            e = enemy(7, (mx + 60.0, my, mz))
            o["mechs"] = [e]
            bot.threat = (o["t"], math.pi)
            bot.target_has_los = False
            act = bot.combat(o, o["self"])
            # Rear guard must NOT hijack aim to pi while laser has active charge
            self.assertLess(abs(B.wrap(act["aim"][0] - 0.0)), 0.3)
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_laser_cookoff_early_release_vs_mortar_carrier(self):
        old = dict(B.FLAGS)
        B.FLAGS["laser_opt"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True, hp=1400.0, max_hp=1400.0)
            mx, my, mz = o["self"]["pos"]
            laser_st = dict(HELLO["config"]["weapons"]["laser"])
            # At charge=0.35 (full=0.75, min_frac=0.35, dmg=315): base est_dmg ~ 206; with cookoff 0.6 -> ~330
            o["self"]["weapons"] = [
                o["self"]["weapons"][0],
                {"kind": "laser", "cooldown": 0.0, "charge": 0.35, "stats": laser_st},
            ]
            e = enemy(7, (mx + 50.0, my, mz))
            e["hp"] = 280.0
            e["weapons"] = ["gun"]
            o["mechs"] = [e]
            o["self"]["aim"] = [0.0, 0.0]
            bot.target_has_los = True
            act_gun = bot.combat(o, o["self"])
            self.assertTrue(act_gun["fire"][1], "Should keep charging against 280 HP gun-only target")

            e["weapons"] = ["gun", "mortar"]
            act_mortar = bot.combat(o, o["self"])
            self.assertFalse(act_mortar["fire"][1], "Should early-release laser against 280 HP mortar carrier via cookoff")
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_weapon_dps_quantized_cd_interleave_and_mortar_gate(self):
        old = dict(B.FLAGS)
        B.FLAGS["weapon_dps"] = True
        B.FLAGS["mortar"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            o = make_obs(in_zone=True, hp=1400.0, max_hp=1400.0, energy=100.0)
            mx, my, mz = o["self"]["pos"]
            # 1. Tick-quantized lightning cooldown (0.10 <= 0.105 fires)
            lt_st = dict(HELLO["config"]["weapons"]["lightning"])
            o["self"]["weapons"] = [
                o["self"]["weapons"][0],
                {"kind": "lightning", "cooldown": 0.10, "stats": lt_st},
            ]
            e = enemy(7, (mx + 30.0, my, mz))
            o["mechs"] = [e]
            o["self"]["aim"] = [0.0, 0.0]
            bot.target_has_los = True
            act_lt = bot.combat(o, o["self"])
            self.assertTrue(act_lt["fire"][1])

            # 2. Gun-during-laser-charge interleaving + pre-release snap to direct
            laser_st = dict(HELLO["config"]["weapons"]["laser"])
            o["self"]["weapons"] = [
                o["self"]["weapons"][0],
                {"kind": "laser", "cooldown": 0.0, "charge": 0.20, "stats": laser_st},
            ]
            e_mv = enemy(7, (mx + 100.0, my, mz), vel=(0.0, 12.0, 0.0))
            o["mechs"] = [e_mv]
            act_mid = bot.combat(o, o["self"])
            self.assertGreater(act_mid["aim"][0], 0.02, "Should aim led during early laser charge")
            o["self"]["weapons"][1]["charge"] = 0.65
            act_snap = bot.combat(o, o["self"])
            self.assertLess(abs(act_snap["aim"][0]), 0.01, "Should snap back to direct 1 tick pre-release")

            # 3. Mortar LOS gate: moving target with ready gun keeps gun aim instead of mortar
            m_st = dict(HELLO["config"]["weapons"]["mortar"])
            o["self"]["weapons"] = [
                o["self"]["weapons"][0],
                {"kind": "mortar", "cooldown": 0.0, "stats": m_st},
            ]
            act_mv = bot.combat(o, o["self"])
            self.assertLess(act_mv["aim"][1], 0.1, "Should not steal LOS gun aim for moving target")
            e_mv["charging"] = True
            act_ch = bot.combat(o, o["self"])
            self.assertGreater(act_ch["aim"][1], 0.08, "Should fire mortar at charging laser target")
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)

    def test_tactical_micro_arc_partial_cover_bullet_surf_and_drop_overwatch(self):
        old = dict(B.FLAGS)
        B.FLAGS["tactical_micro"] = True
        B.FLAGS["cover"] = True
        B.FLAGS["drops"] = True
        try:
            bot = B.Bot(copy.deepcopy(HELLO))
            # 1. Angular span math
            self.assertAlmostEqual(B.Bot._angular_span(0.0, 0.0, [(10.0, 0.0), (-10.0, 0.0)]), math.pi, places=4)
            self.assertLess(B.Bot._angular_span(0.0, 10.0, [(10.0, 0.0), (-10.0, 0.0)]), math.pi * 0.6)

            # 2. Bullet-stream CPA surf preserves laser charge & respects final-circle zrad <= 6.0
            o = make_obs(in_zone=True, hp=1400.0, max_hp=1400.0)
            mx, my = 29.5, -74.0
            mz = WORLD.ground_z(mx, my)
            o["self"]["pos"] = [mx, my, mz]
            o["zone"].update(center=[mx, my], radius=200.0, next_center=[mx, my], next_radius=200.0)
            o["projectiles"] = [{"id": 99, "kind": "gun", "pos": [mx + 80.0, my, mz + 4.5], "vel": [-252.0, 0.0, 0.0]}]
            surf = bot.choose_bullet_surf(o, o["self"], mx, my)
            self.assertIsNotNone(surf)
            self.assertFalse(bot.evading)
            o["zone"]["radius"] = 5.0
            self.assertIsNone(bot.choose_bullet_surf(o, o["self"], mx, my))

            # 3. Staged drop overwatch stages 18-35m away when lands_in > 2.5s
            o["zone"].update(center=[mx, my], radius=200.0, next_center=[mx, my], next_radius=200.0)
            o["projectiles"] = []
            dr = {"id": 50, "pos": [mx + 40.0, my, mz], "radius": 8.0, "lands_in": 6.0, "holder": None, "progress": 0.0}
            stage = bot.choose_drop_overwatch(o, o["self"], dr, mx, my, 160.0)
            self.assertIsNotNone(stage)
            d_stage = math.hypot(stage[0] - dr["pos"][0], stage[1] - dr["pos"][1])
            self.assertGreaterEqual(d_stage, 16.0)
            self.assertLessEqual(d_stage, 36.0)
        finally:
            B.FLAGS.clear()
            B.FLAGS.update(old)


if __name__ == "__main__":
    unittest.main()
