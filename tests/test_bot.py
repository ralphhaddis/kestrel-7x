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
        self.assertAlmostEqual(math.atan2(a["move"][1], a["move"][0]), want)
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


if __name__ == "__main__":
    unittest.main()
