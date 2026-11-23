# kestrel-7x

An autonomous Mech Royale bot written for the **joinBuilders Hackathon** hosted by The Ohio State University.

`kestrel-7x` plays 16-mech free-for-all matches over the `mechfight/1` JSONL protocol. The entire bot is pure Python (standard library only) so it runs cleanly inside the match sandbox: 1 CPU, 512 MB RAM, no network, read-only root filesystem, and the 3D arena map mounted at `/arena`.

## Repo Layout

- `bot.py`: Main agent (`Bot`). Handles perception memory, upgrade selection, looting, evasion, cover, movement, and weapon control.
- `world_map.py`: Loads the static arena geometry (`v2/world.json` and `v2/terrain.i16`) at startup and provides spatial queries (terrain height, walkability, 3D line of sight, building corners, tree canopies, and A* routing).
- `mechfight.py`: Match protocol wrapper (`mechfight/1`). Manages the stdin/stdout message loop, drops stale observations when a tick falls behind, attaches optional debug drawings, and falls back to the last good move/aim/boost if `act()` ever raises.
- `Dockerfile`: Minimal `python:3.13-slim` container that precompiles `*.py` and runs `python3 -u /bot/bot.py`.
- `tests/test_bot.py`: `unittest` suite covering replay-stream schema conformance, `WorldMap` geometry/A*, aim math, evasion, cover selection, and weapon logic.
- `scripts/check.sh`: Runs the unit tests, local protocol conformance check, Docker build, and sandboxed Docker conformance check.

## Arena Geometry & Spatial Indexing (`world_map.py`)

Because the bot runs on a single CPU with tight tick budgets, `WorldMap` parses the map once on `hello` and builds a few lookup structures up front:

- **Terrain heightfield**: Reads `v2/terrain.i16` into a little-endian `array('h')` (1 m grid, 0.01 m vertical scale). `ground_z(x, y)` uses bilinear interpolation and `ground_slope_tan(x, y)` uses 1 m central finite differences to check against the mech's max walkable slope.
- **16 m prism grid**: Every building, stadium, and bridge polygon (including interior holes and `[base, top]` elevation intervals) is hashed into 16 m x 16 m buckets. `is_walkable` tests the mech cylinder against nearby prisms (allowing walking under bridges and across flat roofs), and `has_los` ray-marches 3D segments at 3 m steps against both prisms and terrain.
- **5 m navigation grid + weighted A\***: At startup, `_build_nav_grid()` precomputes a 1-byte-per-cell 2D walkability bitmap. When a straight line to a goal is blocked, `find_path` runs 8-connected weighted A* (octile heuristic weighted 1.5x, no diagonal corner-cutting around walls, capped expansions) and `next_waypoint` string-pulls along the path using `nav_line_clear`.
- **Precomputed tactical points**: Exterior convex building corners (offset outward by `mech_r + 1.5 m`, filtered for walls at least 6 m tall) and walkable tree-canopy centers (`crownR >= 2.2 m` where mech eye height sits inside the crown band) are pre-indexed into 16 m grids for fast lookups during cover and concealment queries.

## Per-Tick Control Loop (`bot.py`)

Every observation runs through five stages in order:

```text
update_memory -> choose_pick -> choose_take -> movement -> combat
```

### Memory & Perception (`update_memory`)
Tracks recently seen mechs (position, velocity, and HP fraction) within `vision.memory_s`, timestamps of incoming damage events and aggressor IDs, kills, and recent combat/movement audio bearings. When a real mech is visible or in memory, chip damage from NPCs is filtered out so NPCs cannot hijack threat facing or cover logic.

### Evasion, Cover & Movement (`movement`)
Movement resolves a single target waypoint through a strict priority order:

1. **Active evasion (`choose_evade`)**: Predicts landing time and splash radius for incoming mortar shells (iterating against `support_z` at the landing point) and checks perpendicular clearance to any enemy charging a laser. For charging lasers, the bot latches a committed lateral dodge side (`laser_dodge_dir`) for 0.9 s so it does not ping-pong across the beam centerline, scoring dodge candidates by clearance, LOS breaks, and multi-enemy angular span.
2. **Zone safety (`_safe_bound`)**: Computes a dynamic safe radius around `next_center`. While there is more than 14 s of walk slack before `shrink_end`, the bound expands past `0.8 * next_radius` so outer-ring spawns can still loot and take cover instead of sprinting straight into the middle of the map. In final circles (`next_radius <= 35 m`), the settle radius tightens to `0.32 * next_radius` and energy reserve increases.
3. **Supply drops & building cover (`choose_drop`, `choose_cover`)**: Captures reachable, uncontested (or winnable) supply drops when ETA and nearby-enemy checks pass. When under fire, or proactively at low HP (`< 58%` in 1v1, `< 88%` in 1v2+), `choose_cover` evaluates precomputed building corners first (up to 48 m), then radial sample rings (12-48 m), picking a reachable point that breaks 3D LOS to exposed threats. Hysteresis (`COVER_HOLD_S = 1.2 s`) prevents corner jitter, and holding behind cover while hurt extends `cover_until` so out-of-combat nanite regeneration can trigger.
4. **Open-field fallback**: If no cover exists, the bot falls back to low-HP kiting (`< 55%` HP), angular-span reduction when pinched between multiple enemies (`_angular_span`), diagonal backpedaling against close-range rushers (`< 42 m` without lightning), or perpendicular post-hit strafing.
5. **Loot, kills, and farming**: Outside active threats, movement falls through to weapon pickups, cube collection (up to the 10-cube stat cap), chasing a single wounded 1v1 target (`< 55%` HP) before it can regenerate, safe crate/NPC farming when no mechs are nearby, and ducking under nearby tree canopies when idle with a sound or recent contact nearby.

Whenever direct travel fails `nav_line_clear`, movement routes through cached A* paths with stuck-waypoint blacklisting and a jump/boost unstick maneuver if ground speed stalls below 1 m/s for 1 s.

### Combat & Weapon Logic (`combat`)
- **Target priority (`pick_target`)**: Scores enemies by `dist * (0.30 + 0.70 * hp_frac)` to focus-fire wounded targets without ignoring point-blank threats, applies a heavy penalty when LOS is blocked, prioritizes active supply-drop holders, and weights weapon-triangle matchups (laser vs. mortar for cookoff bonus, lightning vs. charging laser for short-circuit, mortar vs. lightning/charging laser).
- **Gun**: Uses 3-iteration velocity lead (`aim_at`) and gates shots on both 3D LOS and angular hitbox alignment (`aligned`). When holding a laser, the bot keeps firing the gun with lead aim while the laser charges, then snaps to direct aim right before release.
- **Laser**: Estimates damage including cube scaling and the mortar cookoff multiplier to fire early finishers once charge is enough to kill. Drops charge early if occluded too long, evading, or running to cover so the charging movement slowdown does not get the mech trapped.
- **Mortar**: Uses a closed-form ballistic pitch solver (`solve_mortar_pitch`) with 1-step barrel-offset refinement and damped velocity lead. Fires low-arc trajectories when LOS is clear and high-arc over-building lobs when occluded (including against recently remembered positions in `self.seen`), verifying trajectory clearance against terrain and buildings in `mortar_arc_clear`.
- **Lightning**: Fires within 95% of range and 40% of the cone angle while preserving energy for jets (`ENERGY_RESERVE = 30`, raised to 38 in tight endgames), bypassing the reserve only for low-HP finishers, point-blank fights (`< 25 m`), or interrupting a charging laser.
- **Rear guard**: When idle or retreating without LOS, the torso aims toward the last damage bearing, closest remembered enemy, or recent near/mid-range sound to avoid taking rear-arc multiplier hits.

### Loot & Upgrades (`choose_take`, `choose_pick`)
Weapon scoring adapts to circle size (preferring laser and mortar in open rings and swapping toward lightning once `next_radius <= 70 m`). Pickup detours validate pickup height against `world.ground_z(px, py)` so rooftop spawns are ignored while hillside pickups are accepted, and blacklist pickups if progress stalls. Upgrade selection scores offers against the current loadout, prioritizing sustain (`nanites`, `armor`, `plating`) and DPS upgrades for weapons currently held.

### Feature Flags (`FLAGS`)
Most behaviors in `bot.py` sit behind a `FLAGS` dictionary added over 15 iterations so individual changes could be ablated cleanly. Several experimental mechanics (`early_weapon`, `safe_laser`, `heal`, `kite`, `proactive_cover`, `combat_strafe`, `selective_fire`, `w9_squad_hunter`, `w11_combat_juke`, `w12_bullet_evade`) are intentionally kept `False` after local and server testing showed they either hurt DPS or reduced placement.

## Running & Testing

To run the full test and conformance pipeline, extract the official Mech Royale bot kit into `kit/` so that `kit/check/check.py`, `kit/mechfight.schema.json`, `kit/check/stream.jsonl.gz`, and `kit/arena/` are present, then run:

```sh
./scripts/check.sh
```

That runs:
1. `python3 -m unittest discover -s tests -q`
2. Local conformance check (`python3 kit/check/check.py -- python3 -u bot.py`)
3. `docker build -t kestrel-7x .`
4. Sandboxed conformance check inside Docker (`--network none --cpus 1 --memory 512m --read-only --user 65534:65534`)

To build the image on its own without the kit:

```sh
docker build -t kestrel-7x .
```
