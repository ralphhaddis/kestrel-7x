# kestrel-7x

Control agent (`bot.py`) and local test harness (`scripts/check.sh`, `tests/`).

Extract the official Mech Royale bot kit into `kit/` so that `kit/check/check.py`, `kit/mechfight.schema.json`, and `kit/arena/` are present. Run the following with Python 3 and Docker available.

```sh
./scripts/check.sh
```

## Local Simulator Benchmarking (Important Note for Agents)

Always run `tools/sim.py` with **1 copy of `bot:Bot`** (`bot:Bot=1,aggro=5,runner=5,example=5`, now the default):

```sh
python3 tools/sim.py --matches 48 --bots bot:Bot=1,aggro=5,runner=5,example=5
```

Do **not** benchmark survival/combat flags with `bot:Bot=4`: spawning 4 adjacent copies of Kestrel with 85%+ gun accuracy creates a 4-copy self-cannibalization / truce bias that artificially rewards passive ceasefire flags which drop score on the official 1v15 server.
