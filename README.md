# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Gorustt/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Gorustt/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A calibrated forecaster for every GTA VI PlayStation 5 trophy's Exophase earn rate at the case target. It fits a per-trophy least-squares line of logit(rate) against log-time over the last 24 hours of reads, clamps the forecast inside physical bounds (earners only grow, so the target rate sits between the no-new-earner floor and the all-newcomers-earn ceiling) and tier-based saturation caps, and falls back to persistence with a wide spread when the history is flat — so a launch-night-style Exophase refresh still lands inside the 90% band. Spread combines a random-walk term, the fit residual, and a horizon-scaled model term. Python 3.13, standard library only, deterministic, no network.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` case in the input directory and writes one `<id>.out` forecast per case to the output directory.

---

<sub>Entered on [make.fun](https://make.fun/?ref=XQQQ-RMWD&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
