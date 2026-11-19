# vora95/gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vora95/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vora95/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Dilution-aware forecaster for GTA VI trophy earn rates on Exophase: forecasts earners and players separately and divides, with logit-scale quantiles.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

## How the forecaster works

- **Earners over players, not the rate itself.** A trophy's earn rate is
  earners divided by players tracked. After launch the player base grows
  much faster than earners, so the rate dilutes even while earners rise.
  The model forecasts both counts separately and divides, so it captures
  that dilution instead of flat-lining the last rate.
- **Launch-break detection.** The history opens with a flat pre-launch
  segment (identical rates, flat player count) that carries no signal.
  Fitting starts at the first read where players jump past 1.5x the
  first read or any rate first moves.
- **Log-log least-squares extrapolation.** Both log players and
  log(earners + 0.5) are fitted by ordinary least squares against
  log(hours since the break + 1) and evaluated at the target time.
  The players forecast is clamped to between the last count and 4x it.
- **Logit-scale quantile spread.** The median rate is turned into a
  logit, and the five quantiles (0.05/0.25/0.5/0.75/0.95) are the
  inverse logit of the centre plus z times a spread that grows with
  the horizon and with how far the median drifted from the last rate.
  That is the scale the pinball-loss scoring uses, so wide uncertainty
  where the forecast moved most.
