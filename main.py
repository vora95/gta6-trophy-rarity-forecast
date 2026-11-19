"""GTA VI trophy rarity forecaster.

Forecasts, for each trophy, five quantiles of its Exophase earn rate
(percent of tracked players who have earned it) at a target time.

Key idea -- dilution: the earn rate is earners divided by players.
After launch the tracked player count grows fast while earners grow
more slowly, so the rate falls even as more people earn the trophy.
We therefore forecast earners and players SEPARATELY (each with a
log-log least-squares trend) and divide, instead of extrapolating
the rate itself.  Quantiles live on the logit scale, which is the
scale the pinball-loss scoring uses.

Standard library only. No network, no clock, no randomness:
running twice on the same input gives byte-identical output.
"""

import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_VALUES = [-1.645, -0.674, 0.0, 0.674, 1.645]


def parse_time(text):
    """Parse an ISO-8601 timestamp (with a trailing Z or an offset)."""
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def clamp_pct(value):
    """Clamp a percentage into [0.01, 99.99] for the logit scale."""
    return min(99.99, max(0.01, float(value)))


def logit(pct):
    """Natural log of p / (100 - p), with p clamped first."""
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(level):
    """Inverse logit: 100 / (1 + exp(-L)), guarded against overflow."""
    if level >= 700.0:
        return 100.0
    if level <= -700.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-level))


def round4(value):
    """Round to four decimals, half-up, returned as a plain float."""
    rounded = Decimal(repr(float(value))).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP
    )
    return float(rounded)


def least_squares(xs, ys):
    """Ordinary least squares line; returns (slope, intercept) or None."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0.0:
        return None
    numer = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = numer / denom
    return slope, mean_y - slope * mean_x


def log_hours(base, moment):
    """log(hours from base to moment + 1); never negative."""
    if base is None or moment is None:
        return 0.0
    seconds = (moment - base).total_seconds()
    hours = max(0.0, seconds / 3600.0)
    return math.log(hours + 1.0)


def read_players(read):
    """players_tracked as a positive float, or None."""
    value = read.get("players_tracked")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return float(value)


def read_rate(read, index):
    """The rate for trophy `index` in a read, or None."""
    rates = read.get("rates")
    if not isinstance(rates, list) or index >= len(rates):
        return None
    value = rates[index]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def find_break(history):
    """Index of the first post-launch read; 0 when there is no break.

    The seed history opens with a flat pre-launch segment (identical
    rates, flat player count) that carries no signal. The break is the
    first read at index >= 1 where players_tracked exceeds 1.5x the
    first read's count, or where any trophy's rate differs from its
    rate in the first read, whichever comes first.
    """
    if len(history) < 2:
        return 0
    first = history[0]
    first_players = read_players(first)
    first_rates = first.get("rates") if isinstance(first.get("rates"), list) else []
    for i in range(1, len(history)):
        read = history[i]
        players = read_players(read)
        if first_players is not None and players is not None:
            if players > 1.5 * first_players:
                return i
        rates = read.get("rates") if isinstance(read.get("rates"), list) else []
        for j in range(max(len(first_rates), len(rates))):
            old = first_rates[j] if j < len(first_rates) else None
            new = rates[j] if j < len(rates) else None
            if old is None and new is None:
                continue
            if old != new:
                return i
    return 0


def forecast_players(post_break, target, base=None):
    """Forecast players_tracked at the target by a log-log line fit.

    Fits log(N) against log(hours since the first post-break read + 1).
    Falls back to the last observed count with too few points or a
    degenerate fit. Clamped to [last, 4 * last]. Returns None when no
    usable player count exists at all.
    """
    usable = []
    for read in post_break:
        players = read_players(read)
        moment = parse_time(read.get("at"))
        if players is not None and moment is not None:
            usable.append((moment, players))
    if not usable:
        return None
    last = usable[-1][1]
    if base is None:
        base = usable[0][0]
    if len(usable) >= 3:
        xs = [log_hours(base, moment) for moment, _ in usable]
        ys = [math.log(players) for _, players in usable]
        if len(set(xs)) > 1:
            fit = least_squares(xs, ys)
            if fit is not None:
                slope, intercept = fit
                value = math.exp(slope * log_hours(base, target) + intercept)
                return min(4.0 * last, max(last, value))
    return last


def earners_of(read, index):
    """Earners E = rate/100 * players_tracked for one read, or None."""
    rate = read_rate(read, index)
    players = read_players(read)
    if rate is None or players is None:
        return None
    return rate / 100.0 * players


def forecast_earners(post_break, index, target, base=None):
    """Forecast earners for one trophy at the target.

    Fits log(E + 0.5) against log-hours (the 0.5 smooths zero earners),
    with log-hours measured from the first post-break read, the same
    variable the players fit uses. With too few usable points or a
    degenerate fit, uses the last observed earners value instead.
    Floored at zero.
    """
    points = []
    last_earners = None
    for read in post_break:
        moment = parse_time(read.get("at"))
        earners = earners_of(read, index)
        if earners is not None:
            last_earners = earners
            if moment is not None and earners > 0:
                points.append((moment, earners))
    if last_earners is None:
        return 0.0
    if base is None and points:
        base = points[0][0]
    if len(points) >= 3 and base is not None:
        xs = [log_hours(base, moment) for moment, _ in points]
        ys = [math.log(earners + 0.5) for _, earners in points]
        if len(set(xs)) > 1:
            fit = least_squares(xs, ys)
            if fit is not None:
                slope, intercept = fit
                value = math.exp(slope * log_hours(base, target) + intercept) - 0.5
                return max(0.0, value)
    return max(0.0, last_earners)


def last_rate_of(history, index):
    """Most recent non-null rate for one trophy, or None."""
    for read in reversed(history):
        rate = read_rate(read, index)
        if rate is not None:
            return rate
    return None


def spread_for(horizon_days, median, last_rate):
    """Quantile spread on the logit scale, floored at 0.15, capped at 5."""
    base = 0.5 * math.sqrt(horizon_days)
    if last_rate is None:
        return min(5.0, max(0.15, base))
    drift = abs(logit(median) - logit(last_rate))
    return min(5.0, max(0.15, base + 0.35 * drift))


def forecast_trophy(case, history, post_break, index, target, horizon_days,
                    players_target, base=None):
    """Five quantiles for one trophy."""
    try:
        last_rate = last_rate_of(history, index)
        if last_rate is None:
            median = 0.01
            spread = spread_for(horizon_days, median, None)
        else:
            last_clamped = clamp_pct(last_rate)
            if last_rate >= 99.5:
                # Pinned 100% artifact: one trophy sits at exactly 100%.
                median = 99.99
            elif players_target is None or players_target <= 0:
                median = last_clamped
            else:
                earners_target = forecast_earners(post_break, index, target, base)
                median = clamp_pct(100.0 * earners_target / players_target)
            spread = spread_for(horizon_days, median, last_clamped)
    except Exception:
        # Per-trophy fallback: persistence with a plain horizon spread.
        last_rate = last_rate_of(history, index)
        median = clamp_pct(last_rate) if last_rate is not None else 0.01
        spread = min(5.0, max(0.15, 0.5 * math.sqrt(horizon_days)))
    centre = logit(median)
    quantiles = []
    for z in Z_VALUES:
        quantiles.append(round4(inv_logit(centre + z * spread)))
    # Enforce non-decreasing order, then clamp into [0, 100].
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    return [min(100.0, max(0.0, q)) for q in quantiles]


def forecast_case(case):
    """Forecast every trophy in one case; returns the output object."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    if not isinstance(history, list):
        history = []
    target = parse_time(case.get("target"))
    if target is None:
        target = parse_time(case.get("at"))
    horizon_minutes = case.get("horizon_minutes")
    if not isinstance(horizon_minutes, (int, float)) or isinstance(horizon_minutes, bool):
        start = parse_time(case.get("at"))
        if start is not None and target is not None:
            horizon_minutes = (target - start).total_seconds() / 60.0
        else:
            horizon_minutes = 0.0
    horizon_days = max(0.0, float(horizon_minutes)) / 1440.0

    if history:
        break_at = find_break(history)
        post_break = history[break_at:]
    else:
        post_break = []
    # One shared log-hours origin for both fits: the first post-break read.
    base = parse_time(post_break[0].get("at")) if post_break else None
    players_target = forecast_players(post_break, target, base) if post_break else None

    rows = []
    for index, trophy in enumerate(trophies):
        trophy_id = trophy.get("id") if isinstance(trophy, dict) else None
        quantiles = forecast_trophy(
            case, history, post_break, index, target, horizon_days,
            players_target, base,
        )
        rows.append({"id": trophy_id, "q": quantiles})
    return {"trophies": rows}


def process_file(input_path, output_path):
    """Read one .in case, write its .out forecast. Returns False to skip."""
    with open(input_path, "r", encoding="utf-8") as handle:
        case = json.load(handle)
    if not isinstance(case, dict):
        return False
    if case.get("levels") != LEVELS:
        # Levels must be exactly the five required values; fail the case.
        return False
    result = forecast_case(case)
    text = json.dumps(result, ensure_ascii=False) + "\n"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return True


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR") or "/input")
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".in"):
            continue
        input_path = input_dir / name
        if not input_path.is_file():
            continue
        output_path = output_dir / (name[:-3] + ".out")
        try:
            process_file(input_path, output_path)
        except Exception:
            # Never crash the whole run on one bad case; skip it.
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
