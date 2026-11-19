"""gta6-trophy-rarity-forecast-01: calibrated trophy earn-rate forecaster.

Forecasts, for every PS5 trophy of Grand Theft Auto VI, its Exophase earn
rate at the case's target, as five quantiles [0.05, 0.25, 0.5, 0.75, 0.95].

Model (standard library only, deterministic, no network):
  * Work on the logit scale, against u = ln(1 + hours since 2026-11-19T00:00Z),
    using the case's last 24 hours of reads.
  * Players tracked N(t) is fitted log-linearly and extrapolated to the
    target; earners of a trophy can only grow, which gives a hard floor and
    ceiling for every trophy's target rate:
        floor   = 100 * E_last / N_target          (no new earner)
        ceiling = 100 * (E_last + N_target - N_last) / N_target   (all newcomers earn it)
  * Per trophy, a least-squares line of logit(rate) vs u over the last 24h
    is read at the target, then clamped into [floor, ceiling] and into a
    tier-based saturation cap (platinum 3%, gold 25%, silver 60%, bronze 97%).
  * With fewer than 3 distinct reads (e.g. launch night, when Exophase had
    not refreshed yet and every read is identical), the line is meaningless:
    fall back to persistence with a WIDE spread (sigma ~1.15 logit units),
    wide enough that a refresh dilution of the size Exophase showed on
    launch night (638 -> 3,868 players tracked, every rate but the 100% one
    lower) still lands inside the 90% band.
  * Spread otherwise combines a random-walk term (0.5 * sqrt(days)), the
    fit's residual, and a model term that grows with the horizon and with
    the size of the predicted move, so the 90% band is honest, not bold.

Run: python3 main.py, reading $MAKEFUN_INPUT_DIR (else /input) and writing
$MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Random-walk spread, logit units per sqrt(day) -- same scale the house uses.
SIGMA_RW = 0.5
# Extra spread when the history has too few distinct reads to fit a line:
# covers a launch-night-style refresh dilution (~1.8 logit units) at 90%.
SIGMA_FLAT = 1.15
# Model-uncertainty term, logit units per sqrt(day). Tuned on the harness:
# 0.15 over-covered and lost ~4% to the curve baseline on the growth test;
# 0.00 matched it but leaves nothing for real-world read noise (the
# synthetic generator is smoother than Exophase); 0.08 sits within ~1% of
# the curve baseline on growth while keeping a real noise allowance.
SIGMA_MODEL = 0.08
# Saturation caps on the target rate, by tier (percent).
TIER_CAP = {"platinum": 3.0, "gold": 25.0, "silver": 60.0, "bronze": 97.0}


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0.0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_at(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_epoch(dt: datetime) -> float:
    return (dt - EPOCH).total_seconds() / 3600.0


def ols(xs, ys):
    """Least-squares line y = a + b*x; returns (a, b, residual std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / max(n - 2, 1)
    return a, b, math.sqrt(var)


def last_known_rate(history, idx):
    """Most recent non-null rate for trophy index idx, else 0."""
    for h in reversed(history):
        r = h["rates"][idx]
        if r is not None:
            return r
    return 0.0


def forecast_case(case: dict, name: str = "?") -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    at = parse_at(case["at"])
    target = parse_at(case["target"])
    d_days = case["horizon_minutes"] / 1440.0
    u_target = math.log1p(max(hours_since_epoch(target), 0.0))

    # Reads in the last 24h before the checkpoint.
    cutoff = at.timestamp() - 24.0 * 3600.0
    recent = [h for h in history if parse_at(h["at"]).timestamp() >= cutoff]
    if len(recent) < 3:
        recent = history

    # --- players tracked: log-linear fit, extrapolated to the target ---
    xs_n, ys_n = [], []
    for h in recent:
        n = h.get("players_tracked")
        if n and n > 0:
            xs_n.append(hours_since_epoch(parse_at(h["at"])))
            ys_n.append(math.log(n))
    n_last = history[-1].get("players_tracked") or 0
    if len(xs_n) >= 2 and len(set(xs_n)) >= 2:
        a_n, b_n, _ = ols(xs_n, ys_n)
        n_target = math.exp(a_n + b_n * hours_since_epoch(target))
    else:
        n_target = float(n_last)
    # Sanity: tracked players never shrink in the forecast, and never more
    # than 8x the last count (launch-week influx has a limit).
    n_target = min(max(n_target, float(n_last)), 8.0 * max(float(n_last), 1.0))
    if n_target <= 0:
        n_target = float(n_last) if n_last > 0 else 1.0

    distinct_times = len({h["at"] for h in recent})
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_rate = last_known_rate(history, i)
        if last_rate >= 100.0:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue

        # Physical bounds from earners only growing.
        e_last = last_rate / 100.0 * float(n_last)
        floor_rate = 100.0 * e_last / n_target
        ceil_rate = 100.0 * (e_last + max(n_target - float(n_last), 0.0)) / n_target
        tier_cap = TIER_CAP.get(trophy.get("tier", "bronze"), 97.0)
        lo = max(floor_rate, 0.01)
        hi = min(ceil_rate, tier_cap, 99.99)

        # Per-trophy line of logit(rate) vs u over the last 24h.
        xs, ys = [], []
        for h in recent:
            r = h["rates"][i]
            if r is None:
                continue
            xs.append(math.log1p(max(hours_since_epoch(parse_at(h["at"])), 0.0)))
            ys.append(logit(r))
        distinct_u = len(set(xs))
        if len(xs) >= 3 and distinct_u >= 2 and len(set(ys)) >= 2:
            a, b, resid = ols(xs, ys)
            mu_rate = inv_logit(a + b * u_target)
            mu_rate = min(max(mu_rate, lo), hi)
            mu = logit(mu_rate)
            spread = math.sqrt(
                (SIGMA_RW * math.sqrt(d_days)) ** 2
                + resid ** 2
                + (SIGMA_MODEL * math.sqrt(d_days)
                   + 0.05 * abs(mu - logit(last_rate))) ** 2
            )
        else:
            # Flat history: persistence, wide spread (refresh insurance).
            mu = logit(last_rate)
            spread = math.sqrt(SIGMA_FLAT ** 2 + (SIGMA_RW * math.sqrt(d_days)) ** 2)

        qs = []
        for z in Z:
            q = inv_logit(mu + z * spread)
            q = min(max(q, 0.01), 99.99)
            qs.append(round4(q))
        # Keep quantiles non-decreasing after rounding.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
