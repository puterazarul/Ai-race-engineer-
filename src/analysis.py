"""Telemetry analysis + rule-based insights (no AI/ML involved)."""
import numpy as np
import pandas as pd

REQUIRED = ["lap", "distance_m", "time_s", "speed_kmh", "throttle_pct", "brake_pct"]


def load_telemetry(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


def lap_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("lap")
    out = pd.DataFrame({
        "lap_time_s": g["time_s"].max() - g["time_s"].min(),
        "avg_speed_kmh": g["speed_kmh"].mean(),
        "max_speed_kmh": g["speed_kmh"].max(),
        "min_speed_kmh": g["speed_kmh"].min(),
    })
    return out.reset_index()


def summary(stats: pd.DataFrame) -> dict:
    fast = stats.loc[stats["lap_time_s"].idxmin()]
    slow = stats.loc[stats["lap_time_s"].idxmax()]
    return {
        "total_laps": len(stats),
        "fastest_lap": int(fast["lap"]), "fastest_time": fast["lap_time_s"],
        "slowest_lap": int(slow["lap"]), "slowest_time": slow["lap_time_s"],
        "avg_lap_time": stats["lap_time_s"].mean(),
        "max_speed": stats["max_speed_kmh"].max(),
        "avg_speed": stats["avg_speed_kmh"].mean(),
    }


def format_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:06.3f}"


def _sector_times(lap_df: pd.DataFrame, n: int) -> list:
    """Time spent in each of n equal-distance sectors."""
    length = lap_df["distance_m"].max()
    edges = np.linspace(0, length, n + 1)
    t = np.interp(edges, lap_df["distance_m"], lap_df["time_s"])
    return list(np.diff(t))


def compare_laps(df: pd.DataFrame, lap_a: int, lap_b: int, n_sectors: int = 5) -> list:
    """Return plain-English observations, each based on calculated values."""
    a, b = df[df["lap"] == lap_a], df[df["lap"] == lap_b]
    stats = lap_stats(df).set_index("lap")
    ta, tb = stats.loc[lap_a, "lap_time_s"], stats.loc[lap_b, "lap_time_s"]

    if abs(ta - tb) < 0.005:
        return [f"Lap {lap_a} and Lap {lap_b} had essentially the same lap time."]
    fast, slow = (lap_a, lap_b) if ta < tb else (lap_b, lap_a)
    ff, ss = (a, b) if fast == lap_a else (b, a)
    notes = [f"Lap {fast} was {abs(ta - tb):.3f} seconds faster than Lap {slow}."]

    fa, fb = ff["speed_kmh"].max(), ss["speed_kmh"].max()
    hi = fast if fa > fb else slow
    notes.append(f"Maximum speed was higher on Lap {hi} ({max(fa, fb):.1f} vs {min(fa, fb):.1f} km/h).")

    ma, mb = ff["speed_kmh"].min(), ss["speed_kmh"].min()
    if abs(ma - mb) >= 1:
        hi = fast if ma > mb else slow
        notes.append(f"Lap {hi} carried more speed through the slowest corner "
                     f"({max(ma, mb):.1f} vs {min(ma, mb):.1f} km/h).")

    ba, bb = (ff["brake_pct"] > 5).mean() * 100, (ss["brake_pct"] > 5).mean() * 100
    if abs(ba - bb) >= 0.5:
        more = fast if ba > bb else slow
        notes.append(f"Lap {more} spent more of the lap braking "
                     f"({max(ba, bb):.1f}% vs {min(ba, bb):.1f}% of the distance).")

    wa, wb = (ff["throttle_pct"] >= 99).mean() * 100, (ss["throttle_pct"] >= 99).mean() * 100
    if abs(wa - wb) >= 0.5:
        more = fast if wa > wb else slow
        notes.append(f"Lap {more} was at full throttle for more of the lap "
                     f"({max(wa, wb):.1f}% vs {min(wa, wb):.1f}%).")

    sf, sl = _sector_times(ff, n_sectors), _sector_times(ss, n_sectors)
    diffs = [y - x for x, y in zip(sf, sl)]
    i = int(np.argmax(diffs))
    length = ff["distance_m"].max()
    lo, hi_ = length * i / n_sectors, length * (i + 1) / n_sectors
    notes.append(f"Lap {slow} lost the most time ({diffs[i]:.3f} s) in sector {i + 1} "
                 f"({lo:.0f}-{hi_:.0f} m).")

    sec_f = ff[(ff.distance_m >= lo) & (ff.distance_m < hi_)]
    sec_s = ss[(ss.distance_m >= lo) & (ss.distance_m < hi_)]
    if sec_s["speed_kmh"].min() < sec_f["speed_kmh"].min() - 1:
        gap = sec_f["speed_kmh"].min() - sec_s["speed_kmh"].min()
        notes.append(f"Recommendation: in sector {i + 1}, Lap {slow}'s minimum speed was {gap:.1f} km/h "
                     "lower. Try braking a little less or smoother to carry more corner speed.")
    elif sec_s["brake_pct"].max() > sec_f["brake_pct"].max() + 5:
        notes.append(f"Recommendation: in sector {i + 1}, Lap {slow} braked harder "
                     f"(peak {sec_s['brake_pct'].max():.0f}% vs {sec_f['brake_pct'].max():.0f}%). "
                     "Try braking a little earlier with less pressure.")
    else:
        notes.append(f"Recommendation: study sector {i + 1} on Lap {fast} and copy its throttle and brake timing.")
    return notes
