"""Generate SIMULATED racing telemetry (not from a real simulator)."""
from pathlib import Path
import numpy as np
import pandas as pd

TRACK_LENGTH_M = 5000
STEP_M = 10
TOP_SPEED = 280.0
# (corner position in metres, apex speed in km/h)
CORNERS = [(600, 90), (1300, 140), (1900, 70), (2700, 110), (3400, 60), (4200, 130)]
OUT_PATH = Path("data/sample/telemetry.csv")


def _lap(lap_number: int, rng: np.random.Generator) -> pd.DataFrame:
    dist = np.arange(0, TRACK_LENGTH_M, STEP_M, dtype=float)
    pts_d, pts_v = [0.0], [TOP_SPEED - 40]
    for pos, apex in CORNERS:
        apex_v = apex * rng.normal(1.0, 0.04)        # corner speed varies per lap
        brake_len = rng.uniform(150, 230)             # braking point varies per lap
        exit_len = rng.uniform(170, 230)
        pts_d += [pos - brake_len, pos, pos + exit_len]
        pts_v += [TOP_SPEED * rng.normal(1.0, 0.01), apex_v, TOP_SPEED * 0.9]
    pts_d.append(TRACK_LENGTH_M)
    pts_v.append(TOP_SPEED - 40)
    speed = np.interp(dist, pts_d, pts_v)
    speed = np.convolve(np.pad(speed, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    speed = speed + rng.normal(0, 0.8, speed.size)    # small sensor noise
    dt = STEP_M / (speed / 3.6)
    time = np.cumsum(dt) - dt[0]
    accel = np.gradient(speed, dist)                  # km/h per metre
    brake = np.where(-accel > 0.25, np.clip(-accel * 70, 0, 100), 0.0)
    throttle = np.clip(accel * 25 + 60, 0, 100)
    throttle = np.where(brake > 5, 0, throttle)
    throttle = np.where(speed > TOP_SPEED * 0.92, 100, throttle)
    return pd.DataFrame({
        "lap": lap_number,
        "distance_m": dist,
        "time_s": time.round(3),
        "speed_kmh": speed.round(2),
        "throttle_pct": throttle.round(1),
        "brake_pct": brake.round(1),
    })


def generate(n_laps: int = 6, seed: int = 42, path: Path = OUT_PATH) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.concat([_lap(i, rng) for i in range(1, n_laps + 1)], ignore_index=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    data = generate()
    print(f"Saved {len(data)} rows to {OUT_PATH.resolve()}")
    print(data.head())
