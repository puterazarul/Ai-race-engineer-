from src.generate_sample_data import generate
from src.analysis import load_telemetry, lap_stats, summary, compare_laps


def test_pipeline(tmp_path):
    p = tmp_path / "t.csv"
    generate(n_laps=4, path=p)
    assert p.exists()
    df = load_telemetry(p)
    stats = lap_stats(df)
    assert len(stats) == 4
    assert (stats["lap_time_s"] > 60).all()
    s = summary(stats)
    assert s["fastest_time"] <= s["slowest_time"]
    notes = compare_laps(df, 1, 2)
    assert "faster than" in notes[0]
    assert len(notes) >= 4
