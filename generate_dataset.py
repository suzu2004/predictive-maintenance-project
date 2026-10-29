"""
generate_dataset.py
--------------------
Generates a synthetic industrial IoT sensor dataset matching the exact
schema provided in the project scope document:

Machine_ID | Timestamp | Temperature | Pressure | Vibration | RPM |
Current | Voltage | Operating_Hours | Maintenance_Count | Machine_Age | Failure

Logic:
- 40 machines (M101-M140), each monitored hourly over ~90 days
- Each machine has a baseline "health" that slowly degrades over time
- As a machine approaches failure, Temperature and Vibration drift upward,
  Pressure/RPM become unstable, matching real predictive-maintenance behavior
- Failure = 1 is assigned near the end of a "degradation episode"; most
  readings are Failure = 0 (realistic imbalance, ~6-8% failure rate)
- A few missing values, a few duplicate rows, and a few outlier spikes are
  injected on purpose -- so the Data Cleaning phase has real work to do
  (this matches what the project brief expects you to demonstrate)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(7)

N_MACHINES = 40
HOURS = 24 * 90  # 90 days of hourly readings
start_time = datetime(2026, 5, 1, 0, 0)

rows = []

for m in range(1, N_MACHINES + 1):
    machine_id = f"M{100+m}"
    machine_age = np.random.randint(1, 12)          # years
    maintenance_count = np.random.randint(0, 15)     # historical maintenance events

    base_temp = np.random.uniform(55, 70)
    base_pressure = np.random.uniform(120, 160)
    base_vibration = np.random.uniform(0.2, 0.5)
    base_rpm = np.random.uniform(2600, 3100)
    base_current = np.random.uniform(8, 15)
    base_voltage = np.random.uniform(400, 430)

    # a machine can have MULTIPLE failure episodes over 90 days (fails, gets
    # repaired, degrades again) -- more realistic than a single lifetime
    n_episodes = np.random.choice([0, 1, 2, 3], p=[0.10, 0.30, 0.35, 0.25])
    episode_centers = sorted(np.random.choice(range(150, HOURS - 20), size=n_episodes, replace=False)) if n_episodes else []
    degrade_windows = [np.random.randint(40, 90) for _ in episode_centers]

    operating_hours = np.random.randint(500, 3000)  # cumulative usage at start of window

    for h in range(HOURS):
        ts = start_time + timedelta(hours=h)
        operating_hours += 1

        severity = 0
        is_failure_flag = False
        for center, window in zip(episode_centers, degrade_windows):
            if center - window <= h <= center:
                s = (h - (center - window)) / window
                severity = max(severity, max(0, s) ** 1.5)
            if center - 12 <= h <= center:   # flag failure state in the last 12 hrs of an episode
                is_failure_flag = True

        temperature = base_temp + severity * np.random.uniform(20, 35) + np.random.normal(0, 1.2)
        pressure = base_pressure - severity * np.random.uniform(15, 30) + np.random.normal(0, 2.5)
        vibration = base_vibration + severity * np.random.uniform(1.0, 2.2) + np.random.normal(0, 0.05)
        rpm = base_rpm - severity * np.random.uniform(200, 500) + np.random.normal(0, 25)
        current = base_current + severity * np.random.uniform(3, 8) + np.random.normal(0, 0.3)
        voltage = base_voltage - severity * np.random.uniform(10, 25) + np.random.normal(0, 2)

        failure = 1 if is_failure_flag else 0

        rows.append([
            machine_id, ts, round(temperature, 2), round(pressure, 2), round(vibration, 3),
            round(rpm, 1), round(current, 2), round(voltage, 1),
            operating_hours, maintenance_count, machine_age, failure
        ])

cols = ["Machine_ID", "Timestamp", "Temperature", "Pressure", "Vibration", "RPM",
        "Current", "Voltage", "Operating_Hours", "Maintenance_Count", "Machine_Age", "Failure"]
df = pd.DataFrame(rows, columns=cols)

# ---- Inject realistic data-quality issues on purpose (for the Cleaning phase) ----
rng = np.random.RandomState(1)

# 1. Missing values
for col in ["Temperature", "Pressure", "Vibration", "Voltage"]:
    idx = rng.choice(df.index, size=int(len(df)*0.004), replace=False)
    df.loc[idx, col] = np.nan

# 2. Duplicate rows
dup_idx = rng.choice(df.index, size=30, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

# 3. Outlier spikes (sensor glitches)
idx = rng.choice(df.index, size=25, replace=False)
df.loc[idx, "Temperature"] = df.loc[idx, "Temperature"] * rng.uniform(2.5, 4, size=25)

idx2 = rng.choice(df.index, size=15, replace=False)
df.loc[idx2, "Vibration"] = -abs(df.loc[idx2, "Vibration"])  # invalid negative vibration

df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle rows

df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw_sensor_data.csv"), index=False)
print("Shape:", df.shape)
print("Failure rate:", df.Failure.mean().round(4))
print("Machines:", df.Machine_ID.nunique())
print(df.head())
