"""
tests/test_api.py
------------------
Basic unit tests for the FastAPI backend.

Run with:
    pytest tests/ -v
"""

import sys
import os
import pandas as pd
import pytest

# Make the app/ directory importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from api import build_feature_row, FEATURE_COLS, SensorReading


def _make_reading(**overrides):
    """Return a valid SensorReading with sensible defaults, supporting overrides."""
    defaults = dict(
        Temperature=78.5,
        Pressure=135.2,
        Vibration=0.9,
        RPM=2700.0,
        Current=12.4,
        Voltage=410.0,
        Operating_Hours=2200.0,
        Maintenance_Count=5,
        Machine_Age=6,
    )
    defaults.update(overrides)
    return SensorReading(**defaults)


class TestBuildFeatureRow:
    """Tests for the build_feature_row helper."""

    def test_returns_dataframe(self):
        reading = _make_reading()
        result = build_feature_row(reading)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self):
        reading = _make_reading()
        result = build_feature_row(reading)
        assert list(result.columns) == FEATURE_COLS

    def test_single_row(self):
        reading = _make_reading()
        result = build_feature_row(reading)
        assert len(result) == 1

    def test_all_float_dtype(self):
        reading = _make_reading()
        result = build_feature_row(reading)
        assert all(result.dtypes == float)

    def test_ophours_band_2000_3000(self):
        reading = _make_reading(Operating_Hours=2500.0)
        result = build_feature_row(reading)
        assert result["OpHours_Band_2000-3000"].iloc[0] == 1.0
        assert result["OpHours_Band_1000-2000"].iloc[0] == 0.0

    def test_ophours_band_4000_plus(self):
        reading = _make_reading(Operating_Hours=5000.0)
        result = build_feature_row(reading)
        assert result["OpHours_Band_4000+"].iloc[0] == 1.0

    def test_ophours_band_0_to_1000(self):
        """Op hours below 1000 should produce all zero band flags."""
        reading = _make_reading(Operating_Hours=800.0)
        result = build_feature_row(reading)
        assert result["OpHours_Band_1000-2000"].iloc[0] == 0.0
        assert result["OpHours_Band_2000-3000"].iloc[0] == 0.0
        assert result["OpHours_Band_3000-4000"].iloc[0] == 0.0
        assert result["OpHours_Band_4000+"].iloc[0] == 0.0

    def test_age_group_mid(self):
        reading = _make_reading(Machine_Age=5)
        result = build_feature_row(reading)
        assert result["Age_Group_Mid(3-6y)"].iloc[0] == 1.0
        assert result["Age_Group_Old(6-9y)"].iloc[0] == 0.0

    def test_age_group_very_old(self):
        reading = _make_reading(Machine_Age=11)
        result = build_feature_row(reading)
        assert result["Age_Group_VeryOld(9y+)"].iloc[0] == 1.0

    def test_maintenance_frequency_zero_age(self):
        """Machine_Age=0 should not produce a division-by-zero error."""
        reading = _make_reading(Machine_Age=0, Maintenance_Count=3)
        result = build_feature_row(reading)
        # max(0, 1) = 1, so frequency = 3/1 = 3.0
        assert result["Maintenance_Frequency"].iloc[0] == pytest.approx(3.0)

    def test_maintenance_frequency_normal(self):
        reading = _make_reading(Machine_Age=6, Maintenance_Count=3)
        result = build_feature_row(reading)
        assert result["Maintenance_Frequency"].iloc[0] == pytest.approx(0.5)


class TestSensorReadingValidation:
    """Tests for Pydantic input validation on SensorReading."""

    def test_valid_reading_passes(self):
        reading = _make_reading()
        assert reading.Temperature == 78.5

    def test_temperature_below_zero_fails(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _make_reading(Temperature=-1.0)

    def test_temperature_above_max_fails(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _make_reading(Temperature=501.0)

    def test_machine_age_negative_fails(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _make_reading(Machine_Age=-5)

    def test_machine_age_above_max_fails(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _make_reading(Machine_Age=51)

    def test_vibration_negative_fails(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _make_reading(Vibration=-0.1)

    def test_rpm_above_max_fails(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _make_reading(RPM=99999.0)
