
import unittest
import numpy as np
import xarray as xr
from nwpeval import sedi

class TestSEDI(unittest.TestCase):
    def test_perfect_forecast(self):
        """Test that a perfect forecast yields SEDI = 1.0"""
        obs = xr.DataArray(np.array([0, 0, 1, 1, 1, 0]))
        model = xr.DataArray(np.array([0, 0, 1, 1, 1, 0]))
        threshold = 0.5
        
        result = sedi(obs, model, threshold)
        val = result.values.item()
        
        self.assertTrue(np.isclose(val, 1.0), f"Perfect forecast should be 1.0, got {val}")

    def test_good_forecast(self):
        """Test that a good forecast (H > F) yields positive SEDI"""
        # H=0.8, F=0.1
        obs_list = [1]*10 + [0]*90
        mod_list = [0]*100
        # Set TPs
        for i in range(8): mod_list[i] = 1
        # Set FPs (indices 10 to 18)
        for i in range(10, 19): mod_list[i] = 1
        
        obs = xr.DataArray(np.array(obs_list))
        model = xr.DataArray(np.array(mod_list))
        threshold = 0.5
        
        result = sedi(obs, model, threshold)
        val = result.values.item()
        
        self.assertTrue(val > 0, f"Good forecast (H > F) should have positive SEDI, got {val}")

    def test_worst_forecast(self):
        """Test that a completely wrong forecast yields negative SEDI"""
        obs = xr.DataArray(np.array([0, 0, 1, 1]))
        model = xr.DataArray(np.array([1, 1, 0, 0]))
        threshold = 0.5
        
        result = sedi(obs, model, threshold)
        val = result.values.item()
        
        # Should be close to -1 or at least negative
        self.assertTrue(val < 0, f"Worst forecast should be negative, got {val}")

if __name__ == '__main__':
    unittest.main()
