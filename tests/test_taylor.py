# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from nwpeval import taylor_diagram


# %%

# Create synthetic time series data (reference and two models)
np.random.seed(0)
t = np.arange(200)
reference_data = xr.DataArray(np.sin(t/10.0) + 0.05*np.random.randn(t.size), dims=['time'], coords={'time': t})
modelA_data = xr.DataArray(0.9*np.sin(t/10.0) + 0.1*np.random.randn(t.size), dims=['time'], coords={'time': t})
modelB_data = xr.DataArray(1.1*np.sin(t/10.0 + 0.2) + 0.15*np.random.randn(t.size), dims=['time'], coords={'time': t})

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')

# model_data can be a list of xarray.DataArray or numpy arrays
ax = taylor_diagram(ax,
                         model_names=['modelA','modelB'],
                         model_data=[modelA_data, modelB_data],
                         ref_data=reference_data,
                         rticks=np.arange(0.0, 1.3, 0.2))
ax.legend()
fig.savefig('./taylor_test.png', dpi=500, facecolor='w',
            bbox_inches='tight')
# plt.show()