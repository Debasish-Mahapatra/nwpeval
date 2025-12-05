import numpy as np
from nwpeval import NWP_Stats

# Generate sample observed and modeled data
obs_data = np.random.rand(100, 100)
model_data = np.random.rand(100, 100)

# Create an instance of the NWPMetrics class
metrics = NWP_Stats(obs_data, model_data)

# Define the metrics to compute
metrics_to_compute = ['MAE', 'RMSE', 'ACC', 'FSS', 'ETS', 'POD', 'FAR', 'CSI', 'BSS', 'HSS',
                      'PSS', 'SEDS', 'FB', 'GSS', 'H-KD', 'ORSS', 'EDS', 'SEDI', 'RPSS']

# Define the thresholds for specific metrics
thresholds = {
    'FSS': 0.7,
    'FSS_neighborhood': 5,
    'ETS': 0.6,
    'POD': 0.5,
    'FAR': 0.3,
    'CSI': 0.4,
    'BSS': 0.6,
    'HSS': 0.5,
    'PSS': 0.4,
    'SEDS': 0.8,
    'FB': 0.6,
    'GSS': 0.5,
    'H-KD': 0.7,
    'ORSS': 0.6,
    'EDS': 0.8,
    'SEDI': 0.9,
    'RPSS': 0.7
}

# Compute the metrics
metric_values = metrics.compute_metrics(metrics_to_compute, thresholds)

# Print the computed metric values
for metric, value in metric_values.items():
    print(f"{metric}: {value:.4f}")