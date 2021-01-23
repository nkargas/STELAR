from covid19dh import covid19
import numpy as np
import itertools
from model import STELAR
from utils import rmse, mae

np.random.seed(0)

start_date = "2020-04-01"
end_date = "2020-07-10"

df, _ = covid19("USA", verbose=False, raw=False, level=2, start=start_date, end=end_date)
df = df[['administrative_area_level_2', 'date', 'confirmed', 'deaths', 'tests', 'hosp']]

states = df['administrative_area_level_2'].unique()

# Create the tensor
mytensor = df.iloc[:, 2:].values
mytensor = mytensor.reshape(df['administrative_area_level_2'].nunique(), df['date'].nunique(), -1)
mytensor = np.transpose(mytensor, (0, 2, 1))

for s in range(3):
    mytensor[:, s, 1:] = np.diff(mytensor[:, s, :]).clip(0)
mytensor = mytensor[:, :, 1:]

# Remove states for which we don't have data
non_zero_signals = np.where(np.all(np.sum(mytensor, axis=2) != 0, axis=1))
states = states[non_zero_signals]
signals = np.array(['New infections', 'Deaths', 'Tests', 'Hospitalization'])
mytensor = mytensor[non_zero_signals]

# Normalize data
max_values = np.max(mytensor, axis=2)
mytensor = mytensor / max_values[:, :, np.newaxis]

num_days_train, num_days_valid, num_days_test = 87, 3, 10
num_days = num_days_train + num_days_valid + num_days_test

# --------------------- Compute decomposition --------------------------------------------
mytensor_training = mytensor[:, :, :num_days_train]
mytensor_valid = mytensor[:, :, num_days_train:num_days_train + num_days_valid]
mytensor_test = mytensor[:, :, num_days_train + num_days_valid: num_days]

stelar_model_predictions = None
best_model = None
best_score = float('Inf')
rank = [20]
mu = [1e-3]
nu = [1e-2]

for param in itertools.product(rank, nu, mu):
    stelar_model = STELAR(rank=param[0], nu=param[1], mu=param[2], max_iter=50, inner_max_itr=50)
    val_err = stelar_model.fit(mytensor_training, mytensor_valid)
    if val_err < best_score:
        best_score = val_err
        best_model = stelar_model

stelar_model_predictions = best_model.predict(num_days)
rmse_stelar = rmse(stelar_model_predictions[:, 0, num_days - num_days_test:], mytensor_test[:, 0, :])
mae_stelar = mae(stelar_model_predictions[:, 0, num_days - num_days_test:], mytensor_test[:, 0, :])
print(f'RMSE STELAR : {rmse_stelar}, MAE STELAR : {mae_stelar}')
