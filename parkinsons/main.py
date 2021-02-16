from utils import plot_empirical_rule, train_ensemble

import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import forestci as fci
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os

# options
use_cols = 's+,j+'
ua_ensemble=False
folder_name=use_cols+'_1'
train=True

df = pd.read_csv('parkinsons_updrs.data')

y_m = df['motor_UPDRS']
y_t = df['total_UPDRS']

X_u = df.drop(labels=['subject#', 'motor_UPDRS', 'total_UPDRS', 'test_time'], axis=1)
X = X_u.apply(zscore)

if use_cols=='j,s':
    cols = [['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'PPE'], 
            ['Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA']]
elif use_cols=='s,j':
    cols = [['Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA'],
            ['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'PPE']]
elif use_cols=='j+,s':
    cols = [['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'PPE', 'age', 'sex'], 
            ['Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA']]
elif use_cols=='j,s+':
    cols = [['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'PPE'], 
            ['Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'age', 'sex']]
elif use_cols=='j+,s+':
    cols = [['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'PPE', 'age', 'sex'], 
            ['Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'age', 'sex']]
elif use_cols=='s+,j':
    cols = [['Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'age', 'sex'],
            ['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'PPE']]
elif use_cols=='s,j+':
    cols = [['Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA'],
            ['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'PPE', 'age', 'sex']]
elif use_cols=='s+,j+':
    cols = [['Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'age', 'sex'],
            ['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'PPE', 'age', 'sex']]
else:
    print("Incorrect option")

if folder_name not in os.listdir():
	os.mkdir(folder_name)
print(folder_name)

if ua_ensemble:
	if train:
		ua_all_t_te, ua_all_y_pred_uw, ua_all_y_pred_w, ua_all_y_std0, ua_all_y_std1 = train_ensemble(X, y_t, cols, boosting=True, uncertainty=True)
		np.save(os.path.join(folder_name, 'ua_all_t_te'), ua_all_t_te)
		np.save(os.path.join(folder_name, 'ua_all_y_pred_uw'), ua_all_y_pred_uw)
		np.save(os.path.join(folder_name, 'ua_all_y_pred_w'), ua_all_y_pred_w)
		np.save(os.path.join(folder_name, 'ua_all_y_std0'), ua_all_y_std0)
		np.save(os.path.join(folder_name, 'ua_all_y_std1'), ua_all_y_std1)
	else:
		ua_all_t_te = np.load(os.path.join(folder_name, 'ua_all_t_te.npy'))
		ua_all_y_pred_uw = np.load(os.path.join(folder_name, 'ua_all_y_pred_uw.npy'))
		ua_all_y_pred_w = np.load(os.path.join(folder_name, 'ua_all_y_pred_w.npy'))
		ua_all_y_std0 = np.load(os.path.join(folder_name, 'ua_all_y_std0.npy'))
		ua_all_y_std1 = np.load(os.path.join(folder_name, 'ua_all_y_std1.npy'))

	plot_empirical_rule(ua_all_y_pred_uw, np.asarray([ua_all_y_std0/2, ua_all_y_std1/2]), ua_all_t_te, model_name=os.path.join(folder_name, 'ua ensemble'))
	plot_empirical_rule(ua_all_y_pred_w, np.asarray([ua_all_y_std0/2, ua_all_y_std1/2]), ua_all_t_te, model_name=os.path.join(folder_name, 'ua ensemble weighted'))
else:
	if train:
		van_all_t_te, van_all_y_pred_uw, van_all_y_pred_w, van_all_y_std0, van_all_y_std1 = train_ensemble(X, y_t, cols, boosting=True, uncertainty=False)
		np.save(os.path.join(folder_name, 'van_all_t_te'), van_all_t_te)
		np.save(os.path.join(folder_name, 'van_all_y_pred_uw'), van_all_y_pred_uw)
		np.save(os.path.join(folder_name, 'van_all_y_pred_w'), van_all_y_pred_w)
		np.save(os.path.join(folder_name, 'van_all_y_std0'), van_all_y_std0)
		np.save(os.path.join(folder_name, 'van_all_y_std1'), van_all_y_std1)
	else:
		van_all_t_te = np.load(os.path.join(folder_name, 'van_all_t_te.npy'))
		van_all_y_pred_uw = np.load(os.path.join(folder_name, 'van_all_y_pred_uw.npy'))
		van_all_y_pred_w = np.load(os.path.join(folder_name, 'van_all_y_pred_w.npy'))
		van_all_y_std0 = np.load(os.path.join(folder_name, 'van_all_y_std0.npy'))
		van_all_y_std1 = np.load(os.path.join(folder_name, 'van_all_y_std1.npy'))
	plot_empirical_rule(van_all_y_pred_uw, np.asarray([van_all_y_std0/2, van_all_y_std1/2]), van_all_t_te, model_name=os.path.join(folder_name, 'vanilla ensemble'))


print(folder_name)
