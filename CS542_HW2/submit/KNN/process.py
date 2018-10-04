import numpy as np
import pandas as pd
import sys

def process_data(data, filename):
	data = data.replace('?', np.NaN)

	feature_nan_col = [0, 3, 4, 5, 6]
	for col in feature_nan_col:
		na_val = data[col].mode()[0]
		data[col] = data[col].fillna(na_val)

	num_nan_col = [1, 13]
	for col in num_nan_col:
		data[col] = data[col].apply(float)

		#print(data[col][data[15] == '+'])
		pos_mean = data[col][data[15] == '+'].mean()
		data.loc[(data[col].isnull()) & (data[15] == '+'), col] = pos_mean
		# print(data.loc[(data[col].isnull()) & (data[15] == '+')])
		neg_mean = data[col][data[15] == '-'].mean()
		data.loc[(data[col].isnull()) & (data[15] == '-'), col] = neg_mean

	#print(data)

	num_col = [1, 2, 7, 10, 13, 14]
	for col in num_col:
		data[col] = (data[col]-data[col].mean())/data[col].std()
	# print(data)

	data.to_csv(filename, header=False, index=False)

train = sys.argv[1]
test = sys.argv[2]

train_data = pd.read_csv(train, header=None)
process_data(train_data, 'crx.training.processed')

test_data = pd.read_csv(test, header=None)
process_data(test_data, 'crx.testing.processed')

