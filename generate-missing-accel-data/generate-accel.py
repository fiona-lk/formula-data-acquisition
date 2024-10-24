# Fiona Keaney
# Underclassmen member
# CSU Fullerton Formula SAE Team

import pandas as pd
import numpy as np
import xgboost as xgb


# Functions across datasets
# Load data
def load_data(file):
    with open(file, 'r') as f:
        headers = [next(f) for _ in range(3)]
    tempfile = pd.read_csv(file, delimiter='\t', skiprows=3, header=None)
    return headers, tempfile.to_numpy()

# Truncate 3 datasets
def truncate_data(data1, data2, data3):
    rows1, rows2 , rows3 = data1.shape[0], data2.shape[0], data3.shape[0]
    min_rows = min(rows1, rows2, rows3)
    truncated_data1 = data1[:min_rows, :21]
    truncated_data2 = data2[:min_rows, :21]
    truncated_data3 = data3[:min_rows, :21]

    return truncated_data1, truncated_data2, truncated_data3

# Truncate 2 datasets
def truncate_2data(data1, data2):
    rows1, rows2 = data1.shape[0], data2.shape[0]
    min_rows = min(rows1, rows2)
    truncated_data1 = data1[:min_rows, :21]
    truncated_data2 = data2[:min_rows, :21]

    return truncated_data1, truncated_data2

# Preparing data
def prep_datasets(data1, data2, data3):
    tdata1, tdata2, tdata3 = truncate_data(data1, data2, data3)
    return tdata1, tdata2, tdata3

# Interpolating 8 and 10 psi sets
def interpolate(data_8psi, data_10psi):
    return ((data_10psi + data_8psi) / 2)

# Print data
def print_csv(file, filename="", headers=None):
    output_df = pd.DataFrame(file)
    output_file = filename + ".dat"
    
    if headers is not None:
        with open(output_file, 'w') as f:
            f.writelines(headers)
        output_df.to_csv(output_file, index=False, header=False, sep='\t', mode='a')
    else:
        output_df.to_csv(output_file, index=False, header=False, sep='\t')

# ML model
model = xgb.XGBRegressor()


# Actual dataset application
# loading needed datasets
headers_cor43100_8psi, cornering_43100_8psi = load_data('B2356run32.dat')
headers_cor43100_10psi, cornering_43100_10psi = load_data('B2356run31.dat')
headers_acc43100_8psi, acceleration_43100_8psi = load_data('B2356run73.dat')
headers_acc43100_10psi, acceleration_43100_10psi = load_data('B2356run72.dat')

headers_cor43164_8psi, cornering_43164_8psi = load_data('B2356run18.dat')
headers_cor43164_10psi, cornering_43164_10psi = load_data('B2356run17.dat')
headers_acc43164_8psi, acceleration_43164_8psi = load_data('B2356run52.dat')
headers_acc43164_10psi, acceleration_43164_10psi = load_data('B2356run51.dat')

headers_cor43075_8psi, cornering_43075_8psi = load_data('B2356run6.dat')
headers_cor43075_10psi, cornering_43075_10psi = load_data('B2356run4.dat')

# Based on 43100
# 8psi
# Train model on 43100
cor_100_8psi, acc_100_8psi, cor_075_8psi = prep_datasets(cornering_43100_8psi, acceleration_43100_8psi, cornering_43075_8psi)
model.fit(cor_100_8psi, acc_100_8psi)
# Apply model to 43075 8psi
predict_100_8psi = model.predict(cor_075_8psi)

# 10psi
# Train model on 43100
cor_100_10psi, acc_100_10psi, cor_075_10psi = prep_datasets(cornering_43100_10psi, acceleration_43100_10psi, cornering_43075_10psi)
model.fit(cor_100_10psi, acc_100_10psi)
# Apply model to 43075 10psi
predict_100_10psi = model.predict(cor_075_10psi)

# Interpolate data from 43100 model
tpredict_100_8psi, tpredict_100_10psi = truncate_2data(predict_100_8psi, predict_100_10psi)
acc_075_9psi = interpolate(tpredict_100_8psi, tpredict_100_10psi)
# Export prediction to dat file for Matlab use
print_csv(acc_075_9psi, "100based_acc_075_9psi", headers= headers_cor43075_8psi)



# Based on 43164
# 8psi
# Train model on 43164 8psi
cor_164_8psi, acc_164_8psi, cor_075_8psi = prep_datasets(cornering_43164_8psi, acceleration_43164_8psi, cornering_43075_8psi)
model.fit(cor_164_8psi, acc_164_8psi)
# Apply model to 43075 8psi
predict_164_8psi = model.predict(cor_075_8psi)

# 10psi
# Train model on 43164s 10psi
cor_164_10psi, acc_164_10psi, cor_075_10psi = prep_datasets(cornering_43164_10psi, acceleration_43164_10psi, cornering_43075_10psi)
model.fit(cor_164_10psi, acc_164_10psi)
# Apply model to 43075 10psi
predict_164_10psi = model.predict(cor_075_10psi)

# Interpolate data from 43075 model
tpredict_164_8psi, tpredict_164_10psi = truncate_2data(predict_164_8psi, predict_164_10psi)
acc2_075_9psi = interpolate(tpredict_164_8psi, tpredict_164_10psi)
# Export prediction to dat file for Matlab use
print_csv(acc2_075_9psi, "164based_acc_075_9psi", headers= headers_cor43075_8psi)


# Interpolating both predictions
acc_comb_9psi = interpolate(acc_075_9psi, acc2_075_9psi)
# Export prediction to dat file for Matlab use
print_csv(acc_comb_9psi, "comb_acc_075_9psi", headers= headers_cor43075_8psi)