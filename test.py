from coreg.data_utils import load_data
import numpy as np

# data_dir = 'data/skillcraft'
data_dir = "data/wells"

X, y = load_data(data_dir)

print(X.shape)
print(y.shape)
print(any(np.isnan(y)))