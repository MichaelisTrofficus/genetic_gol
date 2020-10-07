import pandas as pd
import numpy as np

data_path = "./data/"
test = pd.read_csv(data_path + "test.csv")

# Delta 1
test_delta_1 = test.loc[test["delta"] == 1]
test_delta_1.to_csv(data_path + "delta1.csv", index=False)

# Delta 1
test_delta_2 = test.loc[test["delta"] == 2]
test_delta_2.to_csv(data_path + "delta2.csv", index=False)

# Delta 1
test_delta_3 = test.loc[test["delta"] == 3]
test_delta_3.to_csv(data_path + "delta3.csv", index=False)

# Delta 1
test_delta_4 = test.loc[test["delta"] == 4]
test_delta_4.to_csv(data_path + "delta4.csv", index=False)

# Delta 1
test_delta_5 = test.loc[test["delta"] == 5]
test_delta_5.to_csv(data_path + "delta5.csv", index=False)
