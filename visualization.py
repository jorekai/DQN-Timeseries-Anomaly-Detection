import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    swat_normal = pd.read_pickle("./ts_data/ds_swat_attack.pkl", compression=None)
    test = []
    test.append(swat_normal)
    print(test[0].columns)
    test[0] = test[0].replace({"y": "A ttack"}, 1)
    test[0]["y"] = test[0]["y"].astype(float)
    plt.figure()
    test[0].plot(y="y", use_index=True)
    plt.show()
    for i in test[0].columns:
        plt.figure()
        plt.plot(test[0][i])
        plt.show()
