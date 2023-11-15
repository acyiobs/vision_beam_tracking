import pandas as pd
import numpy as np
from tqdm import tqdm


def preprocess(in_path, out_path):
    csv_file_path = in_path
    csv_save_path = out_path
    df = pd.read_csv(csv_file_path)


    cols1 = ['x_'+str(s) for s in range(1,9)]
    cols2 = ['y_'+str(s) for s in range(1,14)]
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        for c in cols1:
            path = row.loc[c]
            path = path.replace('camera_data', 'camera_data_bbox').replace('jpg', 'txt')
            path = 'scenario8/DEV[95%]/' + path[1:]
            try:
                content = np.loadtxt(path)[1:]
            except:
                content = np.zeros(4)
            if not content.size:
                content = np.zeros(4)
            df.at[index, c] = np.array2string(content, separator=',')
        for c in cols2:
            path = row.loc[c]
            content = np.loadtxt(path)
            df.at[index, c] = np.array2string(content, separator=',')

    df.to_csv(csv_save_path, index=False)


#%% training dataset
csv_file_path = 'scenario8_series_train.csv'
csv_save_path = 'scenario8_series_bbox_train.csv'
preprocess(csv_file_path, csv_save_path)

#%% test dataset
csv_file_path = 'scenario8_series_test.csv'
csv_save_path = 'scenario8_series_bbox_test.csv'
preprocess(csv_file_path, csv_save_path)