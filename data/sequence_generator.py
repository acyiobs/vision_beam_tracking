# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 19:54:00 2021

@author: Umut Demirhan
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import copy


def blockage_prediction(csv_frame, data_list_y):
    data_label = np.empty((0, 1), dtype=int)
    for i in np.arange(len(data_list_y)):
        blockage = np.zeros(data_list_y[0])
        for j in range(len(data_list_y[i])):
            blockage[j] = csv_frame[csv_frame.index == data_list_y[i][j]]['blockage'].item()
        blockage = np.sum(blockage) > 0
        data_label = np.append(data_label, blockage)
    return data_label


def beam_prediction(csv_frame, data_list_y):
    data_label = np.empty((0, 1), dtype=int)
    for i in tqdm(np.arange(len(data_list_y))):
        power_levels = np.loadtxt(csv_frame[csv_frame.index == data_list_y[i][0]]['unit1_pwr_60ghz'].item())
        beam_idx = np.argmax(power_levels) + 1
        data_label = np.append(data_label, beam_idx)
    return data_label


def beam_tracking(csv_frame, data_list_y):
    data_label = np.empty((len(data_list_y), len(data_list_y[0])), dtype=int)
    for i in tqdm(np.arange(len(data_list_y))):
        beam_idx = np.zeros(len(data_list_y[0]))
        for j in range(len(data_list_y[i])):
            power_levels = np.loadtxt(csv_frame[csv_frame.index == data_list_y[i][j]]['unit1_pwr_60ghz'].item())
            beam_idx[j] = np.argmax(power_levels) + 1
        data_label[i] = beam_idx
    return data_label


class TimeSeriesGenerator:
    def __init__(self,
                 csv_file='scenario9.csv',
                 x_size=5,
                 y_size=1,
                 delay=0,
                 seed=5,
                 label_function=beam_prediction,
                 save_filename=None):

        # x_size: Size of input samples
        # y_size: Size of label samples to generate the labels
        # delay: The number of samples between x and y sequences. 

        # Example Series: x_size=3, y_size=1, delay=0 --> [1 2 3] [4]
        # Example Series: x_size=3, y_size=1, delay=1 --> [1 2 3] [5]
        # Example Series: x_size=3, y_size=2, delay=0 --> [1 2 3] [4, 5]

        # For Beam Prediction, y_size=1
        # For Blockage Prediction, y_size=# of samples in a blockage duration (e.g., 3 samples)

        if save_filename == None:
            self.save_filename = csv_file.split('.')[0] + '_series' + '.csv'
        else:
            self.save_filename = save_filename

        self.csv_frame = pd.read_csv(csv_file, index_col='index')
        self.num_sequences = self.csv_frame['seq_index'].max()

        self.seq_start = []
        self.seq_end = []
        self._extract_seq_start_end()

        self.x_size = x_size
        self.y_size = y_size
        self.delay = delay

        self.data_list_x = np.empty((0, x_size), dtype=int)
        self.data_list_y = np.empty((0, y_size), dtype=int)
        self.data_list_seq = np.empty((0, y_size), dtype=int)

        self._generate_indices()

        self.data_labels = label_function(self.csv_frame, self.data_list_y)

        # Shuffling indices
        self.num_datapoints = len(self.data_list_y)
        self.data_idx = np.arange(self.num_datapoints)
        rng = np.random.default_rng(seed)
        rng.shuffle(self.data_idx)

        # Shuffling sequences
        self.seq_idx = np.arange(self.num_sequences)
        rng = np.random.default_rng(seed)
        rng.shuffle(self.seq_idx)

    def _extract_seq_start_end(self):
        for i in np.arange(self.num_sequences) + 1:
            data_indices = self.csv_frame[self.csv_frame['seq_index'] == i].index
            self.seq_start.append(data_indices.min())
            self.seq_end.append(data_indices.max())

    def _generate_indices(self):
        for i in range(len(self.seq_start)):
            x_start_ind = self.seq_start[i]
            x_end_ind = x_start_ind + self.x_size

            y_start_ind = x_end_ind + self.delay
            y_end_ind = y_start_ind + self.y_size

            while y_end_ind <= self.seq_end[i] + 1:
                self.data_list_x = np.vstack((self.data_list_x, np.arange(x_start_ind, x_end_ind)))
                self.data_list_y = np.vstack((self.data_list_y, np.arange(y_start_ind, y_end_ind)))
                self.data_list_seq = np.append(self.data_list_seq, i)
                x_start_ind += 1
                x_end_ind += 1
                y_start_ind += 1
                y_end_ind += 1

    def take(self, num_of_data):
        new_dataset = copy.copy(self)
        new_dataset.data_idx = new_dataset.data_idx[:num_of_data]
        new_dataset.num_datapoints = len(new_dataset.data_idx)
        return new_dataset

    def skip(self, num_of_data):
        new_dataset = copy.copy(self)
        new_dataset.data_idx = new_dataset.data_idx[num_of_data:]
        new_dataset.num_datapoints = len(new_dataset.data_idx)
        return new_dataset

    def take_by_idx(self, idx):
        new_dataset = copy.copy(self)
        new_dataset.data_idx = new_dataset.data_idx[idx]
        new_dataset.num_datapoints = len(new_dataset.data_idx)
        return new_dataset

    def __len__(self):
        return self.num_datapoints

    def save_split_files(self, split=(0.7, 0.2, 0.1), data_path_csv_column=None, label_path_csv_column=None,
                         split_names=('train', 'val', 'test'), label_name='beam_index', sequence_split=False,
                         save_y_ind=False):
        if sequence_split:
            num_sequences = self.num_sequences
            num_train = int(num_sequences * split[0])
            num_val = int(num_sequences * split[1])
            idx_train = np.where(np.in1d(self.data_list_seq[self.data_idx], self.seq_idx[:num_train]))
            idx_val = np.where(np.in1d(self.data_list_seq[self.data_idx], self.seq_idx[num_train:num_train + num_val]))
            idx_test = np.where(np.in1d(self.data_list_seq[self.data_idx], self.seq_idx[num_train + num_val:]))


        else:
            # Train, Validation, Test
            num_datapoints = len(self)
            num_train = int(num_datapoints * split[0])
            num_val = int(num_datapoints * split[1])
            idx_train = np.arange(0, num_train)
            idx_val = np.arange(num_train, num_train + num_val)
            idx_test = np.arange(num_train + num_val, num_datapoints)

        idx_list = [idx_train, idx_val, idx_test]
        for n, name in enumerate(split_names):
            self.take_by_idx(idx_list[n]).save_file(file_tag=name, data_path_csv_column=data_path_csv_column,
                                                    label_path_csv_column=label_path_csv_column, label_name=label_name,
                                                    shuffled=True, save_y_ind=save_y_ind)
        '''
        self.take_by_idx(idx_train).save_file(file_tag=split_names[0], data_path_csv_column=data_path_csv_column, label_path_csv_column=label_path_csv_column, label_name=label_name, shuffled=True, save_y_ind=False)
        self.take_by_idx(idx_val).save_file(file_tag=split_names[1], data_path_csv_column=data_path_csv_column, label_path_csv_column=label_path_csv_column, label_name=label_name, shuffled=True, save_y_ind=False)
        self.take_by_idx(idx_test).save_file(file_tag=split_names[2], data_path_csv_column=data_path_csv_column, label_path_csv_column=label_path_csv_column, label_name=label_name, shuffled=True,save_y_ind=False)
        '''

    # data_path_csv_column: If the location of the sequences are required 
    # in the output csv file, input the name of the csv column 
    # (e.g., 'unit1_radar').
    def save_file(self, file_tag='', data_path_csv_column=None, label_path_csv_column=None, label_name='label',
                  shuffled=False, save_y_ind=False):
        if data_path_csv_column is None:
            df_x = pd.DataFrame(self.data_list_x, columns=['x_%i' % (i + 1) for i in range(self.x_size)])
        else:
            df_x = pd.DataFrame(self.csv_frame[data_path_csv_column].to_numpy(str)[self.data_list_x - 1],
                                columns=['x_%i' % (i + 1) for i in range(self.x_size)])
        if save_y_ind:
            if label_path_csv_column is None:
                df_y = pd.DataFrame(self.data_list_y, columns=['y_%i' % (i + 1) for i in range(self.y_size)])
            else:
                df_y = pd.DataFrame(self.csv_frame[label_path_csv_column].to_numpy(str)[self.data_list_y - 1],
                                    columns=['y_%i' % (i + 1) for i in range(self.y_size)])
        else:
            df_y = pd.DataFrame()
        df_label = pd.DataFrame(self.data_labels,
                                columns=['%s_%i' % (label_name, i + 1) for i in range(self.data_labels.shape[1])])
        df = pd.concat([df_x, df_y, df_label], axis=1)
        df.index.name = 'index'
        df.index += 1

        if shuffled:
            df = df.iloc[self.data_idx]
            df.index.name = 'data_index'
            df = df.reset_index()
            df.index.name = 'index'
            df.index += 1

        filename = self.save_filename.split('.')[0] + '_' + file_tag + '.csv'
        df.to_csv(filename)
        print('%i data points are saved to %s' % (len(df), filename))


# %% Generate Time Series & Save Series CSV Files

csv_file = 'scenario8/DEV[95%]/scenario8.csv'

# May define your own label extraction function if needed
# blockage_prediction and beam_prediction are currently available
label_function = beam_tracking

# Data sequence size
x_size = 8
# Label sequence size, beam_prediction --> 1, blockage_prediction --> 3
y_size = x_size + 5
# Delay
delay = -x_size

rng_seed = 5  # Reproducibility

# The file name will be included in the series
data_path_csv_column = 'unit1_rgb'
label_path_csv_column = 'unit1_pwr_60ghz'
save_y_ind = True
# Name of the labels
label_name = 'beam_index'

# Sequence or data split of the files
# If False, the data is fully shuffled
# If True, Train-Validation-Test sets are separated by the sequence
# i.e., any of the sets will not have a shared sequence
sequence_split = True

x = TimeSeriesGenerator(csv_file=csv_file,
                        x_size=x_size,
                        y_size=y_size,
                        seed=rng_seed,
                        delay=delay,
                        label_function=label_function,
                        save_filename='scenario8_series.csv')

x.save_file(file_tag='full', data_path_csv_column=data_path_csv_column, label_path_csv_column=label_path_csv_column,
            label_name=label_name, shuffled=False, save_y_ind=save_y_ind)
x.save_split_files(split=(0.8, 0.0, 0.2), data_path_csv_column=data_path_csv_column,
                   label_path_csv_column=label_path_csv_column, label_name=label_name, sequence_split=sequence_split,
                   save_y_ind=save_y_ind)
