import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import ast


def create_samples(root, portion=1.0, num_beam=64):
    f = pd.read_csv(root)
    bbox_all = []
    beam_power_all = []
    for idx, row in f.iterrows():
        bboxes = np.stack(
            [np.asarray(ast.literal_eval(r)) for r in row.loc["x_1":"x_8"]], axis=0
        )
        bbox_all.append(bboxes)
        beam_powers = np.stack(
            [np.asarray(ast.literal_eval(r)) for r in row.loc["y_1":"y_13"]], axis=0
        )
        beam_power_all.append(beam_powers)

    bbox_all = np.stack(bbox_all, axis=0)
    beam_power_all = np.stack(beam_power_all, axis=0)
    best_beam = np.argmax(beam_power_all, axis=-1)

    print("list is ready")
    num_data = len(beam_power_all)
    num_data = int(num_data * portion)
    return bbox_all[:num_data], best_beam[:num_data, -5:]


class DataFeed(Dataset):
    def __init__(self, root_dir, portion=1.0, num_beam=64):

        self.root = root_dir
        self.samples, self.pred_val = create_samples(
            self.root, portion=portion, num_beam=num_beam
        )
        self.seq_len = 8
        self.num_beam = num_beam

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        samples = self.samples[idx]  # Read one data sample
        pred_val = self.pred_val[idx]

        samples = samples[-self.seq_len :]  # Read a sequence of tuples from a sample

        out_beam = torch.zeros((5,))
        bbox = torch.zeros((self.seq_len, 4))

        if not samples.size:
            samples = np.zeros(4)
        bbox = torch.tensor(samples, requires_grad=False)

        out_beam = torch.tensor(pred_val, requires_grad=False)

        return bbox.float(), out_beam.long()
