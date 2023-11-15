import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary
from tqdm import tqdm
import sys
import datetime
from data_feed import DataFeed
from model import GruModelSimple
from scipy.io import savemat


def train_model(model_path, if_writer=False, num_beam=64):
    num_classes = num_beam + 1

    val_batch_size = 64

    val_dir = "data/scenario8_series_bbox_test.csv"
    
    val_loader = DataLoader(
        DataFeed(val_dir, num_beam=num_beam), batch_size=val_batch_size, shuffle=False
    )

    # check gpu acceleration availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    

    # Instantiate the model
    net = GruModelSimple(num_classes)
    net.load_state_dict((torch.load(model_path)))

    # print model summary
    if if_writer:
        h = net.initHidden(1)
        print(summary(net, torch.zeros((8, 1, 4)), h))
    # send model to GPU

    net.to(device)
    net.eval()
    # test
    predictions = []
    raw_predictions = []
    net.eval()
    with torch.no_grad():
        total = np.zeros((5,))
        top1_correct = np.zeros((5,))
        top2_correct = np.zeros((5,))
        top3_correct = np.zeros((5,))
        top5_correct = np.zeros((5,))
        val_loss = 0
        for (bbox, label) in val_loader:
            bbox = torch.swapaxes(bbox, 0, 1)
            bbox = torch.cat(
                [bbox, torch.zeros(torch.Size((4,)) + bbox.shape[1:]) - 1], dim=0
            )
            label = torch.swapaxes(label, 0, 1)
            bbox = bbox.to(device)
            label = label.to(device)

            bbox = bbox.to(device)
            label = label.to(device)

            h = net.initHidden(bbox.shape[1]).to(device)
            outputs, _ = net(bbox, h)
            outputs = outputs[-5:, ...]
            label = label[-5:, ...]
            val_loss += nn.CrossEntropyLoss(reduction="sum")(
                outputs.view(-1, num_classes), label.flatten()
            ).item()
            total += torch.sum(label != -100, dim=-1).cpu().numpy()
            prediction = torch.argmax(outputs, dim=-1)
            top1_correct += torch.sum(prediction == label, dim=-1).cpu().numpy()

            _, idx = torch.topk(outputs, 5, dim=-1)
            idx = idx.cpu().numpy()
            label = label.cpu().numpy()
            for i in range(label.shape[0]):
                for j in range(label.shape[1]):
                    top2_correct[i] += np.isin(label[i, j], idx[i, j, :2]).sum()
                    top3_correct[i] += np.isin(label[i, j], idx[i, j, :3]).sum()
                    top5_correct[i] += np.isin(label[i, j], idx[i, j, :5]).sum()

            predictions.append(prediction.cpu().numpy())
            raw_predictions.append(outputs.cpu().numpy())

        val_loss /= total.sum()
        val_top1_acc = top1_correct / total
        val_top2_acc = top2_correct / total
        val_top3_acc = top3_correct / total
        val_top5_acc = top5_correct / total

        predictions = np.concatenate(predictions, 1)
        raw_predictions = np.concatenate(raw_predictions, 1)

        val_acc = {
            "top1": val_top1_acc,
            "top2": val_top2_acc,
            "top3": val_top3_acc,
            "top5": val_top5_acc,
        }
        return val_loss, val_acc, predictions, raw_predictions


if __name__ == "__main__":
    torch.manual_seed(1115)
    num_epoch = 100
    val_loss, val_acc, predictions, raw_predictions = train_model(
        model_path = 'checkpoint/22_30_43_23_11_14_GruModelSimple.pth',
       if_writer=True, num_beam=64
    )
    print(val_acc)
