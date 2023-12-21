import sys
import argparse
import glob
import os
import numpy as np
from my_tools.metrics import fast_confusion, fast_metrics
from tools.dataset_converters.ply import read_ply

class_name_titan = [
    "Impervious Ground",
    "Grass",
    "Building",
    "Tree",
    "Car",
    "Power Line",
    "Bare land",
]
class_name_sensaturban = [
    "Ground",
    "Vegetation",
    "Building",
    "Wall",
    "Bridge",
    "Parking",
    "Rail",
    "Traffic Road",
    "Street Furniture",
    "Car",
    "Footpath",
    "Bike",
    "Water",
]
class_name_dict = {"titan": class_name_titan, "sensaturban": class_name_sensaturban}


def out_markdown_table(title, data, log_path):
    if isinstance(data, np.ndarray):
        data = list(data)
    assert len(title) == len(data)
    markdown_title = "|"
    markdown_split = "|"
    markdown_data = "|"
    for i in range(len(title)):
        markdown_title += title[i] + "|"
        markdown_split += "-|"
        markdown_data += f"{data[i]:.4f}|"
    with open(log_path, "a+") as f:
        f.write(markdown_title + "\n")
        f.write(markdown_split + "\n")
        f.write(markdown_data + "\n\n")


def test_ply(preds, labels, dataset, log_path):
    if dataset == "titan":
        class_num = 7
        class_name = class_name_dict["titan"]
    elif dataset == "sensaturban":
        class_num = 13
        class_name = class_name_dict["sensaturban"]
    else:
        raise NotImplementedError

    confusion_metrics = fast_confusion(class_num, labels, preds)
    ACC, ACC_avg, Kappa, mIoU, IoU = fast_metrics(confusion_metrics)
    out_markdown_table(
        ["ACC", "ACC_avg", "Kappa", "IoU"], [ACC, ACC_avg, Kappa, mIoU], log_path
    )
    out_markdown_table(class_name, IoU, log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", dest="path", type=str, help="the path of model workdir"
    )
    args = parser.parse_args()

    log_path = os.path.join(args.path, "log.md")
    paths = glob.glob(os.path.join(args.path, "*.ply"))
    if "titan" in args.path:
        label_root = "data/Titan/origin_data"
        dataset = "titan"
    elif "sensaturban" in args.path:
        label_root = "data/SensatUrban/test"
        dataset = "sensaturban"
    else:
        raise NotImplementedError
    label = []
    pred = []
    for path in paths:
        filename = path.split("/")[-1]
        data = read_ply(os.path.join(label_root, filename))
        label.append(data["class"].astype(np.int32))
        data = read_ply(path)
        pred.append(data["pred"].astype(np.int32))
    label = np.concatenate(label)
    pred = np.concatenate(pred)
    test_ply(pred, label, dataset, log_path)
