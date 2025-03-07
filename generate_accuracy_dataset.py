import os
from tqdm import tqdm
import json
import math 

import torch
import argparse
from torch import distributed as dist
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from ofa.imagenet_classification.elastic_nn.networks import OFAProxylessNASNets
from ofa.tutorial import calib_bn, validate, evaluate_ofa_subnet

from ofa.imagenet_classification.data_providers import AuroraDataset, AuroraDataProvider

num_classes = 2
momentum = 0.9
learning_rate = 1e-2
bn_param = (momentum, learning_rate)
dropout_rate = 0.5
dataset_mean = [0.23280394, 0.24616548, 0.26092353]
dataset_std = [0.16994016, 0.17286949, 0.16250615]


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    # data loader setting
    parser.add_argument(
        "--resolution", default=256, type=int, choices=[32, 64, 96, 128, 144, 160, 224, 256]
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.expanduser("data/"),
        help="path to Aurora data",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.expanduser("acc_datasets"),
        help="output dataset of the accuracy dataset",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="input batch size for training"
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    args = parser.parse_args()

    device_id = torch.cuda.current_device()
    torch.cuda.set_device(device_id)

    os.makedirs(args.output_dir, exist_ok=True)


    ofa_network = OFAProxylessNASNets(
        n_classes=num_classes, 
        bn_param=bn_param, 
        dropout_rate=dropout_rate, 
        base_stage_width="proxyless",
        width_mult=1.3,
        ks_list=[3, 5, 7],
        expand_ratio_list=[3, 4, 6],
        depth_list=[2, 3, 4]
    )

    ofa_network.load_state_dict(
        torch.load("pretrained\OFAProxylessNASNets.pth", map_location="cpu")["state_dict"], strict=True
    )

    ofa_network = ofa_network.to("cuda:%d" % device_id)

    all_results = []
    result_fn = os.path.join(
        args.output_dir,
        f"{ofa_network.__class__.__name__}_r{args.resolution}_gpu{device_id}_acc_table.json",
    )

    for i in tqdm(range(200)):
        cfg = ofa_network.sample_active_subnet()
        subnet = ofa_network.get_active_subnet()

        aurora_dataprovider = AuroraDataProvider(
            data_path=args.data_dir,
            image_size=args.resolution,
            test_batch_size=args.batch_size,
            n_worker=args.workers,
            seed=2
        )
        val_loader = aurora_dataprovider.test

        acc = evaluate_ofa_subnet(subnet, val_loader, args.batch_size, args.resolution)

        cfg["image_size"] = args.resolution
        all_results.append((cfg, acc))
        with open(result_fn, "w") as f:
            json.dump(all_results, f, indent=2)
   

if __name__ == "__main__":
    main()


