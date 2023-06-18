# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# Copyright 2020 - present, Facebook, Inc
# Copyright 2023 Devrim Cavusoglu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Dataset related functions and classes used in main.py. This file is taken
and adapted for STD implementation from Facebook Research DeiT repository.
See the original source below
https://github.com/facebookresearch/deit/blob/main/datasets.py

NOTE: Since the original name of the file `datasets` is conflicting
with a `datasets` Python package, we renamed the file as `dataset.py`.
"""
import json
import os

import datasets
import PIL.Image
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision import datasets as pt_datasets
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader


class HFDataset(Dataset):
    """Dataset from HuggingFace loader."""

    def __init__(self, data_root, split, pipeline=None):
        self.data_source = datasets.load_dataset(data_root, split=split)
        self.pipeline = pipeline

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        instance = self.data_source[idx]
        image: PIL.Image.Image = instance["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.pipeline is not None:
            image = self.pipeline(image)

        return image, instance["label"]


class INatDataset(ImageFolder):
    def __init__(
        self,
        root,
        train=True,
        year=2018,
        transform=None,
        target_transform=None,
        category="name",
        loader=default_loader,
    ):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, "categories.json")) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter["annotations"]:
            king = []
            king.append(data_catg[int(elem["category_id"])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data["images"]:
            cut = elem["file_name"].split("/")
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == "CIFAR":
        dataset = pt_datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform, download=True
        )
        nb_classes = 100
    elif args.data_set == "IMNET":
        dataset = HFDataset(
            "imagenet-1k", split="train" if is_train else "validation", pipeline=transform
        )
        nb_classes = 1000
    elif args.data_set == "IMNET-TINY":
        if is_train:
            dataset = HFDataset(
                "Multimodal-Fatima/Imagenet1k_sample_train", split="train", pipeline=transform
            )
        else:
            dataset = HFDataset("theodor1289/imagenet-1k_tiny", split="train", pipeline=transform)
        nb_classes = 1000
    elif args.data_set == "INAT":
        dataset = INatDataset(
            args.data_path, train=is_train, year=2018, category=args.inat_category, transform=transform
        )
        nb_classes = dataset.nb_classes
    elif args.data_set == "INAT19":
        dataset = INatDataset(
            args.data_path, train=is_train, year=2019, category=args.inat_category, transform=transform
        )
        nb_classes = dataset.nb_classes
    else:
        raise ValueError(f"Unknown dataset {args.data_set}!")

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
