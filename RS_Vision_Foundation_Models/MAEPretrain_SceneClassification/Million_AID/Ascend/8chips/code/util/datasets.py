# Copyright 2021, 2022, 2023 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022, 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

"""
It is the old APIs of LuoJiaNET, in new version we recommend:
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2
"""
import luojianet_ms.dataset.vision.py_transforms as C
import luojianet_ms.dataset.transforms.c_transforms as C2
import cv2

from luojianet_ms.dataset.vision import Inter
import luojianet_ms.common.dtype as mstype
import luojianet_ms.dataset as de

import os
from PIL import Image


# class MillionAIDDataset(data.Dataset):
class MillionAIDDataset:
    # def __init__(self, root, train=True, transform=None, tag=100):
    def __init__(self, root, train=True, tag=100):

        print(os.getcwd())

        with open(os.path.join(root, 'train_labels_{}.txt'.format(tag)), mode='r') as f:
            train_infos = f.readlines()
        f.close()

        # train files
        trn_files = []
        trn_targets = []

        for item in train_infos:
            fname, _, idx = item.strip().split()
            trn_file = (os.path.join(root + '/all_img', fname)).replace("\\", "/")
            trn_files.append(trn_file)
            trn_targets.append(int(idx))

        with open(os.path.join(root, 'valid_labels.txt'), mode='r') as f:
            valid_infos = f.readlines()
        f.close()

        # val files
        val_files = []
        val_targets = []

        for item in valid_infos:
            fname, _, idx = item.strip().split()
            val_file = (os.path.join(root + '/all_img', fname)).replace("\\", "/")
            val_files.append(val_file)
            val_targets.append(int(idx))

        if train:
            self.files = trn_files
            self.targets = trn_targets
        else:
            self.files = val_files
            self.targets = val_targets

        # self.transform = transform

        print('Creating MillionAID dataset with {} examples'.format(len(self.targets)))

    def __len__(self):
        # return len(self.targets)
        return len(self.files)

    def __getitem__(self, i):
        img_path = self.files[i]

        # img = Image.open(img_path).convert('RGB')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # img = self.transform(img)

        # return img, self.targets[i]
        return img


def build_transform(is_train, args):
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    if is_train:
        # train transform
        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_train = [
            C.ToPIL(),  # Convert to PIL image type.
            C.RandomResizedCrop(size=args.input_size, scale=(0.2, 1.0), interpolation=Inter.BICUBIC),
            C.RandomHorizontalFlip(prob=0.5),
            C.ToTensor(),
            C.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            # C.HWC2CHW(),
        ]
        transform = transform_train
        return transform
    else:
        # eval transform
        # t = []
        # crop_pct = 224 / 256
        # size = int(args.input_size / crop_pct)
        # t.append(
        #     transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        # )
        # t.append(transforms.CenterCrop(args.input_size))
        #
        # t.append(transforms.ToTensor())
        # t.append(transforms.Normalize(mean, std))
        # return transforms.Compose(t)
        crop_pct = 224 / 256
        transform_eval = [
            C.ToPIL(),  # Convert to PIL image type.
            C.Resize(int(args.input_size / crop_pct), interpolation=Inter.BICUBIC),
            C.CenterCrop(args.input_size),
        ]
        transform_eval += [
            C.ToTensor(),
            C.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            # C.HWC2CHW(),
        ]
        return transform_eval


def build_dataset(is_train, args):
    """Create LuoJiaNET Dataset object."""
    if args.dataset == 'millionaid':
        print('Loading MillionAID dataset!')
        # data_path = '../Dataset/millionaid/'
        # args.nb_classes = 51

        if args.use_parallel:
            # dataset = MillionAIDDataset(data_path, train=is_train, transform=transform, tag=args.tag)
            # dataset = de.GeneratorDataset(
            #     source=MillionAIDDataset(root=args.data_path, train=is_train, tag=args.tag),
            #     column_names=["image", "label"],
            #     num_parallel_workers=args.num_workers)
            dataset = de.GeneratorDataset(
                source=MillionAIDDataset(root=args.data_path, train=is_train, tag=args.tag),
                column_names="image",
                num_parallel_workers=args.num_workers)

            # num_tasks = misc.get_world_size()
            rank_id = int(os.getenv('RANK_ID', '0'))
            # sampler_train = torch.utils.data.DistributedSampler(
            #     dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            sampler_train = de.DistributedSampler(num_shards=args.device_num, shard_id=rank_id,
                                                  shuffle=True, num_samples=None)
            dataset.use_sampler(sampler_train)
            print("Sampler_train = %s" % str(sampler_train))
        else:
            # sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_train = de.RandomSampler(replacement=False, num_samples=None)
            dataset = de.GeneratorDataset(
                source=MillionAIDDataset(root=args.data_path, train=is_train, tag=args.tag),
                column_names="image",
                sampler=sampler_train,
                num_parallel_workers=args.num_workers)
            print("Sampler_train = %s" % str(sampler_train))

        # data_loader_train = torch.utils.data.DataLoader(
        #     dataset_train, sampler=sampler_train,
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers,
        #     pin_memory=args.pin_mem,
        #     drop_last=True,
        # )
        transform = build_transform(is_train, args)
        ds = dataset.map(input_columns="image", num_parallel_workers=args.num_workers,
                         operations=transform, python_multiprocessing=True)

        # type_cast_op = C2.TypeCast(mstype.int32)
        # ds = ds.map(input_columns="label", num_parallel_workers=args.num_workers, operations=type_cast_op)

        ds = ds.batch(args.batch_size, drop_remainder=True)
        ds = ds.repeat(1)
    else:
        raise NotImplementedError

    return ds


if __name__ == "__main__":
    train_data_set_dir = 'D:/small_dataset_test/Million-AID'
    dataset = de.GeneratorDataset(
        source=MillionAIDDataset(root=train_data_set_dir, train=True, tag=51),
        column_names="image",
        num_parallel_workers=1)

    # dt = dataset.shuffle(buffer_size=10000)
    # dt = dt.batch(batch_size=1)
    # for data in dt.create_dict_iterator():
    #     # print("rgb: {}".format(data["image"].shape), "GT: {}".format(data["label"].shape))
    #     print("rgb: {}".format(data["image"].shape))

    transform_train = [
        C.ToPIL(),
        C.RandomResizedCrop(size=224, scale=(0.2, 1.0), interpolation=Inter.BICUBIC),
        C.RandomHorizontalFlip(prob=0.5),
        C.ToTensor(),
        C.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    ds = dataset.map(input_columns="image", num_parallel_workers=1, operations=transform_train, python_multiprocessing=False)
    ds = ds.shuffle(buffer_size=10000)
    ds = ds.batch(batch_size=1)
    for data in ds.create_dict_iterator():
        # print("rgb: {}".format(data["image"].shape), "GT: {}".format(data["label"].shape))
        print("rgb: {}".format(data["image"].shape))