from src.dataset import DatasetGenerator
import luojianet_ms.dataset as ds
from luojianet_ms import nn, ops
from src.GCNet import GCNet
import os
import luojianet_ms.context as context
from luojianet_ms import dtype as mstype
from luojianet_ms import Model
from luojianet_ms.nn import Metric, rearrange_inputs
from luojianet_ms.train.callback import Callback, ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
import numpy as np
import argparse

from luojianet_ms.communication.management import init,get_rank,get_group_size

# gpu setting
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'


parser = argparse.ArgumentParser(description='LuoJiaNET GCNet Implement')
parser.add_argument("--train_list", type=str, default="list/whu_training.txt", help="the list for training")
parser.add_argument("--valid_list", type=str, default="list/whu_validation.txt", help="the list for training")
parser.add_argument("--data_root", type=str, default="dataset", help="the dataroot for training")
parser.add_argument("--crop_h", type=int, default=256, help="crop height")
parser.add_argument("--crop_w", type=int, default=512, help="crop width")
parser.add_argument("--max_disp", type=int, default=160, help="max disparity")
parser.add_argument("--batch", type=int, default=1, help="batch size")
parser.add_argument("--epochs", type=int, default=30, help="the number of epoch")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--amp_level", type=str, default='O0', help="amp level")
opt = parser.parse_args()

def read_list(list_path):
    with open(list_path, "r") as f:
        data = f.read().splitlines()

    data = [d.split(",") for d in data]

    return data

def create_dataset(list_file, batch_size, crop_w, crop_h):
    # get rank_id and rank_size
    rank_id = get_rank()
    rank_size = get_group_size()

    # define dataset
    ds.config.set_seed(1)
    dataset_generator = DatasetGenerator(list_file, crop_h, crop_w)
    input_data = ds.GeneratorDataset(dataset_generator, ["data", "label"],
                                     shuffle=True,
                                     num_shards=rank_size,
                                     shard_id=rank_id)
    input_data = input_data.batch(batch_size=batch_size)

    return input_data


class L1Loss(nn.LossBase):
    def __init__(self, mask_thre=0):
        super(L1Loss, self).__init__()
        self.abs = ops.Abs()
        self.mask_thre = mask_thre

    def forward(self, predict, label):
        mask = (label >= self.mask_thre).astype(mstype.float32)
        num = mask.shape[0] * mask.shape[1] * mask.shape[2]
        x = self.abs(predict * mask - label * mask)

        return self.get_loss(x) / ops.ReduceSum()(mask) * num


class PixelErrorPercentage(Metric):
    def __init__(self, error_threshold=1.0):
        super(PixelErrorPercentage, self).__init__()
        self.error_threshold = error_threshold
        self._total_pixel_num = 0
        self._valid_pixel_num = 0

        self.clear()

    def clear(self):
        self._total_pixel_num = 0
        self._valid_pixel_num = 0

    @rearrange_inputs
    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Mean absolute error need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        abs_error_sum = np.abs(y.reshape(y_pred.shape) - y_pred)
        mask = (abs_error_sum < self.error_threshold).astype(np.float32)
        total_mask = (np.ones_like(y)).astype(np.float32)

        self._valid_pixel_num = mask.sum()
        self._total_pixel_num = total_mask.sum()

    def eval(self):
        if self._total_pixel_num == 0:
            raise RuntimeError('Total pixels num must not be 0.')
        return self._valid_pixel_num / self._total_pixel_num


if __name__ == "__main__":
    print("training GCNet...")
    print(opt)

    # Set graph mode and target device
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=int(os.environ["DEVICE_ID"]))

    init()
    # model
    net = GCNet(max_disp=opt.max_disp)

    # loss
    loss_func = L1Loss()
    # optimizer
    net_opt = nn.RMSProp(net.trainable_params(), learning_rate=opt.lr)

    # 执行训练
    model = Model(net, loss_func, net_opt,
                  metrics={"PEP(1 pixel)": PixelErrorPercentage(1.0),
                           "PEP(2 pixel)": PixelErrorPercentage(2.0),
                           "PEP(3 pixel)": PixelErrorPercentage(3.0)},
                  amp_level=opt.amp_level)

    data_list = read_list(opt.train_list)
    val_data_list = read_list(opt.valid_list)
    data_path = []
    val_data_path = []
    
    for item_list in data_list:
        tmp_data_path = []
        for item in item_list:
            tmp_data_path.append(opt.data_root + item)
        data_path.append(tmp_data_path)

    for item_list in val_data_list:
        tmp_data_path = []
        for item in item_list:
            tmp_data_path.append(opt.data_root + item)
        val_data_path.append(tmp_data_path)

    ds_train = create_dataset(data_path, opt.batch, opt.crop_w, opt.crop_h)
    ds_val = create_dataset(val_data_path, opt.batch, opt.crop_w, opt.crop_h)

    # save checkpoint of the model
    config_ck = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(), keep_checkpoint_max=opt.epochs)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_gcnet_whu", directory="checkpoint", config=config_ck)
    time_cb = TimeMonitor()

    output = model.train(opt.epochs, ds_train, callbacks=[ckpoint_cb, LossMonitor(1), time_cb],
                         dataset_sink_mode=False)
    accuracy = model.eval(ds_val, dataset_sink_mode=False)
