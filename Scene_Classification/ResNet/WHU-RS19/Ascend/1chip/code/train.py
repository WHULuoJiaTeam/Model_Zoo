from luojianet_ms import context
from luojianet_ms import Tensor
from luojianet_ms.nn import SGD, RMSProp
from luojianet_ms.context import ParallelMode
from luojianet_ms.train.model import Model
from luojianet_ms.communication.management import init, get_rank, get_group_size
from luojianet_ms.train.callback import ModelCheckpoint, CheckpointConfig,TimeMonitor,LossMonitor
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
from luojianet_ms.communication.management import init
from luojianet_ms.train.loss_scale_manager import FixedLossScaleManager
from luojianet_ms.common import dtype as mstype
from luojianet_ms.common import set_seed
import os

from utils import get_lr,create_dataset,CrossEntropySmooth
from config import config
# import moxing as mox
# from vgg import *
from Resnet import *
# from Resnet_se import *
set_seed(1)
CACHE = "/cache/data/"
import moxing as mox

def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)

if __name__ == '__main__':

    # mox.file.copy_parallel(os.environ['DLS_DATA_URL'], "WHU-RS19/")

    context.set_context(device_target=config.device_target)
    if config.device_target == 'Ascend':
        device_id = get_device_id()
        context.set_context(device_id=device_id)
    init()
    config.rank = get_rank()
    device_num = get_group_size()
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=device_num,
                                      parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True)

    # define network
    # net = vgg11_bn(num_classes=config.class_num)
    net = resnet152(num_classes=config.class_num)
    # net = se_resnet18(num_classes=config.class_num)
    
    # define loss
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    # define dataset
    
    mox.file.copy_parallel(src_url=config.dataset_path, dst_url=CACHE)
    dataset = create_dataset(dataset_path=CACHE,
                            do_train=True,
                            batch_size=config.batch_size, num_shards=device_num, shard_id=config.rank)
    step_size = dataset.get_dataset_size()

    # resume
    if config.resume:
        ckpt = load_checkpoint(config.resume)
        load_param_into_net(net, ckpt)

    # get learning rate
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    lr = Tensor(get_lr(lr_init=config.lr_init,
                       lr_end=config.lr_end,
                       lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=config.epoch_size,
                       steps_per_epoch=step_size,
                       lr_decay_mode=config.lr_decay_mode))

    # define optimization
    if config.opt == 'sgd':
        optimizer = SGD(net.trainable_params(), learning_rate=lr, momentum=config.momentum,
                        weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    elif config.opt == 'rmsprop':
        optimizer = RMSProp(net.trainable_params(), learning_rate=lr, decay=0.9, weight_decay=config.weight_decay,
                            momentum=config.momentum, epsilon=config.opt_eps, loss_scale=config.loss_scale)

    # define model
    model = Model(net, loss_fn=loss, optimizer=optimizer, loss_scale_manager=loss_scale,
                  metrics={'acc'})

    callbacks = [LossMonitor(per_print_times=10),
                 TimeMonitor(data_size=step_size)]
    if config.rank == 0:
        time_cb = TimeMonitor(data_size=dataset.get_dataset_size())
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs,
                                    keep_checkpoint_max=config.keep_checkpoint_max)
        save_ckpt_path = os.path.join(config.save_checkpoint_path, 'model' + '/')
        ckpoint_cb = ModelCheckpoint(prefix="net", directory=save_ckpt_path, config=config_ck)
        callbacks.append(ckpoint_cb)

    # begine train
    print("============== Starting Training ==============")
    model.train(config.epoch_size, dataset, callbacks=callbacks)
    mox.file.copy_parallel(src_url=config.save_checkpoint_path, dst_url=config.obs_checkpoint_path)

