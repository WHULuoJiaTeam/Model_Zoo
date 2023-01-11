import os
import ast
import argparse
import time
import numpy as np
from luojianet_ms import context, Tensor
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
from src.ssd import SsdInferWithDecoder, ssd_resnet34
from src.dataset import create_ssd_dataset, create_mindrecord
from src.config import config
from src.eval_utils import metrics
from src.box_utils import default_boxes
from src.logger import get_logger


def ssd_eval(dataset_path, ckpt_path, anno_json):
    """SSD evaluation."""
    batch_size = 1
    ds = create_ssd_dataset(dataset_path, batch_size=batch_size, repeat_num=1,
                            is_training=False, use_multiprocessing=False)
    if config.model == "ssd_resnet34":
        net = ssd_resnet34(config=config)
    else:
        raise ValueError(f'config.model: {config.model} is not supported')
    net = SsdInferWithDecoder(net, Tensor(default_boxes), config)

    print("Load Checkpoint!")
    param_dict = load_checkpoint(ckpt_path)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)

    net.set_train(False)
    i = batch_size
    total = ds.get_dataset_size() * batch_size
    start = time.time()
    pred_data = []
    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    for data in ds.create_dict_iterator(output_numpy=True, num_epochs=1):
        img_id = data['img_id']
        img_np = data['image']
        image_shape = data['image_shape']

        output = net(Tensor(img_np))
        for batch_idx in range(img_np.shape[0]):
            pred_data.append({"boxes": output[0].asnumpy()[batch_idx],
                              "box_scores": output[1].asnumpy()[batch_idx],
                              "img_id": int(np.squeeze(img_id[batch_idx])),
                              "image_shape": image_shape[batch_idx]})

        percent = round(i / total * 100., 2)

        print(f'    {str(percent)} [{i}/{total}]', end='\r')
        i += batch_size
    cost_time = int((time.time() - start) * 1000)
    print(f'    100% [{total}/{total}] cost {cost_time} ms')
    res, mAP = metrics(pred_data, anno_json)
    print("\n========================================\n")
    print(f"mAP: {mAP}")
    config.logger.info('{}'.format(res))



def get_eval_args():
    """Get eval args"""
    parser = argparse.ArgumentParser(description='SSD evaluation')
    parser.add_argument("--data_url", type=str, default="/media/xx/新加卷/DATA/NWPU")
    parser.add_argument("--train_url", type=str, default="/media/xx/新加卷/OUTPUT/NWPU/ssd/test1/train")
    parser.add_argument("--mindrecord", type=str, default="/media/xx/新加卷/DATA/NWPU/NWPU_SSD_EVAL")
    parser.add_argument("--log_path", type=str, default="/media/xx/新加卷/OUTPUT/NWPU/ssd/test2/train")
    parser.add_argument("--run_online", type=ast.literal_eval, default=False)
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
    parser.add_argument("--checkpoint_floder", type=str, required=True, help="Checkpoint file folder.")
    parser.add_argument("--run_platform", type=str, default="GPU", choices=("Ascend", "GPU", "CPU"),
                        help="run platform, support Ascend ,GPU and CPU.")
    return parser.parse_args()


if __name__ == '__main__':
    args_opt = get_eval_args()
    config.checkpoint_path = args_opt.checkpoint_floder
    config.coco_root = args_opt.data_url
    config.mindrecord_dir = args_opt.mindrecord
    config.logger = get_logger(args_opt.log_path, 0)

    if args_opt.dataset == "coco":
        json_path = os.path.join(config.coco_root, config.instances_set.format(config.val_data_type))
    elif args_opt.dataset == "voc":
        json_path = os.path.join(config.voc_root, config.voc_json)
    else:
        raise ValueError('SSD eval only support dataset mode is coco and voc!')

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.run_platform, device_id=args_opt.device_id)
    mindrecord_file = create_mindrecord(args_opt.dataset, "ssd_eval.mindrecord", False)

    if args_opt.run_online:
        import moxing as mox

        mox.file.copy_parallel(mindrecord_file, args_opt.mindrecord)
    print("Start Eval!")
    ckpt_file = os.listdir(config.checkpoint_path)
    for ckpt in ckpt_file:
        if ckpt.endswith('.ckpt'):
            config.logger.info('ckpt_file:{}'.format(os.path.join(config.checkpoint_path, ckpt)))
            ssd_eval(mindrecord_file, os.path.join(config.checkpoint_path, ckpt), json_path)
        else:
            continue