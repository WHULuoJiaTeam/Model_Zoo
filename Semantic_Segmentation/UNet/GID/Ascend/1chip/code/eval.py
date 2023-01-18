import os
from tqdm import tqdm
from dataloder_gid_aug import Dataset_RealUD
from unet import unet
import luojianet_ms
from luojianet_ms import dataset
import luojianet_ms.common.dtype as mstype
from metrics import SegmentationMetric
import luojianet_ms.ops.operations as P
import moxing as mox
from config import config


def run_Gid(ClassN, dataset_path, checkpoint_path):
    ##################### define parameters #############################################################
    batch_size_test = 24
    dataset_test = dataset_path
    ##################### load data ##################################################################

    test_dataset_generator = Dataset_RealUD(json_path = dataset_test)
    test_loader = dataset.GeneratorDataset(test_dataset_generator, ["data", "label"], shuffle = False)
    test_loader = test_loader.batch(batch_size_test)

    ###########################################   define model    #########################################
    model = unet(output_class = ClassN)
    luojianet_ms.load_param_into_net(model, luojianet_ms.load_checkpoint(checkpoint_path))
    metric = SegmentationMetric(ClassN)
    metric.reset()
    model.set_train(False)
    model.set_grad(False)
    for input, target in tqdm(test_loader):
        input = input[:, 0, :, :, :]
        input = luojianet_ms.ops.Cast()(input, mstype.float32)
        target = luojianet_ms.ops.Cast()(target, mstype.int32)
        output = model(input)

        output_v = P.ArgMaxWithValue(axis = 1, keep_dims = False)(output)[0].asnumpy()
        target = target.asnumpy()
        # target[target == 6] = 0
        metric.addBatch(output_v, target)

    validate_result = metric.IntersectionOverUnion() * 100
    print('class_IOU:', [round(i, 3) for i in validate_result])
    # MIOU_target = sum(validate_result[1:])/(len(validate_result)-1)
    MIOU = sum(validate_result) / len(validate_result)
    print('MIOU:', MIOU)
    print('class_precision:', metric.classPixelAccuracy())
    print('class_recall:', metric.classPixelAccuracy2())
    print('OA:', metric.pixelAccuracy())
    print('KAPPA:', metric.kappa())


if __name__ == '__main__':
    import argparse
    import luojianet_ms.context as context

    # load data from obs
    mox.file.copy_parallel('obs://luojianet-benchmark-dataset/Semantic_Segmentation/GID/LSC5C/', config.dataset_path)
    mox.file.copy_parallel('obs://luojianet-benchmark/Semantic_Segmentation/UNet/GID/Ascend/1chip/ckpt/unet_119.ckpt',
                           '/cache/unet_test.ckpt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default = r'/home/ma-user/modelarts/user-job-dir/code/LCC5C_b512_woOverlap_test.json')
    parser.add_argument('--checkpoint_path', default = r'/cache/unet_test.ckpt')
    parser.add_argument('--device_target', default = 'Ascend')
    args = parser.parse_args()

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    context.set_context(mode = context.GRAPH_MODE, device_target = args.device_target)
    run_Gid(6, args.dataset_path, args.checkpoint_path)
