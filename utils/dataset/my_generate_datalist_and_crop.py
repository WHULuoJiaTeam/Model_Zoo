import numpy as np
import os
from osgeo import gdal
import itertools
import cv2


def compress(path, target_path, method="LZW"):
    """使用gdal进行文件压缩，
          LZW方法属于无损压缩，
          效果非常给力，4G大小的数据压缩后只有三十多M"""
    dataset = gdal.Open(path)
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(target_path, dataset, strict=1, options=["TILED=YES", "COMPRESS={0}".format(method)])
    del dataset


def get_isprs_labels():
    return np.array([
        [255, 255, 255],  # 0 : Impervious surfaces (white)
        [0, 0, 255],  # 1 : Buildings (blue)
        [0, 255, 255],  # 2 : Low vegetation (cyan)
        [0, 255, 0],  # 3 : Trees (green)
        [255, 255, 0],  # 4 : Cars (yellow)
        [255, 0, 0]])  # 5 : Clutter (red)


def decode_label(encode_label_path, decode_label_path):
    if not os.path.exists(decode_label_path):
        os.makedirs(decode_label_path)
    for filename in os.listdir(encode_label_path):
        if os.path.isdir(os.path.join(encode_label_path, filename)):
            continue
        label = cv2.imread(os.path.join(encode_label_path, filename), -1)
        h, w = label.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        clr_map = get_isprs_labels()

        for y in range(h):
            for x in range(w):
                vis[y][x] = clr_map[label[y][x]]

        b, g, r = cv2.split(vis)
        vis = cv2.merge([r, g, b])
        cv2.imwrite(os.path.join(decode_label_path, filename), vis)


def encode_label(orign_label_path, encode_label_path):
    if not os.path.exists(encode_label_path):
        os.makedirs(encode_label_path)
    for filename in os.listdir(orign_label_path):
        if os.path.isdir(os.path.join(orign_label_path, filename)):
            continue
        label = cv2.imread(os.path.join(orign_label_path, filename))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        h, w, c = label.shape
        new_label = np.zeros((h, w, 1), dtype=np.int8)

        cls_to_clr_map = get_isprs_labels()

        for i in range(cls_to_clr_map.shape[0]):
            new_label[np.where(np.all(label.astype(np.int32) == cls_to_clr_map[i], axis=-1))] = i

        cv2.imwrite(os.path.join(encode_label_path, filename), new_label)


def generate_datalist(img_folder, label_folder, tatget_dir):
    # img_format = ['.tif', '.TIF', '.tiff', '.TIFF', '.img', '.IMG', '.bmp', '.BMP', '.jpg', '.JPG', '.png', '.PNG']

    # Create folder is not exist
    if not os.path.exists(tatget_dir):
        os.makedirs(tatget_dir)

    all_txt = os.path.join(tatget_dir, "all.txt")
    train_txt = os.path.join(tatget_dir, "train_all.txt")
    val_txt = os.path.join(tatget_dir, "val_all.txt")
    imgA = open(all_txt, 'w')
    imgT = open(train_txt, 'w')
    imgV = open(val_txt, 'w')

    # 适应多文件目录
    # all_img_folders = os.listdir(img_folder)
    # for cur_img_folder in all_img_folders:
    #     cur_all_imgs_path = (os.path.join(img_folder, cur_img_folder)).replace("\\", "/")
    #     cur_all_labs_path = (os.path.join(label_folder, cur_img_folder)).replace("\\", "/")
    #     if (os.path.isdir(cur_all_imgs_path)) and (os.path.isdir(cur_all_labs_path)):
    #         print("need some code adjust")
    #         cur_all_imgs = os.listdir(cur_all_imgs_path)
    #         exit(0)
    #     else:
    #         cur_all_imgs = all_img_folders

    cur_all_imgs = os.listdir(img_folder)
    all_list = cur_all_imgs
    for filename in all_list:
        cur_img_post_fix = filename[filename.rindex('.'):]
        # exist_count = img_format.count(cur_img_post_fix)
        cur_img_full_name = (os.path.join(img_folder, filename)).replace("\\", "/")
        if cur_img_post_fix == '.tiff':
            filename = filename.replace('.tiff', '.tif')
            cur_label_full_name = (os.path.join(label_folder, filename)).replace("\\", "/")
        else:
            cur_label_full_name = (os.path.join(label_folder, filename)).replace("\\", "/")

        # if (os.path.isfile(cur_img_full_name) and os.path.isfile(cur_label_full_name))\
        #         or (exist_count > 0) :  # 有些格式的文件会不被系统视为文件, 排除该情况
        if (os.path.isfile(cur_img_full_name) and os.path.isfile(cur_label_full_name)):
            linetxt = str(cur_img_full_name + " " + cur_label_full_name + "\n")
            imgA.write(str(linetxt))

    data_numbers = len(all_list)
    train_numbers = int(np.floor(data_numbers * 0.7))
    if train_numbers == 0:
        train_list = all_list
        val_list = all_list
    else:
        import random
        train_list = random.sample(all_list, train_numbers)
        val_list = list(set(all_list) - set(train_list))

    for filename in train_list:
        cur_img_post_fix = filename[filename.rindex('.'):]
        # exist_count = img_format.count(cur_img_post_fix)
        cur_img_full_name = (os.path.join(img_folder, filename)).replace("\\", "/")

        if cur_img_post_fix == '.tiff':
            filename = filename.replace('.tiff', '.tif')
            cur_label_full_name = (os.path.join(label_folder, filename)).replace("\\", "/")
        else:
            cur_label_full_name = (os.path.join(label_folder, filename)).replace("\\", "/")

        # if (os.path.isfile(cur_img_full_name) and os.path.isfile(cur_label_full_name))\
        #         or (exist_count > 0):
        if (os.path.isfile(cur_img_full_name) and os.path.isfile(cur_label_full_name)):
            linetxt = str(cur_img_full_name + " " + cur_label_full_name + "\n")
            imgT.write(str(linetxt))

    for filename in val_list:
        cur_img_post_fix = filename[filename.rindex('.'):]
        # exist_count = img_format.count(cur_img_post_fix)
        cur_img_full_name = (os.path.join(img_folder, filename)).replace("\\", "/")

        if cur_img_post_fix == '.tiff':
            filename = filename.replace('.tiff', '.tif')
            cur_label_full_name = (os.path.join(label_folder, filename)).replace("\\", "/")
        else:
            cur_label_full_name = (os.path.join(label_folder, filename)).replace("\\", "/")

        # if (os.path.isfile(cur_img_full_name) and os.path.isfile(cur_label_full_name))\
        #         or (exist_count > 0):
        if (os.path.isfile(cur_img_full_name) and os.path.isfile(cur_label_full_name)):
            linetxt = str(cur_img_full_name + " " + cur_label_full_name + "\n")
            imgV.write(str(linetxt))

    imgA.close()
    imgT.close()
    imgV.close()


def read_data(filename):
    dataset = gdal.Open(filename)
    if dataset == None:
        print(filename + "文件无法打开")
        return
    width = dataset.RasterXSize  # 栅格矩阵的列数  宽
    height = dataset.RasterYSize  # 栅格矩阵的行数  高
    data = dataset.ReadAsArray(0, 0, width, height)
    geotrans = dataset.GetGeoTransform()
    del dataset
    return data, geotrans


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        # itertools.islice(iterable, start, stop[, step])
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


# Slide for GDAL-data array
def sliding_window(image, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, image.shape[1], step):
        if x + window_size[0] > image.shape[1]:
            x = image.shape[1] - window_size[0]
        for y in range(0, image.shape[2], step):
            if y + window_size[1] > image.shape[2]:
                y = image.shape[2] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def write_data(data, geotrans, output_file):
    # Datatype
    if 'int8' in data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # Dimensions
    if len(data.shape) == 3:
        bands, height, width = data.shape
    else:
        bands, (height, width) = 1, data.shape

    # Create file to write
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_file, width, height, bands, datatype)
    dataset.SetGeoTransform(geotrans)
    if bands == 1:
        dataset.GetRasterBand(1).WriteArray(data)
    else:
        for i in range(bands):
            dataset.GetRasterBand(i + 1).WriteArray(data[i])

    del dataset


def make_crop_and_gendatalist(train_all_datalist, val_all_datalist,
                              img_save_path, label_save_path,
                              patch_size, ignore_label, data_info_threshold, special_type='Non-Massachusetts'):
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)

    # 将切分后的文件名写入txt文件
    dataset_txt_name = train_all_datalist.replace("\\", "/")
    root = dataset_txt_name[:dataset_txt_name.rindex("/")]
    out_train_txt = os.path.join(root, "train_{}".format(patch_size[0]) + ".txt")
    trainF = open(out_train_txt, 'w')

    out_valid_txt = os.path.join(root, "val_{}".format(patch_size[0]) + ".txt")
    valF = open(out_valid_txt, 'w')

    # Delete the pre-contents in save_path
    for i in os.listdir(img_save_path):
        path_file = os.path.join(img_save_path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
    for j in os.listdir(label_save_path):
        path_file = os.path.join(label_save_path, j)
        if os.path.isfile(path_file):
            os.remove(path_file)

    # 从train_all.txt大幅面数据列表中生成切分的数据
    for index, line in enumerate(open(train_all_datalist, 'r', encoding='gbk')):
        name = line.split()
        if len(name) == 0 or len(name) != 2 or name[0] == '\n':
            continue

        left_input_name = name[0].replace("\\", '/')
        right_label_name = name[1].replace("\\", '/')
        # if not os.path.isfile(left_input_name):
        #     continue
        # if not os.path.isfile(right_label_name):
        #     continue

        # img_format = ['.tif', '.TIF', '.tiff', '.TIFF', '.img', '.IMG', '.bmp', '.BMP', '.jpg', '.JPG', '.png', '.PNG']
        # cur_left_img_post_fix = left_input_name[left_input_name.rindex('.'):]
        # if cur_left_img_post_fix == '.tiff':
        #     left_input_name = left_input_name.replace('.tiff', '.tif')
        #
        # cur_right_img_post_fix = right_label_name[right_label_name.rindex('.'):]
        # if cur_right_img_post_fix == '.tiff':
        #     right_label_name = right_label_name.replace('.tiff', '.tif')

        # 读取大幅面数据，进行切分
        # GDAL read
        image, image_geotrans = read_data(left_input_name)
        label, label_geotrans = read_data(right_label_name)

        # Get a sequence patch
        for data_number, coords in enumerate(
                grouper(1, sliding_window(image, step=patch_size[0], window_size=patch_size))):
            for x, y, h, w in coords:
                image_patches = image[:, x:x + h, y:y + w]
                label_pathces = label[x:x + h, y:y + w]

                # Ori-ignore value may == 255
                # Massachusetts数据集道路值为255
                if special_type != "Massachusetts":
                    label_pathces[label_pathces == 255] = ignore_label

                # Ignore the non-information patch
                mask = np.zeros_like(label_pathces)
                mask[label_pathces > 0] = 1
                if np.sum(mask) / (patch_size[0] * patch_size[1]) < data_info_threshold:
                    continue

                # Write patch
                cur_img_post_fix = left_input_name[left_input_name.rindex('.'):]
                if cur_img_post_fix == '.tiff':
                    base_name = os.path.basename(left_input_name)[:-5]
                else:
                    base_name = os.path.basename(left_input_name)[:-4]

                patch_name = base_name + '_' + str(data_number) + '_' + str(y) + '_' + str(x) + '.tif'
                img_train_patch_fullname = os.path.join(img_save_path, patch_name)
                label_train_patch_fullname = os.path.join(label_save_path, patch_name)
                img_train_patch_fullname = img_train_patch_fullname.replace("\\", '/')
                label_train_patch_fullname = label_train_patch_fullname.replace("\\", '/')

                linetxt = str(img_train_patch_fullname + " " + label_train_patch_fullname + "\n")
                print(linetxt)
                trainF.write(str(linetxt))

                # Default: ori-image filename == ori-label filename
                write_data(image_patches, image_geotrans, img_train_patch_fullname)
                write_data(label_pathces, label_geotrans, label_train_patch_fullname)

    # 从val_all.txt大幅面数据列表中生成切分的数据
    for index, line in enumerate(open(val_all_datalist, 'r', encoding='gbk')):
        name = line.split()
        if len(name) == 0 or len(name) != 2 or name[0] == '\n':
            continue

        left_input_name = name[0].replace("\\", '/')
        right_label_name = name[1].replace("\\", '/')
        # if not os.path.isfile(left_input_name):
        #     continue
        # if not os.path.isfile(right_label_name):
        #     continue

        # GDAL read
        image, image_geotrans = read_data(left_input_name)
        label, label_geotrans = read_data(right_label_name)

        # Get a sequence patch
        for data_number, coords in enumerate(
                grouper(1, sliding_window(image, step=patch_size[0], window_size=patch_size))):
            for x, y, h, w in coords:
                image_patches = image[:, x:x + h, y:y + w]
                label_pathces = label[x:x + h, y:y + w]

                # Ori-ignore value may == 255
                if special_type != "Massachusetts":
                    label_pathces[label_pathces == 255] = ignore_label

                # Ignore the non-information patch
                mask = np.zeros_like(label_pathces)
                mask[label_pathces > 0] = 1
                if np.sum(mask) / (patch_size[0] * patch_size[1]) < data_info_threshold:
                    continue

                # Write patch
                cur_img_post_fix = left_input_name[left_input_name.rindex('.'):]
                if cur_img_post_fix == '.tiff':
                    base_name = os.path.basename(left_input_name)[:-5]
                else:
                    base_name = os.path.basename(left_input_name)[:-4]
                # base_name = os.path.basename(left_input_name)[:-4]
                patch_name = base_name + '_' + str(data_number) + '_' + str(y) + '_' + str(x) + '.tif'
                img_val_patch_fullname = os.path.join(img_save_path, patch_name)
                label_val_patch_fullname = os.path.join(label_save_path, patch_name)
                img_val_patch_fullname = img_val_patch_fullname.replace("\\", '/')
                label_val_patch_fullname = label_val_patch_fullname.replace("\\", '/')

                linetxt = str(img_val_patch_fullname + " " + label_val_patch_fullname + "\n")
                print(linetxt)
                valF.write(str(linetxt))

                # Default: ori-image filename == ori-label filename
                write_data(image_patches, image_geotrans, img_val_patch_fullname)
                write_data(label_pathces, label_geotrans, label_val_patch_fullname)

    trainF.close()
    valF.close()


if __name__ == '__main__':
    # PreStep(If need): encode and decode color-labels
    orign_label_path = '/zhdata/Vaihingen_dataset/train'
    encode_label_path = '/zhdata/Vaihingen_dataset/label_one'
    encode_label(orign_label_path, encode_label_path)

    # decode_label_path = '/zhdata/Vaihingen_dataset/label_one'
    # decode_label(encode_label_path, decode_label_path)


    # Vaihingen settings
    img_folder = '/zhdata/Vaihingen_dataset/train'
    label_folder = '/zhdata/Vaihingen_dataset/label_one'
    target_dir = '/zhdata/Vaihingen_dataset/Vaihingen_split'
    # Step1: Generate datalist for large RS data.
    generate_datalist(img_folder, label_folder, target_dir)
    # Step2: Crop large RS data from datalist and generate corresponding datalist
    cropsize_list = [1024, 512]
    ignore_label = 0  # Do not change easily
    data_info_threshold = 0

    all_datalist = os.path.join(target_dir, "all.txt")
    train_all_datalist = os.path.join(target_dir, "train_all.txt")
    val_all_datalist = os.path.join(target_dir, "val_all.txt")
    for cropsize in cropsize_list:
        patch_size = (cropsize, cropsize)
        img_save_path = os.path.join(target_dir, "cropped{}/img".format(patch_size[0]))
        label_save_path = os.path.join(target_dir, "cropped{}/label".format(patch_size[0]))
        make_crop_and_gendatalist(train_all_datalist, val_all_datalist,
                                  img_save_path, label_save_path,
                                  patch_size, ignore_label, data_info_threshold)
