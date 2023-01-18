# 将图片和标注数据按比例切分为 训练集和测试集、验证集
import shutil
import random
import os

# 原始路径
image_original_path = '/Users/zhouhuan/Documents/github/Test/CD_data/A/'
label_original_path = '/Users/zhouhuan/Documents/github/Test/CD_data/building_A/'
# 训练集路径
train_image_path = '/Users/zhouhuan/Documents/github/Test/CD_data/A' + '_train/'
train_label_path = '/Users/zhouhuan/Documents/github/Test/CD_data/building_A' + '_train/'
# 验证集路径
val_image_path = '/Users/zhouhuan/Documents/github/Test/CD_data/A' + '_val/'
val_label_path = '/Users/zhouhuan/Documents/github/Test/CD_data/building_A' + '_val/'
# 测试集路径
test_image_path = '/Users/zhouhuan/Documents/github/Test/CD_data/A' + '_test/'
test_label_path = '/Users/zhouhuan/Documents/github/Test/CD_data/building_A' + '_test/'

# 数据集划分比例，训练集70%，验证集00%，测试集30%
train_percent = 0.7
val_percent = 0.0
test_percent = 0.3


# 检查文件夹是否存在
def mkdir():
    if not os.path.exists(train_image_path):
        os.makedirs(train_image_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)

    if not os.path.exists(val_image_path):
        os.makedirs(val_image_path)
    if not os.path.exists(val_label_path):
        os.makedirs(val_label_path)

    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    if not os.path.exists(test_label_path):
        os.makedirs(test_label_path)


def main():
    mkdir()

    label_total_txt = os.listdir(label_original_path)
    image_total_txt = os.listdir(image_original_path)
    num_txt = len(label_total_txt)
    list_all_txt = range(num_txt)  # 范围 range(0, num)

    num_train = int(num_txt * train_percent)
    num_val = int(num_txt * val_percent)
    num_test = num_txt - num_train - num_val

    train = random.sample(list_all_txt, num_train)
    # train从list_all_txt取出num_train个元素
    # 所以list_all_txt列表只剩下了这些元素：val_test
    val_test = [i for i in list_all_txt if not i in train]
    # 再从val_test取出num_val个元素，val_test剩下的元素就是test
    val = random.sample(val_test, num_val)
    # test
    test = [i for i in val_test if not i in val]
    # 检查两个列表元素是否有重合的元素
    # set_c = set(val_test) & set(val)
    # list_c = list(set_c)
    # print(list_c)
    # print(len(list_c))

    print("训练集数目：{}, 验证集数目：{},测试集数目：{}".format(len(train), len(val), len(val_test) - len(val)))

    # txt list
    train_list = []
    train_label_list = []
    val_list = []
    val_label_list = []
    test_list = []
    test_label_list = []

    for i in list_all_txt:
        name = os.path.splitext(label_total_txt[i])[0]
        train_suffix = os.path.splitext(image_total_txt[i])[1]
        label_suffix = os.path.splitext(label_total_txt[i])[1]

        srcImage = image_original_path + name + train_suffix
        srcLabel = label_original_path + name + label_suffix

        if i in train:
            dst_train_Image = train_image_path + name + train_suffix
            dst_train_Label = train_label_path + name + label_suffix
            shutil.copyfile(srcImage, dst_train_Image)
            shutil.copyfile(srcLabel, dst_train_Label)
            train_list.append(dst_train_Image)
            train_label_list.append(dst_train_Label)

        elif i in val:
            dst_val_Image = val_image_path + name + train_suffix
            dst_val_Label = val_label_path + name + label_suffix
            shutil.copyfile(srcImage, dst_val_Image)
            shutil.copyfile(srcLabel, dst_val_Label)
            val_list.append(dst_val_Image)
            val_label_list.append(dst_val_Label)

        else:
            dst_test_Image = test_image_path + name + train_suffix
            dst_test_Label = test_label_path + name + label_suffix
            shutil.copyfile(srcImage, dst_test_Image)
            shutil.copyfile(srcLabel, dst_test_Label)
            test_list.append(dst_test_Image)
            test_label_list.append(dst_test_Label)

    # save txt
    print("SPLIT! SAVING TXT!")
    train_dir = os.path.abspath(os.path.join(train_image_path, ".."))
    val_dir = os.path.abspath(os.path.join(val_image_path, ".."))
    test_dir = os.path.abspath(os.path.join(test_image_path, ".."))

    train_txt_path = os.path.join(train_dir, 'train_list.txt')
    train_txt = open(train_txt_path, 'w')
    train_txt.write(str(train_list))
    train_txt.close()

    val_txt_path = os.path.join(val_dir, 'val_list.txt')
    val_txt = open(val_txt_path, 'w')
    val_txt.write(str(val_list))
    val_txt.close()

    test_txt_path = os.path.join(test_dir, 'test_list.txt')
    test_txt = open(test_txt_path, 'w')
    test_txt.write(str(test_list))
    test_txt.close()
    print(f'TXT SAVED:{train_txt_path},{val_txt_path},{test_txt_path}')


if __name__ == '__main__':
    main()
