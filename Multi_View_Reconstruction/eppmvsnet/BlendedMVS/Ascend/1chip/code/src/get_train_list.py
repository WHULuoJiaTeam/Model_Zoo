import os
import random
def generate_train_list(data_root):
    dir_list = os.listdir(data_root)
    train_list = random.sample(dir_list,int(len(dir_list)*0.7))
    return train_list
if __name__ == "__main__":
    train_list = generate_train_list('D:\\eppmvsnet\\BlendedMVS')
    print(train_list)
    with open('D:\\eppmvsnet\\training_list.txt','w') as f:
        for item in train_list:
            f.write(item)
            f.write('\n')