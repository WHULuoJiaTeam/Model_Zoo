from easydict import EasyDict as ed
'''
You can forward the following directory structure from your dataset files and read by LuoJiaNet's API.

    .. code-block::

        .
        └── image_folder_dataset_directory
             ├── T1
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── T2
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── Label
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
'''
config = ed({
    "device_target":"Ascend",      #GPU \ CPU \ Ascend
    "device_id":0,  #显卡ID
    "dataset_path": "/cache/data",  #数据存放位置
    "save_checkpoint_path": "/cache/checkpoint",  #保存的参数存放位置
})
