from easydict import EasyDict as ed

config = ed({
    "device_target":"Ascend",      #GPU \ CPU \ Ascend
    "device_id":0,  #显卡ID
    "dataset_path": "/cache/data",  #数据存放位置
    "save_checkpoint_path": "/cache/checkpoint",  #保存的参数存放位置
})
