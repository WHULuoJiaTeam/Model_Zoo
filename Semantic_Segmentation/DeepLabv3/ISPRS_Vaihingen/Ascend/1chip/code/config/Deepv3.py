param = {}
param['random_seed']=10000
param['device_target']='Ascend'
param['device_id']=0
# dataset
param['data_dir'] = '/cache/dataset'
param['train_image_id_txt_path']=param['data_dir']+'/train_512.txt'
param['val_image_id_txt_path']=param['data_dir']+'/val_512.txt'
param['train_transform']='train_transform'
param['val_transform']='val_transform'
param['num_workers']=2
param['in_channels'] = 3
param['model_network'] = "DeepLabV3"
# Training parameters
param['epochs'] = 200
param['train_batch_size'] = 8
param['test_batch_size'] = 1
param['lr'] = 0.0001
param['weight_decay'] = 5e-4
param['save_inter'] = 1
param['iter_inter'] = 10
param['min_inter'] = 20
param['n_class'] = 6

# Load the weight path (continue training)
param['load_ckpt_dir'] = None

# Save path
param['extra_log'] = 'vaihingen_6_classes'
param['save_dir'] = (param['model_network'] + '_' + param['extra_log']).strip('_')