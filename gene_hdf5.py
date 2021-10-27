from data_getter.BaseDataGetter import DataGetter
import yaml
import h5py
import numpy as np

data_gettter = DataGetter()

with open('config.yaml', 'r') as f:
	config = yaml.load(f)

datasetDir = config['moyotee_dataset_path']

num_data = 10000
num_data_train = 8000
num_data_valid = 9000
num_data_test = num_data
datasetDir = 'data/moyotee.hdf5'
f = h5py.File(datasetDir, 'w')
train = f.create_group('train')
valid = f.create_group('valid')
test = f.create_group('test')


for i in range(num_data):
    if i % (num_data//100) == 0:
        print(i,num_data)

    split = None

    if i < num_data_train:
        split = train
    elif i >= num_data_train and i < num_data_valid:
        split = valid
    else:
        split = test

    res_img, res_tag_list, res_des_list, res_embedding_list = data_gettter.gene_pic_with_lable_des()
    dt = h5py.special_dtype(vlen=str)


    for j in range(len(res_des_list)):
        example_name = str(i) + '_' + str(j)
        e = res_embedding_list[j]
        ex = split.create_group(example_name)
        ex.create_dataset('name', data=example_name)
        ex.create_dataset('img', data=res_img[:,:,0:3])

        emd = e.detach().numpy().flatten()


        emd_zeros = np.zeros(1024)
        emd_zeros[0:emd.shape[0]] = emd
        emd = emd_zeros


        ex.create_dataset('embeddings', data=emd)
        ex.create_dataset('class', data=example_name)
        ex.create_dataset('txt', data=res_des_list[j]+'\n')

f.close()
