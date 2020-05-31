import os
import sys
import numpy as np
import datetime
import h5py
from plyfile import PlyData, PlyElement

#BASE_DIR = os.path.join(os.environ['HOME'], 'workspace', 'tfg', 'pointnet2')
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
point_clouds_path = os.path.join(os.environ['MEDIA'], 'pointcloud', 'clusters_2048_norm', 'train')
hdf5_path = '../data/outdoor_ply_hdf5_2048'
output_filename_prefix = os.path.join(hdf5_path, 'ply_data')

H5_BATCH_SIZE = 10

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def save_h5(h5_filename, data, label, data_dtype='float32', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    #h5_fout.create_dataset('data', data=data, dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)

    #h5_fout.create_dataset('data', data=data, dtype=data_dtype)
    #h5_fout.create_dataset('label', data=label, dtype=label_dtype)

    h5_fout.close()

def load_ply_data(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

"""
alternative: for i in range(0, len(filenames)):
	plydata = PlyData.read("./ply/" + filenames[i] + ".ply")
	for j in range(0, 2048):
		a_data[i, j] = [plydata['vertex']['x'][j], plydata['vertex']['y'][j], plydata['vertex']['z'][j]]
"""

def get_class_names():
    class_names_file = os.path.join(hdf5_path, 'shape_names.txt')
    class_names = [line.rstrip() for line in open(class_names_file)]
    return class_names

class_names = get_class_names()
class_name_dictionary = {class_names[i]: i for i in range(len(class_names))}

def filename_to_class_label(filename):
    class_name = filename.split('/')[-2]
    return class_name_dictionary[class_name]

def main():
    ply_filelist = os.path.join(hdf5_path, 'files.txt')

    print('class_names: {}'.format(class_names))
    print('class_name_dictionary: {}'.format(class_name_dictionary))

    ply_filenames = [line.rstrip() for line in open(ply_filelist)]

    for n in ply_filenames:
        print('- {}'.format(n))

    labels = [filename_to_class_label(fn) for fn in ply_filenames]

    print('labels: {}'.format(labels))

    N = len(labels)

    data_dim = [2048, 3]
    label_dim = [1]
    data_dtype = 'float32'
    label_dtype = 'uint8'

    # output_dir = os.path.join(BASE_DIR, 'data', 'outdoor_ply_hdf5_2048')
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)

    batch_data_dim = [min(H5_BATCH_SIZE, N)] + data_dim
    batch_label_dim = [min(H5_BATCH_SIZE, N)] + label_dim

    h5_batch_data = np.zeros(batch_data_dim, dtype = data_dtype) #np.float32?
    h5_batch_label = np.zeros(batch_label_dim, dtype = label_dtype) #np.uint8?

    print (h5_batch_data.shape)
    print (h5_batch_label.shape)

    for k in range(N):
        if k % 100 == 0:
            print ('Iteration %d/%d' % (k, N))
        d = load_ply_data(os.path.join(point_clouds_path, ply_filenames[k]), 2048)
        # d = pad_arr_rows(d, row=SAMPLING_POINT_NUM)
        l = labels[k]
        h5_batch_data[k%H5_BATCH_SIZE, ...] = d
        h5_batch_label[k%H5_BATCH_SIZE, ...] = l

        if (k+1)%H5_BATCH_SIZE == 0 or k == N - 1:
            print ('[%s] %d/%d' % (datetime.datetime.now(), k+1, N))
            h5_filename = output_filename_prefix + '_train0.h5'
            begidx = 0
            endidx = k % H5_BATCH_SIZE + 1
            save_h5(h5_filename, h5_batch_data[begidx:endidx, ...],
                    h5_batch_label[begidx:endidx, ...],
                    data_dtype, label_dtype)
            print ('Stored %d objects' % (endidx - begidx))

main()
d, l = load_h5(output_filename_prefix + '_train0.h5')
print ('data shape: ', d.shape)
print ('label shape: ', l.shape)
