import numpy as np
import pandas as pd
import yaml
import os
import json
from tools.trajectory import Trajectory
from tools.dtw_mt import compute_dtw_mt
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


data_path = '/home/jerry/Robobarista/robobarista_dataset/dataset'
# obj_list = ['obj%03d' % (i+1) for i in range(120)]
### ['obj001', 'obj002', 'obj003', 'obj004', 'obj005',...]
### Note: no 'obj028', 'obj036', 'obj041', 'obj100' directories

pointcloud_dim = 1000
language_dim = 250
trajectory_dim = 150


##### Paths
def get_modalality_path():
    """Return [pc_path, language_str] and [pc_path, language_str, traj_path]
    array with shape=(249, 2) and (1225, 3) respectively.
    """

    with open(os.path.join(data_path, 'folds.json')) as f:
        folds = json.load(f)

    p_l_pair_path = []
    p_l_t_pair_path = []

    for obj, k in folds.items():
        # print(obj, k)
        # ### the k-th fold
        # if k == 1:
        obj_path = os.path.join(data_path, obj)

        for file in os.listdir(obj_path):  ### search for manual_*.yaml files
            if 'manual_' in file:
                with open(os.path.join(obj_path, file)) as f:
                    ### load yaml file
                    p_l_yaml = yaml.safe_load(f)
                for p_l_pair in p_l_yaml['steps']:
                    ### store: [pointcloud_file, language_str]
                    p_file = os.path.join(obj_path, '_'.join(['pointcloud', obj, p_l_pair[0]]))
                    language = p_l_pair[1]
                    p_l_pair_path.append([p_file, language])

                    ### trajectory path
                    user_input_path = os.path.join(obj_path, 'user_input')
                    files_in_user_input = os.listdir(user_input_path)
                    for file_ in files_in_user_input:
                        if '.yaml' in file_:
                            with open(os.path.join(user_input_path, file_)) as f:
                                traj_yaml = yaml.safe_load(f)

                            for _, obj_traj_dict in traj_yaml['entries'].items():
                                if obj_traj_dict['part'] == p_l_pair[0]:
                                    traj_file = os.path.join(user_input_path, obj_traj_dict['uuid'])
                                    p_l_t_pair_path.append([p_file, language, traj_file])
    p_l_pair_path = np.unique(p_l_pair_path, axis=0) ### 249 pairs
    p_l_t_pair_path = np.unique(p_l_t_pair_path, axis=0) ### 1225 pairs

    return p_l_pair_path, p_l_t_pair_path


##### Pointclouds
def preprocess_pointcloud(pc_path):
    points = pd.read_csv(pc_path, sep=',', header=None).values[:, 0:3]
    ### fit into a 100x100x100 occupancy grid with side = 0.25 cm (0.0025 m)
    n_grid = 100
    shift = n_grid//2
    grid_100 = np.zeros([n_grid, n_grid, n_grid])

    for n, point in enumerate(points):
        i, j, k = int(point[0]//0.0025), int(point[1]//0.0025), int(point[2]//0.0025) # -50~50
        grid_100[i+shift, j+shift, k+shift] += 1.

    return grid_process(grid_100, n_average=10)

def grid_process(grid_100, n_average=10):
    ### normalize
    grid_100 = grid_100 / (grid_100.max()+1e-5)

    ### exponential distributed to neighboring cells: np.exp(-x)
    nonzero_index = np.argwhere(grid_100!=0) # numpy array
    n_nonzero = nonzero_index.shape[0]
    # print(nonzero_index[:, 0].max(), nonzero_index[:, 0].min(),
    #       nonzero_index[:, 1].max(), nonzero_index[:, 1].min(),
    #       nonzero_index[:, 2].max(), nonzero_index[:, 2].min())

    for n in range(n_nonzero):
        i, j, k = nonzero_index[n]
        # exponential distributed to 5 neighborning cells
        range_ = range(-1, -6, -1) + range(1, 6, 1)
        for si in range_:
            for sj in range_:
                for sk in range_:
                    try:
                        distance = np.sqrt(si**2 + sj**2 + sk**2)
                        grid_100[i+si, j+sj, k+sk] += grid_100[i, j, k]*np.exp(-distance)
                    except IndexError:
                        pass



    # nonzero_index = np.argwhere(grid_100!=0) # numpy array
    # print(nonzero_index[:, 0].max(), nonzero_index[:, 0].min(),
    #       nonzero_index[:, 1].max(), nonzero_index[:, 1].min(),
    #       nonzero_index[:, 2].max(), nonzero_index[:, 2].min())

    ### take average of cells to make new grid
    n_grid = grid_100.shape[0]//n_average # 100//10
    grid_new = np.zeros([n_grid, n_grid, n_grid])
    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(n_grid):
                grid_new[i, j, k] = grid_100[n_average*i:n_average*(i+1),
                                             n_average*j:n_average*(j+1),
                                             n_average*k:n_average*(k+1)].mean()
    # print(np.argwhere(grid_new!=0).shape)
    return grid_new.ravel()


##### Language
def preprocess_language(language, tokenizer):
    """Preprocess language instruction to bag_of_words vector
    :param language: language string
    :param tokenizer: keras tokenizer
    :return: bag_of_words with shape: (250,)
    """

    word_index = tokenizer.texts_to_sequences([language]) # word index start from 1
    onehot_array = to_categorical(word_index, num_classes=language_dim)
    bag_of_words = np.sum(onehot_array, axis=0)

    return bag_of_words

def language_tokenizer(context, num_words=250):
    """Tokenize language:
    :param context: all language instructions
    :return: tokinizer, reverse_tokenizer

    A total of 224 words in training set.
    Use num_words = 250 as default.
    """

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(context)
    reverse_tokenizer = dict(map(reversed, tokenizer.word_index.items()))

    return tokenizer, reverse_tokenizer


##### Trajectory
def preprocess_trajectory(traj_path, length=15):
    """Preprocess one trajectory to model input:
    :param traj_path:
    :param length:
    :return: traj_model_input with shape: (length*10,)

    traj_preprocessed_array with shape: (length, 10)
    each waypoint: (g_onehot_3, tx, ty, tz, rx, ry, rz, rw)
    """

    traj = Trajectory()
    traj.load_from_file(traj_path, print_log=False)
    traj_array = traj.get_length_normalized(length, as_array=True)

    # (g, list with len=7)
    g_state = ['open', 'close', 'hold']
    traj_preprocessed_array = np.zeros([length, 10])

    for i in range(length):
        g_index = g_state.index(traj_array[i][0]) # index = 0 or 1 or 2
        traj_preprocessed_array[i][g_index] = 1.
        traj_preprocessed_array[i][3:] = traj_array[i][1]
    # print(traj_preprocessed_array)

    return traj_preprocessed_array.ravel()

def get_traj_distance_matrix(traj_paths):
    """Compute trajectory distance matrix
    :param traj_paths: list of all trajectory files
    :return: distance_matrix
    """
    '''
    traj_num = len(traj_paths) # 1225
    distance_matrix = np.zeros([traj_num, traj_num])

    for i, path1 in enumerate(traj_paths):
        for j, path2 in enumerate(traj_paths):

            traj1 = Trajectory()
            traj1.load_from_file(path1, print_log=False)

            traj2 = Trajectory()
            traj2.load_from_file(path2, print_log=False)

            distance_matrix[i, j] = compute_dtw_mt(traj1, traj2)
    np.save('traj_distance_matrix.npy', distance_matrix)
    '''
    distance_matrix = np.load('traj_distance_matrix.npy')
    return distance_matrix


if __name__ == '__main__':
    ### main preprocess function for all modalities
    p_l_pair_path, p_l_t_pair_path = get_modalality_path()
    pair_num = p_l_t_pair_path.shape[0]

    ### compute trajectory distance matrix
    # traj_paths = p_l_t_pair_path[:, 2] # list of all trajectory files: 1225
    # distance_matrix = get_traj_distance_matrix(traj_paths)
    # print(np.where(distance_matrix[344,:]>50))
    # print(np.where(distance_matrix[344,:]<10))

    ### tokenizer
    context = p_l_pair_path[:, 1] # list of all language instructions: 248
    tokenizer, reverse_tokenizer = language_tokenizer(context, num_words=language_dim)

    ### prepare empty modal data
    pc_data = np.zeros([pair_num, pointcloud_dim])
    l_data = np.zeros([pair_num, language_dim]) # 0~1
    traj_data = np.zeros([pair_num, trajectory_dim]) # -1~1

    for i in range(pair_num):
        # print(i)
        ### preprocess pointcloud
        pc_path = p_l_t_pair_path[i, 0]
        pc_vector = preprocess_pointcloud(pc_path)

        ### preprocess language
        language = p_l_t_pair_path[i, 1]
        l_vector = preprocess_language(language, tokenizer)

        ### preprocess trajectory
        traj_path = p_l_t_pair_path[i, 2]
        traj_vector = preprocess_trajectory(traj_path)

        ### feed into modal data
        pc_data[i] = pc_vector
        l_data[i] = l_vector
        traj_data[i] = traj_vector

    np.save('Processed_data/pc_data_1225.npy', pc_data)
    np.save('Processed_data/l_data_1225.npy', l_data)
    np.save('Processed_data/traj_data_1225.npy', traj_data)