import numpy as np
from keras.models import Model, Input
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from model import model
from load_data import pointcloud_dim, language_dim, trajectory_dim
from load_data import get_modalality_path, preprocess_pointcloud


def autoencoder(p_dim):
    p_input = Input(shape=(p_dim,), name='p_input')
    p_h1 = Dense(250, activation='relu', name='p_h1')(p_input)
    p_h2 = Dense(125, activation='relu', name='p_h2')(p_h1)
    p_h1_ = Dense(250, activation='relu', name='p_h1_')(p_h2)
    p_rec = Dense(p_dim, activation='relu', name='p_rec')(p_h1_)
    return Model(p_input, p_rec)


def training_data_preparation():
    ### prepare training data
    p_l_pair_path, p_l_t_pair_path = get_modalality_path()
    p_paths = p_l_pair_path[:, 0] # list of all pointcloud paths: 248
    p_samples = p_paths.shape[0]

    x_train = np.zeros([p_samples, pointcloud_dim])

    for i in range(p_samples):
        # print(i)
        pc_path = p_paths[i]
        pc_vector = preprocess_pointcloud(pc_path)
        x_train[i] = pc_vector

    # np.save('Processed_data/pc_data_248.npy', x_train)
    # exit()
    return x_train


epochs = 200
batch_size = 20


if __name__ == '__main__':
    ### load model
    pretrain_p_model = autoencoder(pointcloud_dim)
    pretrain_p_model.compile(optimizer='adam', loss='mse')
    pretrain_p_model.summary()

    x_train = training_data_preparation()

    callbacks = ModelCheckpoint(filepath='Weights/pretrain_p_weights.h5',
                                 monitor='loss',
                                 save_weights_only=True,
                                 save_best_only=True)

    history = pretrain_p_model.fit(x_train, x_train,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   callbacks=[callbacks])
    np.save('History/pretrain_p_history.npy', history.history)


