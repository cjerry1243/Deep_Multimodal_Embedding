import numpy as np
from keras.models import Model, Input
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from model import model
from load_data import pointcloud_dim, language_dim, trajectory_dim
from load_data import get_modalality_path, preprocess_trajectory


def autoencoder(tau_dim):
    tau_input = Input(shape=(tau_dim,), name='tau_input')
    tau_h1 = Dense(100, activation='relu', name='tau_h1')(tau_input)
    tau_h2 = Dense(100, activation='relu', name='tau_h2')(tau_h1)
    tau_h1_ = Dense(100, activation='relu', name='tau_h1_')(tau_h2)
    tau_rec = Dense(tau_dim, activation='tanh', name='tau_rec')(tau_h1_)
    return Model(tau_input, tau_rec)


def training_data_preparation():
    ### prepare training data
    p_l_pair_path, p_l_t_pair_path = get_modalality_path()
    traj_paths = p_l_t_pair_path[:, 2]  # list of all trajectory files: 1225
    traj_samples = traj_paths.shape[0]
    x_train = np.zeros([traj_samples, trajectory_dim]) # -1~1

    for i in range(traj_samples):
        ### preprocess trajectory
        traj_path = traj_paths[i]
        traj_vector = preprocess_trajectory(traj_path)

        ### feed into modal data
        x_train[i] = traj_vector

    return x_train


epochs = 200
batch_size = 50


if __name__ == '__main__':
    ### load model
    pretrain_tau_model = autoencoder(trajectory_dim)
    pretrain_tau_model.compile(optimizer='adam', loss='mse')
    pretrain_tau_model.summary()

    x_train = training_data_preparation()

    callbacks = ModelCheckpoint(filepath='Weights/pretrain_tau_weights.h5',
                                 monitor='loss',
                                 save_weights_only=True,
                                 save_best_only=True)

    history = pretrain_tau_model.fit(x_train, x_train,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   callbacks=[callbacks])
    np.save('History/pretrain_tau_history.npy', history.history)