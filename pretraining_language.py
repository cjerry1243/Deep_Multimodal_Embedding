import numpy as np
from keras.models import Model, Input
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from model import model
from load_data import pointcloud_dim, language_dim, trajectory_dim
from load_data import get_modalality_path, language_tokenizer, preprocess_language


def autoencoder(l_dim):
    l_input = Input(shape=(l_dim,), name='l_input')
    l_h1 = Dense(150, activation='relu', name='l_h1')(l_input)
    l_h2 = Dense(125, activation='relu', name='l_h2')(l_h1)
    l_h1_ = Dense(150, activation='relu', name='l_h1_')(l_h2)
    l_rec = Dense(l_dim, activation='sigmoid', name='l_rec')(l_h1_)
    return Model(l_input, l_rec)


def training_data_preparation():
    ### prepare training data
    p_l_pair_path, p_l_t_pair_path = get_modalality_path()
    context = p_l_pair_path[:, 1]  # list of all language instructions: 248 sentences
    tokenizer, reverse_tokenizer = language_tokenizer(context, num_words=language_dim)

    l_samples = context.shape[0]
    x_train = np.zeros([l_samples, language_dim])

    for i in range(l_samples):
        language = context[i]
        l_vector = preprocess_language(language, tokenizer)
        x_train[i] = l_vector

    # np.save('Processed_data/l_data_248.npy', x_train)
    # exit()
    return x_train


epochs = 200
batch_size = 20


if __name__ == '__main__':
    ### load model
    pretrain_l_model = autoencoder(language_dim)
    pretrain_l_model.compile(optimizer='adam', loss='mse')
    pretrain_l_model.summary()

    x_train = training_data_preparation()

    callbacks = ModelCheckpoint(filepath='Weights/pretrain_l_weights.h5',
                                 monitor='loss',
                                 save_weights_only=True,
                                 save_best_only=True)

    history = pretrain_l_model.fit(x_train, x_train,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   callbacks=[callbacks])
    np.save('History/pretrain_l_history.npy', history.history)


