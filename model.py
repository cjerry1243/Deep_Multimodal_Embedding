from keras.models import Model, Input
from keras.layers import Dense, Add




def model(p_dim, l_dim, tau_dim):
    p_input = Input(shape=(p_dim,), name='p_input')
    l_input = Input(shape=(l_dim,), name='l_input')
    tau_input = Input(shape=(tau_dim,), name='tau_input')

    p_h1 = Dense(250, activation='relu', name='p_h1')(p_input)
    l_h1 = Dense(150, activation='relu', name='l_h1')(l_input)
    tau_h1 = Dense(100, activation='relu', name='tau_h1')(tau_input)


    p_h2 = Dense(125, activation='relu', name='p_h2')(p_h1)
    l_h2 = Dense(125, activation='relu', name='l_h2')(l_h1)
    tau_h2 = Dense(100, activation='relu', name='tau_h2')(tau_h1)

    p_l_joint = Add(name='p_l_joint')([p_h2, l_h2])

    p_l_h3 = Dense(25, activation='relu', name='p_l_h3')(p_l_joint)
    tau_h3 = Dense(25, activation='relu', name='tau_h3')(tau_h2)

    pretrain_p_model = Model(p_input, p_h2)
    pretrain_l_model = Model(l_input, l_h2)
    pretrain_tau_model = Model(tau_input, tau_h2)

    p_l_joint_model = Model([p_input, l_input], p_l_joint)

    p_l_embedding = Model([p_input, l_input], p_l_h3)
    tau_embedding = Model(tau_input, tau_h3)

    return pretrain_p_model, pretrain_l_model, pretrain_tau_model, p_l_joint_model, p_l_embedding, tau_embedding


if __name__ == '__main__':
    p_dim, l_dim, tau_dim = 300, 300, 300
    pretrain_p_model, pretrain_l_model, pretrain_tau_model, p_l_joint_model, p_l_embedding, tau_embedding = model(p_dim, l_dim, tau_dim)
    pretrain_p_model.summary()