import numpy as np
from keras.optimizers import Adam, Adadelta
from keras import backend as K
from keras.models import Input, Model
from load_data import pointcloud_dim, language_dim, trajectory_dim
from load_data import get_modalality_path
from model import model


def build_training_model(traj_model):
    target_tau_input = traj_model.input
    relevant_tau_input = Input(shape=(trajectory_dim,))
    violate_tau_input = Input(shape=(trajectory_dim,))

    target_tau_output = traj_model.output
    relevant_tau_output = traj_model(relevant_tau_input)
    violate_tau_output = traj_model(violate_tau_input)

    inputs = [target_tau_input, relevant_tau_input, violate_tau_input]
    outputs = [target_tau_output, relevant_tau_output, violate_tau_output]

    training_model = Model(inputs=inputs, outputs=outputs)
    tau_model = Model(inputs=target_tau_input, outputs=target_tau_output)
    return training_model, tau_model


def training_function(training_model):
    def sim(v1, v2):
        norm_v1 = K.l2_normalize(v1, axis=-1)
        norm_v2 = K.l2_normalize(v2, axis=-1)
        return K.sum(norm_v1*norm_v2, axis=-1)

    target_out = training_model.outputs[0]
    relevant_out = training_model.outputs[1]
    violate_out = training_model.outputs[2]
    margin = K.placeholder(shape=(None,))

    loss = K.abs(margin + sim(target_out, violate_out) - sim(target_out, relevant_out))
    # adam = Adadelta(lr=1e-4)
    adam = Adam(lr=5e-4)
    updates = adam.get_updates(params=training_model.trainable_weights, loss=loss)

    return K.function(inputs=[training_model.inputs[0], # target_input
                              training_model.inputs[1], # relevant_input
                              training_model.inputs[2], # violate_input
                              margin],                  # traj distance
                      outputs=[loss],
                      updates=updates)


def main():
    ### build training model and function
    _, _, pretrain_tau_model, _, _, _ = model(pointcloud_dim, language_dim, trajectory_dim)
    pretrain_tau_model.load_weights('Weights/pretrain_tau_weights.h5', by_name=True)

    training_model, tau_model = build_training_model(pretrain_tau_model)
    joint_training = training_function(training_model)
    training_model.summary()
    ### prepare pair data
    traj_data = np.load('Processed_data/traj_data_1225.npy')
    distance_matrix = np.load('traj_distance_matrix.npy')

    n_pair = 1225
    max_iter = n_pair//batchsize # 1225//20

    alpha = .2
    losses = []
    ### training starts
    for epoch in range(epochs):
        print('epoch:', epoch, '-----------------------------')
        shuffle = np.random.permutation(n_pair)
        for it in range(max_iter):
            batch_index = shuffle[batchsize * it: batchsize * (it + 1)]
            batch_target_traj = traj_data[batch_index, ...]

            ### find the most violating trajectory
            traj_embedd = tau_model.predict(traj_data)
            batch_traj_embedd = tau_model.predict(batch_target_traj)

            batch_relevant_index = np.zeros([batchsize,]).astype(int)
            batch_violate_index = np.zeros([batchsize,]).astype(int)

            for b in range(batchsize):
                path_index = shuffle[b]
                relevant_traj_indices = np.argwhere(distance_matrix[path_index, :] < 10).squeeze()
                traj_index = np.random.choice(relevant_traj_indices)

                # traj_vector = batch_traj_embedd[b]
                # sim_score = np.sum(traj_vector* traj_embedd, axis=-1)\
                #             /(np.linalg.norm(traj_vector)+1e-4)/(np.linalg.norm(traj_embedd)+1e-4)
                # violate_score = sim_score + alpha*distance_matrix[traj_index]/100
                #
                irrelevant_traj_indices = np.argwhere(distance_matrix[traj_index, :] > 30).squeeze()
                irr_traj_index = np.random.choice(irrelevant_traj_indices)

                batch_relevant_index[b] = traj_index
                batch_violate_index[b] = irr_traj_index
                # batch_violate_index[b] = violate_score.argmax()

            batch_relevant_traj = traj_data[batch_relevant_index, ...]
            batch_violate_traj = traj_data[batch_violate_index, ...]
            batch_margin = distance_matrix[batch_relevant_index, batch_violate_index]/100

            ### Updates
            loss = joint_training([batch_target_traj, batch_relevant_traj, batch_violate_traj, batch_margin])[0]
            losses.append(loss)
            # print('iter:', it, 'loss:', loss)
        print('average loss:', np.mean(losses[-max_iter:]))
        # if alpha > 0.2: alpha -= 0.001

    ### save model and loss history
    tau_model.save_weights('Weights/joint_tau_weights.h5')
    np.save('History/joint_tau_history.npy', losses)
    return


def test():
    _, p_l_t_pair_path = get_modalality_path()
    context = p_l_t_pair_path[:, 1]

    _, _, traj_model, _, _, _ = model(pointcloud_dim, language_dim, trajectory_dim)
    traj_model.load_weights('Weights/joint_tau_weights.h5', by_name=True)

    traj_data = np.load('Processed_data/traj_data_1225.npy')

    traj_embedd = traj_model.predict(traj_data)
    distance_matrix = np.load('traj_distance_matrix.npy')

    n_pair = 1225
    sim_matrix = np.zeros([n_pair, n_pair])
    for i in range(n_pair):
        traj_vector = traj_embedd[i]
        sim_score = np.sum(traj_vector * traj_embedd, axis=-1)/np.linalg.norm(traj_vector)/np.linalg.norm(traj_embedd, axis=-1)
        sim_matrix[i] = sim_score
        print('traj index:', i, context[i], 'magnitude:', np.linalg.norm(traj_vector))
        print('most relevant:', sim_score.argmax(), context[sim_score.argmax()])
        print('most irrelevant:', sim_score.argmin(), context[sim_score.argmin()], 'original:', context[distance_matrix[i].argmax()])
        print('-------------------------------------------------------------------------')
    return


epochs = 500
batchsize = 20


if __name__ == '__main__':
    main()
    test()