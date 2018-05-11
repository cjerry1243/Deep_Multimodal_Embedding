import numpy as np
from keras.optimizers import Adam, Adadelta
from keras import backend as K
from keras.models import Input, Model
from load_data import pointcloud_dim, language_dim, trajectory_dim
from load_data import get_modalality_path
from model import model


def build_training_model(pl_embedding_model, tau_embedding_model):
    target_p_input, target_l_input = pl_embedding_model.inputs
    relevant_tau_input = tau_embedding_model.input
    violate_tau_input = Input(shape=(trajectory_dim,))

    target_pl_output = pl_embedding_model.output
    relevant_tau_output = tau_embedding_model(relevant_tau_input)
    violate_tau_output = tau_embedding_model(violate_tau_input)

    inputs = [target_p_input, target_l_input, relevant_tau_input, violate_tau_input]
    outputs = [target_pl_output, relevant_tau_output, violate_tau_output]

    training_model = Model(inputs=inputs,
                           outputs=outputs)
    pl_model = Model(inputs=[target_p_input, target_l_input],
                     outputs=target_pl_output)
    tau_model = Model(inputs=relevant_tau_input,
                      outputs=relevant_tau_output)
    return training_model, pl_model, tau_model


def training_function(training_model):
    def sim(v1, v2):
        return K.sum(v1*v2, axis=-1)

    target_out = training_model.outputs[0]
    relevant_out = training_model.outputs[1]
    violate_out = training_model.outputs[2]
    margin = K.placeholder(shape=(None,))

    loss = K.abs(margin + sim(target_out, violate_out) - sim(target_out, relevant_out))
    # adam = Adadelta(lr=0.01)
    adam = Adam(lr=1e-4)
    updates = adam.get_updates(params=training_model.trainable_weights, loss=loss)

    return K.function(inputs=[training_model.inputs[0], # target_inputs: p_input
                              training_model.inputs[1], #                l_input
                              training_model.inputs[2], # relevant_input
                              training_model.inputs[3], # violate_input
                              margin],                  # traj distance
                      outputs=[loss],
                      updates=updates)


def main():
    ### build training model and function
    _, _, _, _, p_l_embedding, tau_embedding = model(pointcloud_dim, language_dim, trajectory_dim)
    p_l_embedding.load_weights('Weights/joint_pl_p_weights.h5', by_name=True)
    p_l_embedding.load_weights('Weights/joint_pl_l_weights.h5', by_name=True)
    tau_embedding.load_weights('Weights/joint_tau_weights.h5', by_name=True)


    training_model, pl_model, tau_model = build_training_model(p_l_embedding, tau_embedding)

    joint_training = training_function(training_model)
    training_model.summary()

    ### prepare pair data
    pc_data = np.load('Processed_data/pc_data_1225.npy')
    l_data = np.load('Processed_data/l_data_1225.npy')
    traj_data = np.load('Processed_data/traj_data_1225.npy')
    distance_matrix = np.load('traj_distance_matrix.npy')

    n_pair = 1225
    max_iter = n_pair//batchsize # 1225//20

    alpha = 0.5
    losses = []
    ### training starts
    for epoch in range(epochs):
        print('epoch:', epoch, '-----------------------------')
        shuffle = np.random.permutation(n_pair)
        for it in range(max_iter):
            batch_index = shuffle[batchsize * it: batchsize * (it + 1)]
            batch_target_pc = pc_data[batch_index, ...]
            batch_target_l = l_data[batch_index, ...]

            ### find the most violating trajectory
            traj_embedd = tau_model.predict(traj_data)
            batch_pl_embedd = pl_model.predict([batch_target_pc, batch_target_l])

            batch_relevant_index = np.zeros([batchsize,]).astype(int)
            batch_violate_index = np.zeros([batchsize,]).astype(int)

            for b in range(batchsize):
                path_index = shuffle[b]
                relevant_traj_indices = np.argwhere(distance_matrix[path_index, :] < 8).squeeze()
                traj_index = np.random.choice(relevant_traj_indices)

                pl_vector = batch_pl_embedd[b]
                sim_score = np.sum(pl_vector* traj_embedd, axis=-1)
                violate_score = sim_score + alpha*distance_matrix[traj_index]

                batch_relevant_index[b] = traj_index
                batch_violate_index[b] = violate_score.argmax()

            batch_relevant_traj = traj_data[batch_relevant_index, ...]
            batch_violate_traj = traj_data[batch_violate_index, ...]
            batch_margin = distance_matrix[batch_relevant_index, batch_violate_index]

            ### Updates
            loss = joint_training([batch_target_pc, batch_target_l, batch_relevant_traj, batch_violate_traj, batch_margin])[0]
            losses.append(loss)
            # print('iter:', it, 'loss:', loss)
        print('average loss:', np.mean(losses[-max_iter:]))
        if alpha > 0.2: alpha -= 0.01

    ### save model and loss history
    pl_model.save_weights('Weights/joint_pltau_pl_weights.h5')
    tau_model.save_weights('Weights/joint_pltau_tau_weights.h5')
    np.save('History/joint_pltau_history.npy', losses)
    return


def test():
    _, p_l_t_pair_path = get_modalality_path()
    context = p_l_t_pair_path[:, 1]

    _, _, _, _, p_l_embedding, traj_model = model(pointcloud_dim, language_dim, trajectory_dim)
    p_l_embedding.load_weights('Weights/joint_pltau_pl_weights.h5', by_name=True)
    traj_model.load_weights('Weights/joint_pltau_tau_weights.h5', by_name=True)

    pc_data = np.load('Processed_data/pc_data_1225.npy')
    l_data = np.load('Processed_data/l_data_1225.npy')
    traj_data = np.load('Processed_data/traj_data_1225.npy')
    distance_matrix = np.load('traj_distance_matrix.npy')

    pl_embedd = p_l_embedding.predict([pc_data, l_data])
    traj_embedd = traj_model.predict(traj_data)

    n_pair = 1225
    sim_matrix = np.zeros([n_pair, n_pair])
    for i in range(n_pair):
        pl_vector = pl_embedd[i]
        sim_score = np.sum(pl_vector * traj_embedd, axis=-1)/np.linalg.norm(pl_vector)/np.linalg.norm(traj_embedd, axis=-1)
        sim_matrix[i] = sim_score
        print('traj index:', i, context[i])
        print('most relevant:', context[sim_score.argmax()])
        print('most irrelevant:', context[sim_score.argmin()], 'original:', context[distance_matrix[i].argmax()])
        print('-------------------------------------------------------------------------')
    return


epochs = 500
batchsize = 20



if __name__ == '__main__':
    # main()
    test()