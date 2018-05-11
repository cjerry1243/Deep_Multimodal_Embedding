import numpy as np
from keras.optimizers import Adam, Adadelta
from keras import backend as K
from keras.models import Input, Model
from load_data import pointcloud_dim, language_dim, trajectory_dim
from load_data import get_modalality_path
from model import model


def build_training_model(pretrain_p_model, pretrain_l_model):
    p_input = pretrain_p_model.input
    l_input = pretrain_l_model.input
    violate_l_input = Input(shape=(language_dim,))
    inputs = [p_input, l_input, violate_l_input]

    p_output = pretrain_p_model.output
    l_output = pretrain_l_model.output
    violate_l_output = pretrain_l_model(violate_l_input)
    outputs = [p_output, l_output, violate_l_output]

    training_model = Model(inputs=inputs, outputs=outputs)
    p_model = Model(inputs=p_input, outputs=p_output)
    l_model = Model(inputs=l_input, outputs=l_output)
    return training_model, p_model, l_model


def training_function(training_model):
    def sim(v1, v2):
        return K.sum(v1*v2, axis=-1)

    ### placeholder defined at p,l joint space: 125
    p_out = training_model.outputs[0]
    l_out = training_model.outputs[1]
    violate_l_out = training_model.outputs[2]
    margin = K.placeholder(shape=(None,))

    loss = K.mean(K.abs(margin + sim(p_out, violate_l_out) - sim(p_out, l_out)))
    adam = Adam(lr=1e-4)
    updates = adam.get_updates(params=training_model.trainable_weights, loss=loss)

    return K.function(inputs=[training_model.inputs[0], # p_input
                              training_model.inputs[1], # l_input
                              training_model.inputs[2], # violate_l_input
                              margin],                  # traj distance
                      outputs=[loss],
                      updates=updates)


def main():
    ### build training model and function
    pretrain_p_model, pretrain_l_model, _, _, _, _ = model(pointcloud_dim, language_dim, trajectory_dim)
    pretrain_p_model.load_weights('Weights/pretrain_p_weights.h5', by_name=True)
    pretrain_l_model.load_weights('Weights/pretrain_l_weights.h5', by_name=True)

    training_model, p_model, l_model = build_training_model(pretrain_p_model, pretrain_l_model)
    joint_training = training_function(training_model)
    training_model.summary()
    ### prepare pair data
    pc_data = np.load('Processed_data/pc_data_1225.npy')
    l_data = np.load('Processed_data/l_data_1225.npy')
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
            batch_pc = pc_data[batch_index, ...]
            batch_l = l_data[batch_index, ...]

            ### find the most violating language
            l_embedd = l_model.predict(l_data)
            batch_pc_embedd = p_model.predict(batch_pc)
            batch_violate_index = np.zeros([batchsize,]).astype(int)
            for b in range(batchsize):
                path_index = shuffle[b]
                relevant_traj_indices = np.argwhere(distance_matrix[path_index, :] < 10).squeeze()
                traj_index = np.random.choice(relevant_traj_indices)

                pc_vector = batch_pc_embedd[b]
                sim_score = np.sum(pc_vector* l_embedd, axis=-1)
                violate_score = sim_score + alpha*distance_matrix[traj_index]
                batch_violate_index[b] = violate_score.argmax()

            batch_violate_l = l_data[batch_violate_index, ...]
            batch_margin = distance_matrix[batch_index, batch_violate_index]

            ### Updates
            loss = joint_training([batch_pc, batch_l, batch_violate_l, batch_margin])[0]
            losses.append(loss)
            # print('iter:', it, 'loss:', loss)
        print('average loss:', np.mean(losses[-max_iter:]))

        if alpha > 0.2: alpha -= 0.01

    ### save model and loss history
    p_model.save_weights('Weights/joint_pl_p_weights.h5')
    l_model.save_weights('Weights/joint_pl_l_weights.h5')
    np.save('History/joint_pl_history.npy', losses)
    return


def test():
    p_l_pair_path, _ = get_modalality_path()
    context = p_l_pair_path[:, 1]

    p_model, l_model, _, _, _, _ = model(pointcloud_dim, language_dim, trajectory_dim)
    p_model.load_weights('Weights/joint_pl_p_weights.h5', by_name=True)
    l_model.load_weights('Weights/joint_pl_l_weights.h5', by_name=True)

    pc_data = np.load('Processed_data/pc_data_248.npy')
    l_data = np.load('Processed_data/l_data_248.npy')

    pc_embedd = p_model.predict(pc_data)
    l_embedd = l_model.predict(l_data)

    n_pair = 248
    sim_matrix = np.zeros([n_pair, n_pair])
    for i in range(n_pair):
        pc_vector = pc_embedd[i]
        sim_score = np.sum(pc_vector * l_embedd, axis=-1)
        sim_matrix[i] = sim_score
        print('context index:', i, 'magnitude:', np.linalg.norm(pc_vector))
        print('--------------', context[i])
        print('most relevant:', context[sim_score.argmax()])
        print('most irrelevant:', context[sim_score.argmin()])
    return


epochs = 500
batchsize = 20



if __name__ == '__main__':
    # main()
    test()