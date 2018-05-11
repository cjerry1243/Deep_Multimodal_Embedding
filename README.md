# Deep_Multimodal_Embedding
Reproduction of 'Deep Multimodal Embedding: Manipulating Novel Objects with Point-clouds, Language and Trajectories'

(http://robobarista.cs.cornell.edu/assets/papers/sung_icra2017_dme.pdf)

## Requirements:
Use python2
1. rospy
2. keras
3. tensorflow

## To run:
### Prepare Robobarista dataset
(See: http://robobarista.cs.cornell.edu/dataset)

Modify data_path in line 12 in load_data.py.

### Script discription:
#### Autoencoder pretraining:
##### pretraining_pointcloud.py: 
pretrain pointcloud to h2 layer
##### pretraining_language.py: 
pretrain language to h2 layer
##### pretraining_trajectory.py: 
pretrain trajectory to h2 layer
#### Joint training:
##### joint_training_p_l.py: 
joint pointcloud, language
##### joint_training_traj.py: 
joint trajectory
##### joint_training_p_l_tau.py: 
joint (pointcloud, language), trajectory
#### Others:

