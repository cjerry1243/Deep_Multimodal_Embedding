import numpy as np
import math
from operator import add
from transformations import quaternion_multiply, quaternion_inverse, inverse_matrix


from parameters import *


def pose_to_two_arrays(pose):
    p = pose.position
    o = pose.orientation
    a = [[p.x, p.y, p.z], [o.x, o.y, o.z, o.w]]
    return a

def distance_euclidean(pt1, pt2):
    t = [pt2[0]-pt1[0], pt2[1] - pt1[1], pt2[2] - pt1[2]]
    return math.sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])

def displacement_vector(pt1, pt2):
    t = [pt2[0]-pt1[0], pt2[1] - pt1[1], pt2[2] - pt1[2]]
    return t

def distance_euclidean_from_zero(t):
    return math.sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])

def quat_to_matrix(q):
    qx, qy, qz, qw = q
    m = [[1.-2.*(qy**2) - 2*(qz**2), 2*qx*qy - 2.*qz*qw, 2.*qx*qz + 2.*qy*qw],
         [2.*qx*qy + 2.*qz*qw, 1. - 2.*(qx**2) - 2.*(qz**2), 2.*qy*qz - 2.*qx*qw],
         [2.*qx*qz - 2.*qy*qw, 2.*qy*qz + 2.*qx*qw, 1. - 2.*(qx**2) - 2.*(qy**2)]]

    return np.matrix(m, dtype=np.float64)



def angle_between_quaternions(pt1_quat, pt2_quat):
    if USE_ROS:
        ## if you use ROS, this will be slightly more accurate (and give same number as used in Sung et al. 2015)
        import PyKDL
        pt1_quat_kdl = PyKDL.Rotation.Quaternion(pt1_quat[0], pt1_quat[1], pt1_quat[2], pt1_quat[3])
        pt2_quat_kdl = PyKDL.Rotation.Quaternion(pt2_quat[0], pt2_quat[1], pt2_quat[2], pt2_quat[3])
        rotation_kdl = pt2_quat_kdl * pt1_quat_kdl.Inverse()
        theta_kdl, _ = rotation_kdl.GetRotAngle()   # return (float, vector)
        return theta_kdl

    q1 = quat_to_matrix(np.array(pt1_quat))
    q2 = quat_to_matrix(np.array(pt2_quat))
    rotation = q2 * inverse_matrix(q1)
    theta = np.arccos((np.trace(rotation) - 1.) / 2.)

    return theta
    


def weight(pt):
    (pt_pos, pt_quat) = pose_to_two_arrays(pt[1])
    pt_d = distance_euclidean_from_zero(pt_pos)
    #w = scipy.stats.norm(0, _SIGMA).pdf(pt_d)
    w = math.exp(-GAMMA * pt_d)
    return w

def distance(pt1, pt2):
    pt1_status = pt1[0]
    pt2_status = pt2[0]
    (pt1_pos, pt1_quat) = pose_to_two_arrays(pt1[1])
    (pt2_pos, pt2_quat) = pose_to_two_arrays(pt2[1])

    d_t = distance_euclidean(pt1_pos, pt2_pos)
    d_r = angle_between_quaternions(pt1_quat, pt2_quat)
    d_g = 0 if pt1_status == pt2_status else 1
    pt1_norm = distance_euclidean_from_zero(pt1_pos)

    d = ( ( ( (1.0 / ALPHA_T) * d_t) + ( (1.0 / ALPHA_R) * d_r) ) * (1 + BETA * d_g))

    #print "---- %f %f %f %f" % (d, d_t, d_r, pt1_norm)
    return (d, d_t, math.degrees(d_r), pt1_norm)




def euclidean_dist(e1, e2):
    e1 = np.array(e1)
    e2 = np.array(e2)

    dist = np.linalg.norm(e1 - e2)

    return dist

# each quaternion is in the form xyzw
def angular_distance_between_quats(quat1, quat2):
    cos_theta = (2 * (np.dot(quat1, quat2) ** 2)) - 1
    if np.allclose(cos_theta, 1.0):
        cos_theta = 1.0

    return np.degrees(np.arccos(cos_theta))


## Computes score
def dtw_score(seq1, seq2, cost_func):
    # Initialize table
    n = len(seq1)            # seq1 ===> n
    m = len(seq2)            # seq2 ===> m

    D = np.zeros((n,m))

    # Fills in the initial values of the table
    D[0][0] = cost_func(seq1[0], seq2[0])

    for i in range(1, m):
        D[0][i] = reduce(add, D[0], 0) + cost_func(seq1[0], seq2[i])

    D = np.transpose(D)

    for i in range(1, n):
        D[0][i] = reduce(add, D[0], 0) + cost_func(seq1[i], seq2[0])
    
    D = np.transpose(D)

    # Finished initializing first row and first column of DP table.
    # Dynamically fill in the rest of the table
    for i in range(1, n):
        # i indexes row
        # j indexes column
        for j in range(1, m):
            D[i][j] = min(D[i-1][j-1], D[i-1][j], D[i][j-1]) + cost_func(seq1[i], seq2[j])

    return D[n-1][m-1], D



def get_optimal_path(D):
    numb_rows, numb_cols = D.shape

    row_index = numb_rows - 1
    col_index = numb_cols - 1

    optimal_path = []
    optimal_path.append((row_index, col_index))

    while not (row_index, col_index) == (0,0):
        if row_index == 0:
            col_index -= 1
            optimal_path.append((row_index, col_index))
        elif col_index == 0:
            row_index -= 1
            optimal_path.append((row_index, col_index))
        else:
            from_diagonal = (row_index - 1, col_index - 1)
            from_above = (row_index - 1, col_index)
            from_side = (row_index, col_index - 1)

            argmax_helper = [from_diagonal, from_above, from_side]

            # negative sign added as we want argmin, not argmax
            optimal_index = np.argmax([-D[argmax_helper[0]],-D[argmax_helper[1]], -D[argmax_helper[2]]])

            optimal_origin = argmax_helper[optimal_index]

            row_index = optimal_origin[0]
            col_index = optimal_origin[1]

            optimal_path.append(optimal_origin)


    optimal_path.reverse()

    return optimal_path



def mhd_pointwise_distance(pt1, pt2):
    return weight(pt1) * weight(pt2) * distance(pt1, pt2)[0]


# Path (where each element indicates which point is compared to which point for the overall dtw comparison)
# for near dtdr, all edges that have at least one end that is within the threshold
# for far dtdr, all other edges will be considered
def path_score(traj_array1, traj_array2, path, cost_func):
    end_pair = path[-1]
    
    assert len(traj_array1) - 1 == end_pair[0]
    assert len(traj_array2) - 1 == end_pair[1]

    total_score = 0
    for entry in path:
        total_score += cost_func(traj_array1[entry[0]], traj_array2[entry[1]])

    return total_score




def compute_dtw_mt(traj1, traj2, cost_func=mhd_pointwise_distance):

    traj_array1 = np.array(traj1.trajectory_array)
    traj_array2 = np.array(traj2.trajectory_array)

    score, D = dtw_score(traj_array1, traj_array2, cost_func)
    
    path = get_optimal_path(D)

    p_score = path_score(traj_array1, traj_array2, path, cost_func)

    assert score == p_score

    return (score / float(len(path)))




