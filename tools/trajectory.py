import rospy
import copy
import os, sys

import random
import math

import logging as log

from wp_interpolation import *
from transformations import quaternion_matrix, quaternion_multiply, quaternion_inverse, quaternion_from_matrix, inverse_matrix

import ipdb

def array_to_pose(a):
    pose = Pose()
    p = pose.position
    o = pose.orientation
    [p.x, p.y, p.z, o.x, o.y, o.z, o.w] = a
    return pose

def pose_to_arrays(p):
    pos = p.position
    ori = p.orientation
    return [pos.x, pos.y, pos.z], [ori.x, ori.y, ori.z, ori.w]

def pose_to_array(p):
    pos = p.position
    ori = p.orientation
    return [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]

def pos_array_to_column_vector(pa):
    return np.vstack((np.asarray(pa)[:,None], 1))

class Trajectory:

    ACTION_OPEN = "open"
    ACTION_CLOSE = "close"
    ACTION_HOLD = "hold"

    def __init__(self, inter_meters = 0.01, inter_ang = 5, in_part_frame = True):
        self.in_part_frame = in_part_frame

        self.waypoint_array = []                    ## waypoints
        self.waypoint_gripper_action = {}
        self.trajectory_array = []                  ## interpolated
        

        if inter_meters is not None:
            self.INTERPOLATION_TRAN_METERS = inter_meters
        if inter_ang is not None:
            self.INTERPOLATION_TRAN_ANG_DEGREE = inter_ang


    def get_length(self):
        return len(self.waypoint_array) + self.get_action_length()

    def get_interpolated_length(self):
        return len(self.trajectory_array)

    def get_action_length(self):
        return len(self.waypoint_gripper_action)

    def get_full_trajectory(self):
        return self.trajectory_array


    ## transform this trajectory from object part frame to ground frame
    def transform_to_ground(self, data_info, part_id):
        assert self.in_part_frame

        USE_TF_LISTENER = True  ## TODO FIXME FIXME FIXME FIXME FIXME FIXME !!!!!

        part_frame = data_info.get_stored_part_frame(part_id)

        if USE_TF_LISTENER:
            print("~~~~~~~~~~~~~~~~~~~~~~~~")
            print(part_frame)
            import tf
            listener = tf.TransformListener()
            while True:
                try:
                    part_frame2 = listener.lookupTransform('/world', '/object_frame', rospy.Time(0))
                    print(part_frame2)
                    break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
            print("~~~~~~~~~~~~~~~~~~~~~~~~")

            part_frame = part_frame2[0] + part_frame2[1]

        trans = part_frame[:3]
        rot_quat = part_frame[3:]

        rot_quat_mat = quaternion_matrix(rot_quat)

        # rotate first and translate
        new_waypoint_array = []
        for i in range(len(self.waypoint_array)):
            waypoint = self.waypoint_array[i]
            way_pos, way_ori = pose_to_arrays(waypoint)

            way_pos_rotated = np.dot(rot_quat_mat, pos_array_to_column_vector(way_pos))
            way_ori_rotated = quaternion_multiply(rot_quat, way_ori)

            # translate
            way_pos_translated = np.transpose(way_pos_rotated[:-1])[0] + trans

            assert way_ori_rotated.ndim == 1 and way_pos_translated.ndim == 1
            assert way_ori_rotated.shape[0] == 4 and way_pos_translated.shape[0] == 3

            new_waypoint = array_to_pose(np.hstack((way_pos_translated, way_ori_rotated)))
            new_waypoint_array.append(new_waypoint)

        self.waypoint_array = new_waypoint_array
        self.update_interpolation()
        self.in_part_frame = False

        return part_frame

    ## transform from ground frame to object part frame
    def transform_to_object_part_frame(self, part_frame):
        assert not self.in_part_frame

        trans = part_frame[:3]
        rot_quat = part_frame[3:]

        rot_quat_mat = quaternion_matrix(rot_quat)

        rot_quat_inv = quaternion_inverse(rot_quat)
        rot_quat_inv_mat = quaternion_matrix(rot_quat_inv)

        # translate and rotate
        new_waypoint_array = []
        for i in range(len(self.waypoint_array)):
            waypoint = self.waypoint_array[i]
            way_pos, way_ori = pose_to_arrays(waypoint)

            way_pos_trans = np.asarray(way_pos) - np.asarray(trans)

            way_pos_rotated = np.dot(rot_quat_inv_mat, pos_array_to_column_vector(way_pos_trans))
            way_ori_rotated = quaternion_multiply(rot_quat_inv, way_ori)
            
            way_pos_rotated = np.transpose(way_pos_rotated)[0][:-1]
            assert way_ori_rotated.ndim == 1 and way_pos_rotated.ndim == 1
            assert way_ori_rotated.shape[0] == 4 and way_pos_rotated.shape[0] == 3

            new_waypoint = array_to_pose(np.hstack((way_pos_rotated, way_ori_rotated)))
            new_waypoint_array.append(new_waypoint)
 
        self.waypoint_array = new_waypoint_array
        self.update_interpolation()
        self.in_part_frame = False


    def add_waypoint(self, waypoint):
        if isinstance(waypoint, Pose):
            self.waypoint_array.append(waypoint)
        else:
            print("Wrong type of waypoint %s" % waypoint)


    def add_gripper_action(self, action): 
        self.waypoint_gripper_action[len(self.waypoint_array)] = action

    def add_gripper_action_open(self):
        self.add_gripper_action(self.ACTION_OPEN)

    def add_gripper_action_close(self):
        self.add_gripper_action(self.ACTION_CLOSE)

    def add_gripper_action_hold(self, options):
        self.add_gripper_action(self.ACTION_HOLD) #, options) 

    def print_gripper_action(self):
        print (self.waypoint_gripper_action)

    def print_trajectory(self):
        print ("-------------------------------------")
        self.print_gripper_action()
        for i in range(len(self.waypoint_array) + 1):
            if i in self.waypoint_gripper_action:
                print ("-- %s ---!!" % self.waypoint_gripper_action[i].upper())
            if i == len(self.waypoint_array):
                break
            t = self.waypoint_array[i]
            print_pose(t)
        print ("-------------------------------------")


    def load_from_file(self, filename, print_log = True):
        if not os.path.isfile(filename):
            log.error("Trajectory file %s does not exist!!" % filename)
            sys.exit(-1)

        f = open(filename, 'r')
        lines = f.readlines()
        f.close()
        if print_log:
            print ("%s read" % filename)

        for l in lines:
            l = l.strip()
            if l == '"open"':
                self.add_gripper_action_open()
            elif l == '"close"':
                self.add_gripper_action_close()
            elif l.startswith('"hold"'):
                options = l.split()[1:]
                self.add_gripper_action_hold(options)
            #elif l == '"close thick"':
            #    self.add_gripper_action("close_thick")
            else:
                self.add_waypoint(array_to_pose(eval(l)))

        self.update_interpolation(print_log = print_log)


    def save_to_file(self, filename):
        if os.path.exists(filename):
            print ("File already exists!!!!!! Not going to save!!")
            return

        if len(filename) > 0:
            print ("writing to file %s" % filename)
            print ("writing traj of length %d" % len(self.waypoint_array))
            f = open(filename, 'w')
            #f.write("%s\n" % gripper_action)

            if 0 in self.waypoint_gripper_action: 
                if 0 in self.waypoint_gripper_action_options:
                    ga_opt = self.waypoint_gripper_action_options[0]
                    assert len(ga_opt) == 2
                    f.write("\"%s %s %s\"\n" % (self.waypoint_gripper_action[0], ga_opt[0], ga_opt[1]))
                else:
                    f.write("\"%s\"\n" % self.waypoint_gripper_action[0])

            for i in range(len(self.waypoint_array)):
                t = self.waypoint_array[i]
                t_str = str(pose_to_array(t))
                f.write("%s\n" % t_str)

                if (i+1) in self.waypoint_gripper_action: 
                    if (i+1) in self.waypoint_gripper_action_options: 
                        ga_opt = self.waypoint_gripper_action_options[i+1]
                        assert len(ga_opt) == 2
                        f.write("\"%s %s %s\"\n" % (self.waypoint_gripper_action[i+1], ga_opt[0], ga_opt[1]))
                    else:
                        f.write("\"%s\"\n" % self.waypoint_gripper_action[i+1])


            f.close()
            print ("File saved!")


    def get_length_normalized(self, length, as_array = False):
        interpolated = self.get_full_trajectory()
        
        inter_len = len(interpolated)
        avg_over_len = int(inter_len) / int(length)

        # if already shorter than needed length
        if inter_len <= length:
            normalized = []
            for i in range(len(interpolated)):
                (gripper_status, pt_loc) = interpolated[i]
                if as_array:
                    pt_loc_array = pose_to_array(pt_loc)
                else:
                    pt_loc_array = pt_loc

                if i == 0:
                    for j in range(length - inter_len):
                        normalized.append((gripper_status, pt_loc_array))

                normalized.append((gripper_status, pt_loc_array))
            assert (length == len(normalized))
            return normalized

        # Interate backwards
        normalized = []
        p = inter_len - 1
        for i in range(length):
            current_gripper = None
            middle_pt = None
            middle_pt_status = None
            last_change = None
            last_change_status = None
            gripper_change_count = 0

            # find mid-point or gripper change pt
            for j in range(avg_over_len): ## TODO maybe +1 ?
                pt = interpolated[p]
                (gripper_status, pt_loc) = pt
                if as_array:
                    pt_loc_array = pose_to_array(pt_loc)
                else:
                    pt_loc_array = pt_loc

                if j == 0:
                    current_gripper = gripper_status
                
                if j == (avg_over_len / 2):
                    middle_pt = pt_loc_array
                    middle_pt_status = gripper_status
                
                if current_gripper != gripper_status:
                    gripper_change_count += 1
                    current_gripper = gripper_status
                    last_change = pt_loc_array
                    last_change_status = gripper_status

                p -= 1

            # add 
            if gripper_change_count > 1:
                for _ in range(20):
                    log.error("gripper change: %d  !!!" % gripper_change_count)
                #sys.exit(-1) # TODO 

            if last_change:
                normalized.append((last_change_status, last_change))
            elif middle_pt:
                normalized.append((middle_pt_status, middle_pt))
            else:
                print ("empty pt while normalizing traj!!")
                print (normalized)
                sys.exit(-1)

        normalized = list(reversed(normalized))
        assert (length == len(normalized))
        return normalized

    def to_traj_with_gripper_status_appended(self):
        current_gripper_status = "unknown"
        traj = []
        for i in range(len(self.waypoint_array) + 1):
            if i in self.waypoint_gripper_action:
                current_gripper_status = self.waypoint_gripper_action[i]
                if i is not 0:
                    traj.append([current_gripper_status, self.waypoint_array[i-1]])
            if i == len(self.waypoint_array):
                break

            traj.append([current_gripper_status, self.waypoint_array[i]])

        return traj

    def update_interpolation(self, print_log = True): 
        traj = self.to_traj_with_gripper_status_appended()
        interpolated_traj = []
        for i in range(len(traj)-1):
            g_status_c = traj[i][0]
            g_status_n = traj[i+1][0]
            if (g_status_c != g_status_n):
                interpolated_traj.append(traj[i])
            else:
                wp_c = traj[i][1]
                wp_n = traj[i+1][1]

                interpolated = interpolate(wp_c, wp_n, self.INTERPOLATION_TRAN_METERS, 
                        math.radians(self.INTERPOLATION_TRAN_ANG_DEGREE), end_inclusive=False)   
                for j in range(len(interpolated)):
                    interpolated_traj.append([g_status_c, array_to_pose(interpolated[j])])
        interpolated_traj.append(traj[-1])

        self.trajectory_array = interpolated_traj
        if print_log:
            print ("number of waypoints and actions - %d" % self.get_length())
            print ("interpolated trajectory length  - %d" % self.get_interpolated_length())
        


