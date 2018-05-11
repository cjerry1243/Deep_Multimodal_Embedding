import pdb
import math
import numpy as np
from numpy import linalg, arange

from scipy.spatial import distance

from parameters import *

if USE_ROS:
    from geometry_msgs.msg import Pose
else:
    from pose import Pose 


_EPS = np.finfo(float).eps * 100

def parseWaypoint(wp1, wp2):
    if isinstance(wp1, Pose):
        oCoord = [wp1.position.x, wp1.position.y, wp1.position.z] 
        tCoord = [wp2.position.x, wp2.position.y, wp2.position.z] 

        oQuat = [wp1.orientation.x, wp1.orientation.y, wp1.orientation.z, wp1.orientation.w]
        tQuat = [wp2.orientation.x, wp2.orientation.y, wp2.orientation.z, wp2.orientation.w]
        
    else:
        oCoord = wp1[0:3]
        tCoord = wp2[0:3]

        oQuat = wp1[3:7]
        tQuat = wp2[3:7]

    return (oCoord, tCoord, oQuat, tQuat)

def slerp(qi, qn, t, eps):

    qm = None

    if linalg.norm(qi + qn) < linalg.norm(qi - qn):
        qn = -qn

    if t == 0:
        qm = qi
    elif t == 1:
        qm = qn
    else:
        C = np.dot(qi, qn)

        theta = np.arccos(C)

        if (1 - C) <= eps:
            qm = np.matlib.repmat(qn, 1, 1)
        elif (1 + C) <= eps:
            qm = np.matlib.repmat(qn, 1, 1)
        else:
            qm = (qi * ((np.sin((1 - t) * theta))/np.sin(theta))) + (qn * (np.sin(t * theta) / np.sin(theta)))

    return qm

def getIntermediateQuats(q1, q2, numberOfAngles, eps, end_inclusive=False):
    origin = np.array(q1)
    target = np.array(q2)

    quats = origin

    quatStep = 1 / float(numberOfAngles)
    nextQuat = 1 / float(numberOfAngles)
    for i in range(numberOfAngles - 1):
        quat = slerp(origin, target, nextQuat, eps)
        quats = np.vstack((quats, quat))
        nextQuat += quatStep

    if end_inclusive:
        quats = np.vstack((quats, target))

    return quats

def getIntermediatePoints(p1, p2, numberOfSteps, end_inclusive=False):
    origin = np.array(p1)
    target = np.array(p2)
    difference = target - origin
    step = difference / float(numberOfSteps)

    positions = origin
    nextPos = origin + step
    for i in range(numberOfSteps - 1):
        positions = np.vstack((positions, nextPos))
        nextPos += step

    if end_inclusive:
        positions = np.vstack((positions, target))

    return positions

def getIntermediateWPs(wp1, wp2, steps, quatEps=_EPS, end_inclusive=False):
    (originCoord, targetCoord, originQuat, targetQuat) = parseWaypoint(wp1, wp2)

    interPoints = getIntermediatePoints(originCoord, targetCoord, steps, end_inclusive=end_inclusive)
    interQuats = getIntermediateQuats(originQuat, targetQuat, steps, quatEps, end_inclusive=end_inclusive)

    if interPoints.ndim == 2 :      # If there is more than one waypoints to return
        wps = np.array([np.append(interPoints[i], interQuats[i]) for i in range(interPoints.shape[0])])
    else:                           # If there is only one waypoint to return
        wps = np.array([np.append(interPoints, interQuats)])

    return wps


# Given the translation velocity, rotational velocity, distance to cover, rotation to cover
# returns the number of intermediate steps to calculate. The number returned is always 
# the number of steps the 'slower' of the translation or rotation. For example, if 
# distance to cover can be done in 1 step and the rotation needs 10 steps, the method
# returns 10.

# Given the translation speed, rotational speed, euclidean distance to cover, rotation to cover
# returns the number of intermediate steps to calculate
# Input:
#   vel_tran -      translation speed. floating point  value
#   vel_rot -       rotational speed. floating point in RADIANS
#   euc_distance -  Euclidean distance to cover
#   angular_distance -  Angular distance to cover
def getNumberOfSteps(vel_tran, vel_rot, euc_distance, angular_distance):

    euclidean_steps = euc_distance / float(vel_tran)
    angular_steps = angular_distance / float(vel_rot)

    # Would have to return the math.ceil as the starting point must be included in the trajectory
    return int(math.ceil(max(euclidean_steps, angular_steps)))
    

# Main interpolation code that outputs the intermediate waypoints given start point (inclusive) and
# end point (inclusive).
# Input:
#    wp1, wp2 -    start and end points, respectively. They are 1-by-7 dimensional vectors 
#                first 3 are coordinates (xyz), next 4 are rotations (xyzw)
#    vel_tran -    A floating point value. velocity (simply unit / timestep) for translation. 
#    vel_rot_scalar  -  A floating point value in Euler angle in RADIANS
#    end_inclusive (optional)    Default=True. When True, the points returned will
#                                include the end point at the end of the list. 
#    quatEps (optional)            Default=0.00001.
# Output:
#    n-by-7 dimensional numpy ndarray consisting of intermediate points


def interpolate(wp1, wp2, vel_tran, vel_rot_scalar, end_inclusive=True, quatEps=_EPS):

    (oCoord, tCoord, oQuat, tQuat) = parseWaypoint(wp1, wp2)
    # oQuat and tQuat are in form xyzw

    # angular_distance from oQuat to tQuat in RADIANS
    angular_distance = np.arccos(2 * (np.dot(oQuat, tQuat) ** 2) - 1)
    # euc_distance EUCLIDEAN distance between two waypoints
    euc_distance = distance.euclidean(oCoord, tCoord)

    numberOfSteps = getNumberOfSteps(vel_tran, vel_rot_scalar, euc_distance, angular_distance)
    waypoints = getIntermediateWPs(wp1, wp2, numberOfSteps, quatEps, end_inclusive=end_inclusive)
    
    return waypoints

