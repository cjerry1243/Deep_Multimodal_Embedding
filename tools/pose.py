## Pose class
##  - behavior same as geometry_msgs.msg.Pose in ROS
##  - created to remove dependency on ROS


class Pose:
    def __init__(self):
        class DotAccess:
            def __init__(self, **kwds):
                self.__dict__.update(kwds)

        position = DotAccess(x = 0., y = 0., z = 0.)
        orientation = DotAccess(x = 0., y = 0., z = 0., w = 0.)
        self.__dict__.update(position = position, orientation = orientation)



