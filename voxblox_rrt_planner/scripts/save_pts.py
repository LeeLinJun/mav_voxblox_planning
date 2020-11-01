#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Header
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import PointCloud2
from voxblox_msgs.msg import Layer, Mesh
from geometry_msgs.msg import PoseStamped, Vector3, Point
import numpy as np
#from nav_msgs.msg import Odometry
def callback_mesh(data):
    print("Here goes Mesh!")
    data = data.markers
    # print(data)
    map_array = []
    for marker in data:
        points = marker.points
        # print(points)
        for data in points:
            # print(data.x,data.y,data.z)
            map_array.append((data.x,data.y,data.z))
    map_array = np.array(map_array)
    np.save('data_occupied_nodes.npy', map_array)
    print(map_array.shape)

if __name__ == '__main__':
    rospy.init_node('mesh', anonymous=True)
    print("mesh node started")
    rospy.Subscriber("/voxblox_rrt_planner/occupied_nodes", MarkerArray, callback_mesh)
    rospy.spin()
