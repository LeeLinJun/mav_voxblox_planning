#!/usr/bin/env python

import rospy
import mav_msgs
from os import path
import numpy as np
# import fcntl
import shutil

from geometry_msgs.msg import PoseStamped, Vector3

from mav_planning_msgs.srv import PlannerService
from visualization_msgs.msg import MarkerArray

def save_hash_file(start_goal_hash, save_path='data/hash.npy'):
    if path.exists(save_path):
        shutil.copyfile(save_path, save_path+'.bk')
    np.save(save_path, start_goal_hash)

def load_hash_file(save_path='data/hash.npy'):
    if path.exists(save_path):
        #reason for tolist: npy format will save to array(***, dtype=object)
        start_goal_hash = np.load(save_path).tolist()
    else:
        start_goal_hash = set([])
    return start_goal_hash


class StartGoalGenerator:
    def __init__(self, namespace="", planner_name='voxblox_sst_planner'):
        rospy.init_node('start_goal_generator')
        rospy.Subscriber("/{}/path".format(planner_name), MarkerArray, lambda marker_array: self.callback(marker_array))

        self.service_name = namespace + '/' + planner_name + '/plan'
        self.path = np.array([])

    def callback(self, marker_array):
        self.data = []
        for mi in marker_array.markers[0].points:
            self.data.append([mi.x, mi.y, mi.z])         
        # self.data = data.markers.points
    @staticmethod
    def state_to_posestamped(position, orientation):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()

        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]

        msg.pose.orientation.x = orientation[0]
        msg.pose.orientation.y = orientation[1]
        msg.pose.orientation.z = orientation[2]
        msg.pose.orientation.w = orientation[3]
        return msg

    def requestPlannerService(self, start, goal, start_goal_hash, start_ori=[0, 0, 0, 1], goal_ori=[0, 0, 0, 1]):
        rospy.wait_for_service(self.service_name)
        request_planner_service = rospy.ServiceProxy(self.service_name, PlannerService)
        start_pose = self.state_to_posestamped(start, start_ori)
        start_velocity = Vector3()
        goal_pose = self.state_to_posestamped(goal, goal_ori)
        goal_velocity = Vector3()
        bounding_box = Vector3()

        try:
            resp = request_planner_service(start_pose, start_velocity, goal_pose, goal_velocity, bounding_box)
            # print(resp)
            if resp.success:
                # print("success")
                rospy.sleep(0.1)
            #    print(self.data)
                np.save("data/traj_{}.npy".format(str(start_goal_hash)[1:-1].replace(", ", "_")),
                        self.data)
                np.save("data/start_goal_{}.npy".format(str(start_goal_hash)[1:-1].replace(", ", "_")),
                        np.concatenate((start, goal), axis=0))
                return True
        except:
            # print("Failed")
            return False


def random_start_goal(start_goal_hash=set(), max_xyz=[5, 25, 1.75], min_xyz=[-20, -5, 0], min_distance=5, max_distance=15):
    def pose_to_hash(pose, min_xyz, voxel_size=0.1):
        return list(((pose - min_xyz) / voxel_size).astype(int))
    while True:
        start = np.random.random(3) * (np.array(max_xyz) - np.array(min_xyz)) + np.array(min_xyz)
        goal = np.random.random(3) * (np.array(max_xyz) - np.array(min_xyz)) + np.array(min_xyz)
        start_goal_id = pose_to_hash(start, min_xyz) + pose_to_hash(goal, min_xyz)
        if np.linalg.norm(start-goal) > min_distance and np.linalg.norm(start-goal) < max_distance and tuple(start_goal_id) not in start_goal_hash:
            # print(np.linalg.norm(start-goal))
            break
    return start, goal, start_goal_id

def main(n_traj=1000):
    node = StartGoalGenerator()
    start_goal_hash = load_hash_file()
    while len(start_goal_hash) < n_traj:
        start_goal_hash = load_hash_file()
        start, goal, start_goal_id = random_start_goal(start_goal_hash) # s_g_id is a list
        if node.requestPlannerService(start, goal, start_goal_id):
            start_goal_hash.add(tuple(start_goal_id))
            save_hash_file(start_goal_hash)
            print("Current #traj: {}".format(len(start_goal_hash)))


    # rospy.spin()

if __name__ == '__main__':
    main()
