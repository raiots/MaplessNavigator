import rospy
import math
import copy
import random
import numpy as np
from shapely.geometry import Point
from simple_laserscan.msg import SimpleScan
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState

class GazeboEnvironment:
    def __init__(self,
                 laser_scan_half_num=9,
                 laser_scan_min_dis=0.35,
                 laser_scan_scale=1.0,
                 scan_dir_num=36,
                 goal_dis_min_dis=0.5,
                 goal_dis_scale=1.0,
                 obs_near_th=0.35,
                 goal_near_th=0.5,
                 goal_reward=10,
                 obs_reward=-5,
                 goal_dis_amp=5,
                 step_time=0.1):
        self.g_pos_list = None
        self.obs_poly_list = None
        self.rob_init_pose_list = None
        self.l_scan_half_num = laser_scan_half_num
        self.l_scan_min_dis = laser_scan_min_dis
        self.l_scan_scale = laser_scan_scale
        self.scan_dir_num = scan_dir_num
        self.g_dis_min_dis = goal_dis_min_dis
        self.g_dis_scale = goal_dis_scale
        self.obs_near_th = obs_near_th
        self.g_near_th = goal_near_th
        self.g_reward = goal_reward
        self.obs_reward = obs_reward
        self.g_dis_amp = goal_dis_amp
        self.step_time = step_time
        self.rob_pose = [0., 0., 0.]
        self.rob_speed = [0., 0.]
        self.rob_scan = np.zeros(self.scan_dir_num)
        self.rob_state_init = False
        self.rob_scan_init = False
        self.g_position = [0., 0.]
        self.g_dis_dir_pre = [0., 0.]
        self.g_dis_dir_cur = [0., 0.]
        rospy.Subscriber('gazebo/model_states', ModelStates, self._rob_state_cb)
        rospy.Subscriber('simplescan', SimpleScan, self._rob_scan_cb)
        self.pub_action = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=5)
        self.pause_gazebo = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.unpause_gazebo = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.set_model_target = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        self.reset_simulation = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        while not self.rob_state_init:
            continue
        while not self.rob_scan_init:
            continue
        rospy.loginfo("Finish Subscriber Init...")

    def step(self, action):
        assert self.g_pos_list is not None
        assert self.obs_poly_list is not None
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.angular.z = action[1]
        self.pub_action.publish(move_cmd)
        rospy.sleep(self.step_time)
        next_rob_state = self._get_next_rob_state()
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)
        g_dis, g_dir = self._compute_dis_dir_2_g(next_rob_state[0])
        self.g_dis_dir_cur = [g_dis, g_dir]
        state = self._rob_state_2_ddpg_state(next_rob_state)
        reward, done = self._compute_reward(next_rob_state)
        self.g_dis_dir_pre = [self.g_dis_dir_cur[0], self.g_dis_dir_cur[1]]
        return state, reward, done

    def reset(self, ita):
        assert self.g_pos_list is not None
        assert self.obs_poly_list is not None
        assert self.rob_init_pose_list is not None
        assert ita < len(self.g_pos_list)
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)
        self.g_position = self.g_pos_list[ita]
        target_msg = ModelState()
        target_msg.model_name = 'target'
        target_msg.pose.position.x = self.g_position[0]
        target_msg.pose.position.y = self.g_position[1]
        rospy.wait_for_service('gazebo/set_model_state')
        try:
            resp = self.set_model_target(target_msg)
        except rospy.ServiceException as e:
            print("Set Target Service Failed: %s" % e)
        self.pub_action.publish(Twist())
        rob_init_pose = self.rob_init_pose_list[ita]
        rob_init_quat = self._euler_2_quat(yaw=rob_init_pose[2])
        rob_msg = ModelState()
        rob_msg.model_name = 'mobile_base'
        rob_msg.pose.position.x = rob_init_pose[0]
        rob_msg.pose.position.y = rob_init_pose[1]
        rob_msg.pose.orientation.x = rob_init_quat[1]
        rob_msg.pose.orientation.y = rob_init_quat[2]
        rob_msg.pose.orientation.z = rob_init_quat[3]
        rob_msg.pose.orientation.w = rob_init_quat[0]
        rospy.wait_for_service('gazebo/set_model_state')
        try:
            resp = self.set_model_target(rob_msg)
        except rospy.ServiceException as e:
            print("Set Target Service Failed: %s" % e)
        rospy.sleep(0.5)
        rob_state = self._get_next_rob_state()
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)
        g_dis, g_dir = self._compute_dis_dir_2_g(rob_state[0])
        self.g_dis_dir_pre = [g_dis, g_dir]
        self.g_dis_dir_cur = [g_dis, g_dir]
        state = self._rob_state_2_ddpg_state(rob_state)
        return state

    def set_new_environment(self, init_pose_list, goal_list, obstacle_list):
        self.rob_init_pose_list = init_pose_list
        self.g_pos_list = goal_list
        self.obs_poly_list = obstacle_list

    def _euler_2_quat(self, yaw=0, pitch=0, roll=0):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        return [w, x, y, z]

    def _compute_dis_dir_2_g(self, pose):
        dx = self.g_position[0] - pose[0]
        dy = self.g_position[1] - pose[1]
        dis = math.sqrt(dx**2 + dy**2)
        ego_dir = math.atan2(dy, dx)
        rob_dir = pose[2]
        while rob_dir < 0:
            rob_dir += 2 * math.pi
        while rob_dir > 2 * math.pi:
            rob_dir -= 2 * math.pi
        while ego_dir < 0:
            ego_dir += 2 * math.pi
        while ego_dir > 2 * math.pi:
            ego_dir -= 2 * math.pi
        pos_dir = abs(ego_dir - rob_dir)
        neg_dir = 2 * math.pi - abs(ego_dir - rob_dir)
        if pos_dir <= neg_dir:
            dir = math.copysign(pos_dir, ego_dir - rob_dir)
        else:
            dir = math.copysign(neg_dir, -(ego_dir - rob_dir))
        return dis, dir

    def _get_next_rob_state(self):
        tmp_rob_pose = copy.deepcopy(self.rob_pose)
        tmp_rob_spd = copy.deepcopy(self.rob_speed)
        tmp_rob_scan = copy.deepcopy(self.rob_scan)
        state = [tmp_rob_pose, tmp_rob_spd, tmp_rob_scan]
        return state

    def _rob_state_2_ddpg_state(self, state):
        tmp_g_dis = self.g_dis_dir_cur[0]
        if tmp_g_dis == 0:
            tmp_g_dis = self.g_dis_scale
        else:
            tmp_g_dis = self.g_dis_min_dis / tmp_g_dis
            if tmp_g_dis > 1:
                tmp_g_dis = 1
            tmp_g_dis = tmp_g_dis * self.g_dis_scale
        ddpg_state = [self.g_dis_dir_cur[1], tmp_g_dis, state[1][0], state[1][1]]
        tmp_laser_scan = self.l_scan_scale * (self.l_scan_min_dis / state[2])
        tmp_laser_scan = np.clip(tmp_laser_scan, 0, self.l_scan_scale)
        for num in range(self.l_scan_half_num):
            ita = self.l_scan_half_num - num - 1
            ddpg_state.append(tmp_laser_scan[ita])
        for num in range(self.l_scan_half_num):
            ita = len(state[2]) - num - 1
            ddpg_state.append(tmp_laser_scan[ita])
        return ddpg_state

    def _compute_reward(self, state):
        done = False
        near_obs = False
        rob_point = Point(state[0][0], state[0][1])
        for poly in self.obs_poly_list:
            tmp_dis = rob_point.distance(poly)
            if tmp_dis < self.obs_near_th:
                near_obs = True
                break
        if self.g_dis_dir_cur[0] < self.g_near_th:
            reward = self.g_reward
            done = True
        elif near_obs:
            reward = self.obs_reward
            done = True
        else:
            reward = self.g_dis_amp * (self.g_dis_dir_pre[0] - self.g_dis_dir_cur[0])
        return reward, done

    def _rob_state_cb(self, msg):
        if self.rob_state_init is False:
            self.rob_state_init = True
        quat = [msg.pose[-1].orientation.x,
                msg.pose[-1].orientation.y,
                msg.pose[-1].orientation.z,
                msg.pose[-1].orientation.w]
        siny_cosp = 2. * (quat[0] * quat[1] + quat[2] * quat[3])
        cosy_cosp = 1. - 2. * (quat[1] ** 2 + quat[2] ** 2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        linear_spd = math.sqrt(msg.twist[-1].linear.x**2 + msg.twist[-1].linear.y**2)
        self.rob_pose = [msg.pose[-1].position.x, msg.pose[-1].position.y, yaw]
        self.rob_speed = [linear_spd, msg.twist[-1].angular.z]

    def _rob_scan_cb(self, msg):
        if self.rob_scan_init is False:
            self.rob_scan_init = True
        self.rob_scan = np.array(msg.data)

