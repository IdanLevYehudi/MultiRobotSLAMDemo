import typing

from MapEnvironment import MapEnvironment2D
from DistrubtedMapper import DistributedMapper
from MathUtils import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gtsam


class Robot:
    id: int
    env: MapEnvironment2D
    mapper: DistributedMapper

    def __init__(self, id, env: MapEnvironment2D, start_xytheta=np.array([0, 0, 0])):
        self.id = id
        self.env = env
        self.start_pose = start_xytheta.reshape((3,))
        start_cov = np.array([self.env.uncertainty_odom_x,
                              self.env.uncertainty_odom_y,
                              self.env.uncertainty_odom_theta])
        self.mapper = DistributedMapper(self.id, self.start_pose, start_cov,
                                        env.uncertainty_odom_x,
                                        env.uncertainty_odom_y,
                                        env.uncertainty_odom_theta,
                                        env.observation_uncertainty_depth,
                                        env.observation_uncertainty_ang)

        self.increment_max_r = 0.5
        self.increment_max_theta = 5 * np.pi / 180

        self.milestones = np.zeros(shape=(2, 1))
        self.path_increments = np.zeros(shape=(3, 1))  # A sequence of x,y,theta commands the robot will send
        self.path = np.zeros(shape=(3, 1))  # The cumulative expected pose from all path increments

        self.path_index = 0  # Will be incremented as the simulation runs

    def set_random_milestones(self):
        num_milestones = np.random.randint(low=10, high=15)
        self.milestones = np.random.uniform(low=self.env.map_min, high=self.env.map_max, size=(2, num_milestones))
        self.milestones = np.hstack([self.milestones, self.start_pose[:2].reshape(2, 1)])

    def set_milestones(self, milestones=None):
        if milestones is None:
            self.set_random_milestones()
        else:
            self.milestones = milestones

    def plan_path_between_milestones(self):
        current_pose = np.squeeze(self.start_pose)

        current_milestone_ind = 0
        current_milestone = self.milestones[:, current_milestone_ind]

        path = []
        increments = []
        path.append(current_pose.copy())
        while True:
            dist_to_goal = np.linalg.norm(current_pose[:2] - current_milestone)

            if dist_to_goal < 1e-9:
                current_milestone_ind += 1
                if current_milestone_ind >= self.milestones.shape[1]:
                    break
                else:
                    current_milestone = self.milestones[:, current_milestone_ind]

            orientation_to_milestone = wrap_to_pi(np.arctan2(current_milestone[1] - current_pose[1],
                                                             current_milestone[0] - current_pose[0]) - current_pose[2])

            current_inc = np.zeros(shape=(3, 1))
            if orientation_to_milestone > 1e-9:
                if self.increment_max_theta > orientation_to_milestone:
                    current_inc[2] = orientation_to_milestone
                else:
                    current_inc[2] = self.increment_max_theta
            elif orientation_to_milestone < -1e-9:
                if -self.increment_max_theta < orientation_to_milestone:
                    current_inc[2] = orientation_to_milestone
                else:
                    current_inc[2] = -self.increment_max_theta
            else:  # Meaning orientation is already towards goal, we can start advancing towards it
                if dist_to_goal < self.increment_max_r:
                    current_inc[:2] = np.array([[dist_to_goal], [0]])
                else:
                    current_inc[:2] = np.array([[self.increment_max_r], [0]])

            increments.append(np.squeeze(current_inc))
            current_pose = np.squeeze(robot_to_global_poses(current_pose, current_inc))
            path.append(current_pose.copy())

        self.path_increments = np.array(increments).T
        self.path = np.array(path).T

    def generate_movement(self):
        if self.path_index >= self.path_increments.shape[1]:
            return np.zeros(shape=(3,))
        else:
            increment_to_return = self.path_increments[:, self.path_index]
            self.path_index += 1
            return increment_to_return


if __name__ == "__main__":
    env = MapEnvironment2D()
    robot = Robot(0, env)
    robot.set_milestones()
    robot.plan_path_between_milestones()

    path_colors = cm.rainbow(np.linspace(0, 1, robot.path.shape[1]))

    plt.scatter(x=env.objects_locations[0], y=env.objects_locations[1], c='0.8')
    for i in range(robot.path.shape[1]):
        plt.scatter(x=robot.path[0, i], y=robot.path[1, i], color=path_colors[i])

    # plt.show()
    plt.savefig("test_robot.png")
