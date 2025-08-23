import typing

from datetime import datetime
import imageio

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from MapEnvironment import MapEnvironment2D
from Robot import Robot
from MathUtils import *


class Simulation:
    robots: typing.List[Robot]

    def __init__(self, num_robots=16, rendezvous_on=True):
        self.num_robots = num_robots
        self.iteration_number = 0

        self.env = MapEnvironment2D()

        self.robots = []
        self.robots_gt_paths = []

        self.rendezvous_on = rendezvous_on

        self.robot_objects_colors = cm.rainbow(np.linspace(0, 1, self.num_robots))
        self.robot_colors = cm.rainbow(np.linspace(0, 1, self.num_robots))
        self.robot_colors[:, 3] *= 0.85
        self.path_marker_size = 1

        self.plot_images = []  # A list of images, in the end write as gif

        self.set_robot_start_locations()
        self.initialize_robot_paths()

    def set_robot_start_locations(self, robot_locations=None):
        for i in range(self.num_robots):
            if robot_locations is None:
                rand_xytheta = np.zeros(shape=(3,))
                rand_xytheta[:2] = np.random.uniform(self.env.map_min, self.env.map_max, size=(2,))
                rand_xytheta[2] = np.random.uniform(-np.pi, np.pi)
                location_i = rand_xytheta
            else:
                location_i = robot_locations[i].reshape((3,))
            self.robots.append(Robot(id=i, env=self.env, start_xytheta=location_i))
            self.robots_gt_paths.append(location_i.reshape(3, 1))

    def initialize_robot_paths(self, milestones=None):
        for i in range(self.num_robots):
            self.robots[i].set_milestones(milestones)
            self.robots[i].plan_path_between_milestones()

    def advance_robots(self):
        movements = np.zeros(shape=(3, len(self.robots)))
        for i in range(len(self.robots)):
            movements[:, i] = self.robots[i].generate_movement()
            # TODO: is this a semplification we do that the odometry measurements correespond to the movements commanded by each robot?
            # Idan: No, the robots attempt to move the planned movement, but in reality they advance by an action with
            # some noise added. Since the noise mean is 0, we can directly say that the expected value of the odometry
            # should be what the robots planned to execute.
            self.robots[i].mapper.add_odometry(np.squeeze(movements[:, i]), self.robots[i].path_index)
        return movements, self.env.add_movement_noise(movements)

    def update_gt_poses(self, movements):
        for i in range(self.num_robots):
            last_pose_i = self.robots_gt_paths[i][:, -1]
            next_pose_i = robot_to_global_poses(np.squeeze(last_pose_i), movements[:, i])
            self.robots_gt_paths[i] = np.hstack([self.robots_gt_paths[i], next_pose_i])

    def collect_measurements(self):
        objects_seen_per_robot = []
        for i in range(self.num_robots):
            # Append the objects seen from each robot's latest gt pose.
            objects_seen_per_robot.append(self.env.robot_measure_objects_range_bearing(self.robots_gt_paths[i][:, -1]))
        return objects_seen_per_robot

    def send_measurements_to_robots(self, objects_seen_per_robot):
        for i in range(self.num_robots):
            self.robots[i].mapper.add_current_objects_seen(objects_seen_per_robot[i], self.robots[i].path_index)

    def add_rendezvous_shared_ids(self, objects_seen_per_robot):
        for i in range(self.num_robots):
            for j in range(i):
                # If the robots are further away than twice the sensing range, they cannot perceive the same objects
                # Also, the robots must be in communication range
                gt_pose_i = self.robots_gt_paths[i][:, -1]
                gt_pose_j = self.robots_gt_paths[j][:, -1]
                robot_distance = np.linalg.norm(gt_pose_i[:2] - gt_pose_j[:2])
                if robot_distance > 2 * self.env.camera_max_range or robot_distance > self.env.communication_range:
                    continue

                # Otherwise, check for common objects that they see, and pass each's estimate to the other robot.
                intersect_ids = intersect_objects_ids(objects_seen_per_robot[i], objects_seen_per_robot[j])

                # Add these to the shared objects of each robot
                self.robots[i].mapper.add_shared_ids(j, intersect_ids)
                self.robots[j].mapper.add_shared_ids(i, intersect_ids)

    def communicate_shared_objects(self):
        for i in range(self.num_robots):
            for j in range(i):
                # If the robots are further away than twice the sensing range, they cannot communicate
                gt_pose_i = self.robots_gt_paths[i][:, -1]
                gt_pose_j = self.robots_gt_paths[j][:, -1]
                robot_distance = np.linalg.norm(gt_pose_i[:2] - gt_pose_j[:2])
                if robot_distance > self.env.communication_range:
                    continue

                objects_robot_j_wants = self.robots[j].mapper.get_shared_objects_ids(i)
                if objects_robot_j_wants:
                    objects_estimates_for_j, objects_uncertainty_for_j = self.robots[i].mapper.get_object_estimates(
                        objects_robot_j_wants)
                    self.robots[j].mapper.update_shared_objects_estimates(i, objects_estimates_for_j,
                                                                          objects_uncertainty_for_j)

                objects_robot_i_wants = self.robots[i].mapper.get_shared_objects_ids(j)
                if objects_robot_i_wants:
                    objects_estimates_for_i, objects_uncertainty_for_i = self.robots[j].mapper.get_object_estimates(
                        objects_robot_i_wants)
                    self.robots[i].mapper.update_shared_objects_estimates(j, objects_estimates_for_i,
                                                                          objects_uncertainty_for_i)

    def update_robot_estimates(self):
        for i in range(self.num_robots):
            self.robots[i].mapper.update_current_estimates()

    def plot_robot_gt_paths(self, plot):
        for i in range(self.num_robots):
            plot.plot(self.robots_gt_paths[i][0], self.robots_gt_paths[i][1], '-o', markersize=self.path_marker_size,
                      color='0.5')
            # print("self.robots_gt_paths[%i][0]" % i)
            # print(self.robots_gt_paths[i][0])
            # print("self.robots_gt_paths[%i][1]" % i)
            # print(self.robots_gt_paths[i][1])

    def plot_objects_gt(self, plot):
        self.env.plot_objects(plot)

    def plot_robot_estimated_paths_objects(self, plot):
        # for t in range(self.iteration_number + 1):
        #     for i in range(self.num_robots):
        #         plot.scatter(self.robots[i].mapper.poses_estimates[0, t], self.robots[i].mapper.poses_estimates[1, t],
        #                      s=self.path_marker_size, color=self.robot_colors[i])

        for i in range(self.num_robots):
            plot.plot(self.robots[i].mapper.poses_estimates[0], self.robots[i].mapper.poses_estimates[1], '-o',
                      markersize=self.path_marker_size, color=self.robot_colors[i])
            plot.scatter(self.robots[i].mapper.objects_estimates[0], self.robots[i].mapper.objects_estimates[1],
                         color=self.robot_objects_colors[i])
            # print("self.robots[%i].mapper.poses_estimates[0]" % i)
            # print(self.robots[i].mapper.poses_estimates[0])
            # print("self.robots[%i].mapper.poses_estimates[1]" % i)
            # print(self.robots[i].mapper.poses_estimates[1])

    def visualize_estimates(self):
        self.plot_robot_gt_paths(plt)
        self.plot_objects_gt(plt)
        self.plot_robot_estimated_paths_objects(plt)

        plt.draw()
        plt.pause(0.01)  # Continue with simulation
        # plt.pause(0)  # Pause for user

        canvas = plt.gca().figure.canvas
        canvas.draw()
        data = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(canvas.get_width_height()[::-1] + (3,))
        self.plot_images.append(data)

        plt.clf()

    def save_plan_images_gif(self):
        plan_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        imageio.mimsave(f'slam_{plan_time}.gif', self.plot_images, 'GIF', duration=0.05)

    def run_simulation(self):
        # Simulate until one robot will run out of commands.
        num_timesteps = min([self.robots[i].path_increments.shape[1] for i in range(len(self.robots))])

        for i in range(num_timesteps):
            print("iteration number: %i" % self.iteration_number)
            planned_movements, noised_movements = self.advance_robots()
            self.update_gt_poses(noised_movements)
            objects_seen_per_robot = self.collect_measurements()
            self.send_measurements_to_robots(objects_seen_per_robot)
            self.update_robot_estimates()
            if self.rendezvous_on:
                self.add_rendezvous_shared_ids(objects_seen_per_robot)
                self.communicate_shared_objects()
            self.visualize_estimates()
            self.iteration_number += 1

        self.save_plan_images_gif()


if __name__ == "__main__":
    plt.ion()
    plt.show()

    simulation = Simulation()
    simulation.run_simulation()
