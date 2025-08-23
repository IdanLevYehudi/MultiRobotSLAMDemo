import numpy as np

from MathUtils import *

import matplotlib.pyplot as plt


class MapEnvironment2D:
    """
    The map is assumed to be a 2D square area, centered at (0,0) and of size map_size X map_size
    """

    def __init__(self, map_size=60, num_objects=50):
        self.map_size = map_size
        self.map_min = -self.map_size / 2
        self.map_max = self.map_size / 2

        self.num_objects = num_objects
        # self.objects_locations is 3xN array. 1st row is x, 2nd row is y, and 3rd row is object id (unique for each
        # object).
        self.objects_locations = np.vstack([np.random.uniform(self.map_min, self.map_max, size=(2, self.num_objects)),
                                            np.array(range(self.num_objects))])

        self.camera_field_of_view = np.pi / 3  # 60 degrees fov
        self.camera_max_range = 10  # In meters
        self.camera_max_range_squared = self.camera_max_range ** 2

        self.observation_uncertainty_depth = 0.4  # Uncertainty in object depth measurement - in meters
        self.observation_uncertainty_ang = 5 * np.pi / 180  # Uncertainty in object angle measurement - in radians

        self.uncertainty_odom_x = 0.01  # In meters
        self.uncertainty_odom_y = 0.01  # In meters
        self.uncertainty_odom_theta = 0.0175  # Approx 1 deg in radians

        self.communication_range = 20.0  # In meters

    # Reference: https://stackoverflow.com/questions/13652518/efficiently-find-points-inside-a-circle-sector
    def objects_seen_global_frame(self, robot_xytheta):
        """
        Coordinates of objects seen by a robot, in the global reference frame.
        :param robot_xytheta:
        :return:
        """
        sector_angle_min = robot_xytheta[2] - self.camera_field_of_view / 2
        sector_angle_max = robot_xytheta[2] + self.camera_field_of_view / 2
        sector_center = robot_xytheta[:2, np.newaxis]
        objects_rel = self.objects_locations[:2] - sector_center
        sector_start = np.array([[np.cos(sector_angle_min)], [np.sin(sector_angle_min)]])
        sector_end = np.array([[np.cos(sector_angle_max)], [np.sin(sector_angle_max)]])

        objects_in_sector = np.logical_not(are_clockwise(sector_start, objects_rel)) & are_clockwise(
            sector_end, objects_rel) & is_within_radius(objects_rel, self.camera_max_range_squared)

        return self.objects_locations[:, objects_in_sector]

    def robot_measure_objects_range_bearing(self, robot_xytheta):
        objects_seen_global = self.objects_seen_global_frame(robot_xytheta)

        # The ids of the objects remain the same
        objects_seen_global_ids = objects_seen_global[2]

        # For each object, add measurement noise in the radial and tangent directions
        objects_seen_global_points = objects_seen_global[:2]
        objects_seen_robot_points = global_to_robot_points(robot_xytheta, objects_seen_global_points)

        objects_rtheta = xy_to_rangetheta(objects_seen_robot_points)
        objects_rtheta[0] = np.maximum(0, objects_rtheta[0] + np.random.normal(0, self.observation_uncertainty_depth,
                                                                               size=objects_rtheta[0].shape))
        objects_rtheta[1] += np.random.normal(0, self.observation_uncertainty_ang, size=objects_rtheta[1].shape)
        return np.vstack([objects_rtheta, objects_seen_global_ids])

    def robot_measure_objects_xy(self, robot_xytheta):
        range_bearing_ids = self.robot_measure_objects_range_bearing(robot_xytheta)
        range_bearing_ids[:2] = rangetheta_to_xy(range_bearing_ids[:2])
        return range_bearing_ids

    def add_movement_noise(self, movement_increments):
        movement_noise_x = np.random.normal(0, self.uncertainty_odom_x, size=movement_increments.shape[1])
        movement_noise_y = np.random.normal(0, self.uncertainty_odom_y, size=movement_increments.shape[1])
        movement_noise_theta = np.random.normal(0, self.uncertainty_odom_theta, size=movement_increments.shape[1])

        movement_noised = movement_increments + np.vstack([movement_noise_x, movement_noise_y, movement_noise_theta])
        movement_noised[2] = wrap_to_pi(movement_noised[2])
        return movement_noised

    def plot_objects(self, plot):
        plot.scatter(x=self.objects_locations[0], y=self.objects_locations[1], c='0.0')


if __name__ == "__main__":
    e = MapEnvironment2D()
    robot_pose_xytheta = np.array([-4, -2, np.pi / 4])
    objects_seen = e.objects_seen_global_frame(robot_pose_xytheta)

    objects_measured = e.robot_measure_objects_xy(robot_pose_xytheta)
    objects_measured_global = robot_to_global_points(robot_pose_xytheta, objects_measured[:2])

    plt.scatter(x=robot_pose_xytheta[0], y=robot_pose_xytheta[1], c='r')
    plt.scatter(x=e.objects_locations[0], y=e.objects_locations[1], c='0.8')
    plt.scatter(x=objects_seen[0], y=objects_seen[1], c='g')
    plt.scatter(x=objects_measured_global[0], y=objects_measured_global[1], c='0.0')
    # plt.savefig("test_mapping.png")
    plt.show()
