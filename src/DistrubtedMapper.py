import typing

import numpy
from MathUtils import *
import gtsam
from gtsam.symbol_shorthand import L, X


def gtsam_point2(np_point):
    """
    Constructs a gtsam Point2 from a numpy array of [x, y].
    """
    return gtsam.Point2(x=np_point[0], y=np_point[1])


def gtsam_pose2(np_pose):
    """
    Constructs a gtsam Pose2 from a numpy array of [x, y, theta].
    """
    # Simply passing the np array results in a wrong pose2, so we need to explicitly set x, y, theta.
    return gtsam.Pose2(x=np_pose[0], y=np_pose[1], theta=np_pose[2])


def gtsam_rot2(theta):
    """
    Constructs a gtsam Rot2 from a float theta.
    """
    return gtsam.Rot2(theta)


class DistributedMapper:
    trajectory: gtsam.Values
    shared_objects: typing.Dict[typing.Any, typing.Dict[typing.Any, typing.Tuple[np.array, gtsam.gtsam.Marginals]]]
    objects_estimates_with_cov: typing.Dict[typing.Any, typing.Tuple[np.array, gtsam.gtsam.Marginals]]

    def __init__(self, id, start_pose, start_cov, uncertainty_odom_x, uncertainty_odom_y, uncertainty_odom_theta,
                 uncertainty_measurement_range, uncertainty_measurement_theta):
        self.id = id  # id of this robot - unique for each robot

        self.poses_estimates = start_pose.reshape((3, 1))  # 3xT, T is the number of time steps.
        self.objects_estimates = np.zeros(shape=(3, 0))  # 3xK, K the number of objects seen so far.
        self.poses_cov = list()  # List of poses' uncertainty ordered by the path index
        self.objects_cov = dict()  # Maps between object id to uncertainty

        # landmarks
        self.shared_objects = dict()  # Maps each robot id to a dict of shared objects

        # graph
        self.graph = gtsam.NonlinearFactorGraph()
        # self.last_graph will hold the graph with the shared objects constraints. The reason we need two graphs is that
        # gtsam does not support nicely factor deletion or replacement. So instead, we will keep the latest estimate of
        # the shared objects in self.shared_objects, and at each time update_current_estimates is called, we will add
        # these constraints to a copy of self.graph.
        self.last_graph = None
        start_pose_pose2 = gtsam_pose2(start_pose)
        start_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(start_cov, dtype=float))
        self.graph.add(gtsam.PriorFactorPose2(X(0), start_pose_pose2, start_noise))

        # trajectory
        self.trajectory = gtsam.Values()
        self.trajectory.insert(X(0), start_pose_pose2)
        self.path_indices = dict()
        self.path_indices[X(0)] = 0
        self.landmark_indices = dict()

        # noise models
        self.odom_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([uncertainty_odom_x, uncertainty_odom_y, uncertainty_odom_theta], dtype=float))
        self.obs_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([uncertainty_measurement_theta, uncertainty_measurement_range], dtype=float))

        # Initialize estimates for time 0
        self.last_pose_estimate = start_pose
        self.poses_cov.append(start_noise)

    def add_odometry(self, latest_odometry, path_index):
        """
        :param latest_odometry: np array pose of [x, y, theta], indicating the latest odometry measurement.
        :return: None
        """
        # Update latest estimate to be odometry increment composed on last estimate.
        self.last_pose_estimate = np.squeeze(
            robot_to_global_poses(self.last_pose_estimate, latest_odometry.reshape(3, 1)))

        self.graph.add(gtsam.BetweenFactorPose2(key1=X(path_index - 1),
                                                key2=X(path_index),
                                                relativePose=gtsam_pose2(latest_odometry),
                                                noiseModel=self.odom_noise_model))
        self.trajectory.insert(X(path_index), gtsam_pose2(self.last_pose_estimate))
        self.path_indices[X(path_index)] = path_index

    def add_current_objects_seen(self, current_objects_seen, path_index):
        """
        Adds objects seen at last pose, given in robot frame.
        :param current_objects_seen: np array of 3xN, 1st row ranges, 2nd row bearings, 3rd row ids.
        :return: None
        """

        for i in range(current_objects_seen.shape[1]):
            object_id = int(current_objects_seen[2, i])
            # Idan: Changed from Point2 factors to BearingRange, otherwise we put no constraints on the robot
            # orientation.
            self.graph.add(gtsam.BearingRangeFactor2D(poseKey=X(path_index),
                                                      pointKey=L(object_id),
                                                      measuredBearing=gtsam_rot2(current_objects_seen[1, i]),
                                                      measuredRange=current_objects_seen[0, i],
                                                      noiseModel=self.obs_noise_model))

            if L(object_id) not in self.landmark_indices:
                object_rangetheta = current_objects_seen[:2, i]
                object_xy = rangetheta_to_xy(object_rangetheta)
                object_global = np.squeeze(robot_to_global_points(self.last_pose_estimate, object_xy.reshape(2, 1)))
                self.trajectory.insert(L(object_id), gtsam_point2(object_global))
                self.landmark_indices[L(object_id)] = object_id

    def add_shared_ids(self, other_robot_id, objects_ids):
        """
        Adds object ids to the shared objects with another robot.
        :param other_robot_id: Id of the other robot.
        :param objects_ids: Iterable of ids of objects that the other robot estimated and our now shared with this
        robot.
        """
        if other_robot_id not in self.shared_objects:
            self.shared_objects[other_robot_id] = dict()

        for object_id in objects_ids:
            object_id = int(object_id)
            if object_id not in self.shared_objects[other_robot_id]:
                self.shared_objects[other_robot_id][object_id] = (None, None)

    def update_shared_objects_estimates(self, other_robot_id, objects_seen, objects_seen_uncertainty):
        """
        Updated estimates of shared objects from another robot.
        :param other_robot_id: id of the other robot
        :param objects_seen: np array of ids of the objects seen by the
        :param objects_seen_uncertainty: list of gtsam.gtsam.Marginals, one for each object in objects_seen
        :return: None
        """
        if objects_seen is None or objects_seen_uncertainty is None:
            return

        if other_robot_id not in self.shared_objects:
            self.shared_objects[other_robot_id] = dict()

        for i in range(objects_seen.shape[1]):
            object_id = int(objects_seen[2, i])
            if object_id in objects_seen_uncertainty:
                self.shared_objects[other_robot_id][object_id] = (
                    objects_seen[:, i], gtsam.noiseModel.Gaussian.Covariance(objects_seen_uncertainty[object_id]))

    def get_shared_objects_ids(self, other_robot_id):
        """
        Returns for a given robot id the ids of shared objects this robot has with it.
        :param other_robot_id: Id of the other robot
        """
        if other_robot_id in self.shared_objects:
            return self.shared_objects[other_robot_id].keys()
        else:
            return np.array([])

    def _add_shared_constraints_to_graph(self, graph: gtsam.NonlinearFactorGraph):
        """
        Adds prior factors on landmarks based on the latest estimates of shared objects.
        :param graph: gtsam graph
        """
        for shared_objects in self.shared_objects.values():
            for object_id, value in shared_objects.items():
                (object_seen, object_uncertainty) = value
                if object_seen is not None and object_uncertainty is not None:
                    graph.add(
                        gtsam.PriorFactorPoint2(L(object_id), gtsam_point2(object_seen[:2]), object_uncertainty))

    def update_current_estimates(self):
        """
        Updates current estimates to the MAP based on all information gathered until now.
        :return: None.
        """
        # Build updated graph with latest shared objects constraints
        self.last_graph = self.graph.clone()
        self._add_shared_constraints_to_graph(self.last_graph)

        # Solve graph with latest trajectory estimate

        # params = gtsam.gtsam.LevenbergMarquardtParams()
        # optimizer = gtsam.gtsam.LevenbergMarquardtOptimizer(self.last_graph, self.trajectory, params)
        params = gtsam.gtsam.DoglegParams()
        optimizer = gtsam.gtsam.DoglegOptimizer(self.last_graph, self.trajectory, params)

        self.trajectory = optimizer.optimize()

        # Calculate marginals
        marginals = gtsam.Marginals(self.last_graph, self.trajectory)

        # Update poses estimates and uncertainties
        self.poses_estimates = np.zeros(shape=(3, len(self.path_indices)))
        self.poses_cov = [None] * len(self.path_indices)
        for key, value in self.path_indices.items():
            curr_pose2 = self.trajectory.atPose2(key)
            pose_estimate_np = np.array([curr_pose2.x(), curr_pose2.y(), curr_pose2.theta()])
            self.poses_estimates[:, value] = pose_estimate_np

            # Update explicitly the last pose estimate
            if value == len(self.path_indices) - 1:
                self.last_pose_estimate = pose_estimate_np
            self.poses_cov[value] = marginals.marginalCovariance(key)

        # Update object estimates and uncertainties
        self.objects_estimates = np.zeros(shape=(3, len(self.landmark_indices)))
        object_index = 0
        for key, value in self.landmark_indices.items():
            self.objects_estimates[2, object_index] = value
            self.objects_estimates[:2, object_index] = self.trajectory.atPoint2(key)
            self.objects_cov[value] = marginals.marginalCovariance(key)
            object_index += 1

    def latest_pose_estimate(self):
        return self.poses_estimates[:, -1], self.poses_cov[-1]

    def get_object_estimates(self, object_ids):
        ids_seen = np.in1d(self.objects_estimates[2].astype(int), np.array([key for key in object_ids]),
                           assume_unique=True)
        return self.objects_estimates[:, ids_seen], {object_id: self.objects_cov[object_id] for object_id in object_ids
                                                     if object_id in self.objects_cov}
