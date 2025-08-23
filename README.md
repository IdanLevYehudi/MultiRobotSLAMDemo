# Vision-Aided Navigation - Distributed Object-Based SLAM (Technion Course 86761)

This repo contains the final project from the Technion course [Vision Based Navigation and Mapping (086761)](https://anpl-technion.github.io/Teaching/#VAN). This is an adapted re-implementation of distributed, object-based multi-robot SLAM under privacy and communication constraints, inspired by:

```
Choudhary, S., Carlone, L., Nieto, C., Rogers, J., Christensen, H. I., & Dellaert, F. (2017).
Distributed mapping with privacy and communication constraints: Lightweight algorithms and object-based models.
The International Journal of Robotics Research, 36(12), 1286-1311.
```

[Link to full paper](https://journals.sagepub.com/doi/full/10.1177/0278364917732640).

Please cite the original paper if you use this work academically.

# Simulation Animations

![video1](dist_slam_gifs/compressed/rendezvous_true_depth_10_comm_20/slam_02-02-2022_15_15_40_robots_8_map_100_objects_200.gif)

![video2](dist_slam_gifs/compressed/rendezvous_true_depth_10_comm_20/slam_02-02-2022_20_01_49_robots_16_map_80_objects_50.gif)

- Robots move along pre-determined paths, each computing its individual SLAM (indicated by different colors).
    - The robots' paths are the drawn lines: each robot's estimate in a different color, ground truths in grey.
    - Object positions are drawn as scattered dots: each robot's estimate in a different color, ground truths in grey.
- Robots measure objects' positions when they're in their field of view (according to robot heading and distance), adding objects incrementally to their individual factor graph.
- Whenever two robots view the same object at the same time, they share to one another its most updated position estimate (with object ID for data association purposes).

# Summary of Implementation

This project simulates distributed, object-based multi-robot SLAM in 2D using a non-linear factor-graph backend (GTSAM).

The key difference from a "naive" SLAM is the ***rendezvous*** - whenever two robots observe the same object at the same time, they exchange the object ID + position estimate. Each robot incorporates the received object estimate as a unary factor over that object position. This communication scheme respects privacy, i.e. does not communicate any information about path history of each robot, yet dramatically improves the position accuracy of each individual robot.

- **Backend (GTSAM, non-linear)**: We build a NonlinearFactorGraph with a pose prior, odometry Between(Pose2) factors, and BearingRangeFactor2D measurements to object landmarks (modeled as Point2). Optimization uses Dogleg; after each solve we compute marginals for poses and landmarks.

- **Sensing and noise model**: A pinhole-like range-bearing sensor with FOV $60\degree$ and max range $10 \text{m}$ - this demonstrates semantic object detections in camera images. Measurement noise $\sigma_{r} = 0.4 \text{m}$, $\sigma_{\theta} = 5\degree$. Odometry noise $\sigma_{x} = \sigma_{y} = 0.01 \text{m}$, $\sigma_{\theta} = 1\degree$. These parameters are used in the diagonal noise models in the transition and observation factors.

- **Communication constraints in simulation**: Robots only exchange data when within communication range of $20 \text{m}$. Rendezvous/sharing is triggered only when close and actually observe common objects (ID intersection).

- **Operation cycle per iteration (simplified from the paper)**: move $\to$ measure objects $\to$ optimize locally (non-linear) $\to$ (if in range) share objects/covariances. The motivation for "optimize-then-share" is to ensure new objects have a global estimate before being transmitted.

- **Environment & motion planning**: Square map (default $L=60$) with $K$ random objects (default $50$). Each robot samples $10$ - $15$ milestones and follows turn-then-go increments (max step $0.5 \text{m}$, max turn $5\degree$), with zero-mean process noise added by the environment.


## Key Differences in Implementation from Paper

- Non-Linear factor graph solution: 
    - Unlike the original two-stage, distributed linear approach with JOR/SOR updates, we solve the full problem directly with a Dogleg optimizer and treat inter-robot sharing as priors on shared landmarks.
    - This allows to incorporate general factors, like the bearing-range factors we use for more realistic observations.
- Communication Range:
    - Shared objects positions are exchanged only when robots are within a finite communication range.
    - The goal is to study how realistic, intermittent communication affects accuracy and stability.
    

## Code map:

- MapEnvironment2D: Represents the world and physics, including sensor model, noise, FOV/range, comm range.
- Robot.py: Represents a single robot in the simulation: milestone planning and path increments; constructs DistributedMapper for each robot.
- DistributedMapper.py: Distributed SLAM module: builds/updates the graph, adds odometry and bearing-range factors, optimizes (Dogleg), computes marginals, manages shared-object priors.
- Simulation.py: Multi-robot simulation management: iterates move/measure/optimize; handles rendezvous (ID intersection) and communication of estimates with covariances.

# Getting Started

```
pip install -r requirements.txt
python3 src/Simulation.py
```

# Acknowledgements

Course [Vision Based Navigation and Mapping (086761)](https://anpl-technion.github.io/Teaching/#VAN), Technion - Prof. Vadim Indelman.

Project by Idan Lev-Yehudi & Pietro Brach Del Prever.