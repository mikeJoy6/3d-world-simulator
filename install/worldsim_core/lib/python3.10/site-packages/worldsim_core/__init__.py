"""Core of the local 3D world simulator, shared by the simulated arms and real robots (no ROS imports here).

  tasks      released tasks, 14-d action layout, orientation rule
  robot      the ViperX 300 S pair extracted from the release: calibration, FK/IK, URDF and SRDF
  mapping    robot end-effector poses -> model action (absolute or relative), and when to step the model
  teleop     keyboard / target control in action space (the dataset's control rule)
  engine     the world model itself (torch, CUDA): seed, step, scene particles without the robot, render
"""
