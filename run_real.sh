#!/usr/bin/env bash
# The world model driven by the real G2. Needs the G2 network and stack (see README); extra args go to the launch file.
set -e
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${G2_ENV:-$HOME/agi@g2_robot/env_g2.sh}"   # the G2 GDK ROS environment (override with G2_ENV=...)
source "$root/install/local_setup.bash"
export ROS_LOG_DIR="$root/log/ros"
exec ros2 launch worldsim_g2 real.launch.py "$@"
