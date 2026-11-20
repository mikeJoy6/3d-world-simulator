#!/usr/bin/env bash
set -e
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
source "$root/install/local_setup.bash"
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID="${WORLDSIM_DOMAIN_ID:-78}"
export ROS_LOG_DIR="$root/log/ros"
exec ros2 run worldsim_sim keyboard
