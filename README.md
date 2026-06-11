# ubiss-2026
```
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/jarkkokotaniemi/ubiss-2026.git
cd ..
colcon build --symlink-install
source install/setup.bash
ros2 launch people_avoidance people_avoidance.launch.py
```
people_avoidance_msgs has custom messages for leg measurements, more to come...

required packages for visualization2.py:

`sudo apt install python3-pyqt5 python3-pyqtgraph`
