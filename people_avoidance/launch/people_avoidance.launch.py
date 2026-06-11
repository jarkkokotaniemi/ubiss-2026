from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='people_avoidance',
            executable='people_avoidance_node',
            name='people_avoidance',
            output='screen',
            parameters=[{
                # ── Topics ──────────────────────────────────────────────────
                'scan_topic':            '/scan',
                'cmd_vel_topic':         '/cmd_vel',
                'odom_topic':            '/odom',
                # ── Frames ──────────────────────────────────────────────────
                'laser_frame':           'rplidar_link',
                'odom_frame':            'odom',
                # ── Kalman filter ────────────────────────────────────────────
                'dt':                    0.1,    # seconds; match to LiDAR scan rate
                'max_misses':            5,      # frames before a track is deleted
                # ── Leg detection ────────────────────────────────────────────
                'distance_threshold':    0.1,    # segmentation gap (m)
                'leg_radius':            0.10,   # expected single-leg radius (m)
                'max_leg_width':         0.25,   # max distance between paired legs (m)
                # ── Controller ───────────────────────────────────────────────
                'max_linear_speed':      0.2,    # m/s
                'max_angular_speed':     1.0,    # rad/s
                'obstacle_radius_scale': 2.0,    # uncertainty inflation factor k
            }],
        ),
        Node(
            package='people_avoidance',
            executable='visualizer',
            name='lidar_visualizer',
            output='screen'
        ),
    ])
