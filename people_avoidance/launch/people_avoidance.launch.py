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
                'goal_pose_topic':       '/goal_pose',  # PoseStamped destination (set via visualizer click)
                'lookahead_distance':    0.3,    # m; CBF lookahead point L
                'cbf_gamma':             2.0,    # CBF class-K gain (smaller = earlier/wider avoidance)
                'omega_weight':          0.1,    # QP cost weight on omega (steer-before-brake)
                'heading_gain':          2.5,    # pure-pursuit heading P-gain
                'goal_tolerance':        0.15,   # m; stop within this distance of the goal
            }],
        ),
        Node(
            package='people_avoidance',
            executable='visualizer',
            name='lidar_visualizer',
            output='screen'
        ),
    ])
