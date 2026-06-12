#!/usr/bin/env python3
"""
odom_to_tf_broadcaster.py — Subscribes to /odom and broadcasts the transform
from "odom" to the robot's base frame (e.g., base_link, base_footprint).
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomToTFBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_to_tf_broadcaster')

        # Parameters to allow flexibility in frame names
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('child_frame_id', 'base_link')
        self.declare_parameter('publish_rate_hz', 50.0)

        odom_topic = self.get_parameter('odom_topic').value
        self.odom_frame = self.get_parameter('odom_frame_id').value
        self.child_frame = self.get_parameter('child_frame_id').value

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscription to odometry
        self.sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )

        # Timer to enforce publish rate (optional, but good practice)
        self.timer = self.create_timer(
            1.0 / self.get_parameter('publish_rate_hz').value,
            self.timer_callback
        )
        self.latest_odom = None

        self.get_logger().info(
            f'Broadcasting TF: {self.odom_frame} → {self.child_frame} '
            f'(from topic {odom_topic})'
        )

    def odom_callback(self, msg: Odometry):
        # Store the latest odometry message
        self.latest_odom = msg
        # Immediately publish the transform (timer will also republish)
        self.publish_transform(msg)

    def timer_callback(self):
        # If we have an odometry message, republish at a steady rate
        if self.latest_odom is not None:
            self.publish_transform(self.latest_odom)

    def publish_transform(self, odom_msg: Odometry):
        t = TransformStamped()
        t.header.stamp = odom_msg.header.stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.child_frame

        # Copy translation
        t.transform.translation.x = odom_msg.pose.pose.position.x
        t.transform.translation.y = odom_msg.pose.pose.position.y
        t.transform.translation.z = odom_msg.pose.pose.position.z

        # Copy rotation
        t.transform.rotation = odom_msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomToTFBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()