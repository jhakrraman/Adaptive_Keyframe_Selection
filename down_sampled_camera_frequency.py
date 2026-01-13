import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class DownsampledCameraNode(Node):
    def __init__(self):
        super().__init__('downsampled_camera_node')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.original_sub = self.create_subscription(
            Image,
            '/wilbur/forward/color/image_rect_color',
            self.image_callback,
            qos_profile
        )
        self.downsampled_pub = self.create_publisher(
            Image,
            '/wilbur/forward/color/image_rect_color/down_sampled',
            10
        )
        self.timer = self.create_timer(1.0/15.0, self.timer_callback)  # 15 Hz (1/15 seconds interval)
        self.latest_image = None

    def image_callback(self, msg):
        self.latest_image = msg

    def timer_callback(self):
        if self.latest_image is not None:
            self.downsampled_pub.publish(self.latest_image)
            self.latest_image = None  # Reset to avoid republishing the same image

def main(args=None):
    rclpy.init(args=args)
    downsampled_camera_node = DownsampledCameraNode()
    rclpy.spin(downsampled_camera_node)
    downsampled_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()