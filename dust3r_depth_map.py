import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Header
import numpy as np
import torch
from sensor_msgs_py import point_cloud2
import cv_bridge

class PointCloudToDepthMap(Node):
    def __init__(self):
        super().__init__('pointcloud_to_depthmap')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/race13/res1_pts3d_cloud',
            self.pointcloud_callback,
            10)
        self.publisher = self.create_publisher(Image, 'new_output_depth_map', 10)
        self.cv_bridge = cv_bridge.CvBridge()

    def pointcloud_callback(self, msg):
        # Convert PointCloud2 to structured numpy array
        pc = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pc_struct = np.array(list(pc))

        if pc_struct.size == 0:
            self.get_logger().warn("Received empty point cloud")
            return

        # Extract x, y, z coordinates
        points = np.column_stack((pc_struct['x'], pc_struct['y'], pc_struct['z'])).astype(np.float32)

        # Reshape to 512x384x3 (original image size)
        input_shape = (512, 384, 3)
        if points.shape[0] >= 512 * 384:
            points_reshaped = points[:512*384].reshape(input_shape)
        else:
            # If we don't have enough points, we'll need to pad the array
            points_reshaped = np.zeros(input_shape, dtype=np.float32)
            points_reshaped.flat[:points.shape[0]*3] = points.flat

        # Convert numpy array to torch tensor
        points_tensor = torch.from_numpy(points_reshaped)

        # Generate depth map
        depth_map = self.project_points_to_depth_map_current(points_tensor)

        # Convert depth map to image message
        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_map.numpy().astype(np.float32), encoding="32FC1")
        depth_msg.header = msg.header

        # Publish depth map
        self.publisher.publish(depth_msg)

    def project_points_to_depth_map_current(self, points):
        depth_map = torch.norm(points, dim=-1)
        max_depth = torch.max(depth_map)
        depth_map = torch.where(torch.isinf(depth_map), max_depth, depth_map)
        min_depth = torch.min(depth_map)
        depth_range = torch.max(depth_map) - min_depth
        depth_map = (depth_map - min_depth) / depth_range
        return depth_map

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudToDepthMap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image, PointCloud2
# from std_msgs.msg import Float32MultiArray
# import numpy as np
# import torch
# from cv_bridge import CvBridge
# import sensor_msgs_py.point_cloud2 as pc2

# class PointCloudTensorToDepthMap(Node):
#     def __init__(self):
#         super().__init__('pointcloud_tensor_to_depthmap')
#         self.subscription = self.create_subscription(
#             PointCloud2,
#             '/race13/res1_pts3d_topic',
#             self.pointcloud_callback,
#             10)
#         self.publisher = self.create_publisher(Image, 'new_output_depth_map', 10)
#         self.cv_bridge = CvBridge()

#     def pointcloud_callback(self, msg):
#         # Convert PointCloud2 to numpy array
#         pc = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
#         points = np.array(list(pc), dtype=np.float32)

#         if points.size == 0:
#             self.get_logger().warn("Received empty point cloud")
#             return

#         # Reshape points to (512, 384, 3) if necessary
#         if points.shape[0] != 512 * 384:
#             self.get_logger().warn(f"Unexpected number of points: {points.shape[0]}")
#             return
#         points_reshaped = points.reshape(512, 384, 3)

#         # Convert numpy array to torch tensor
#         points_tensor = torch.from_numpy(points_reshaped)

#         # Generate depth map
#         depth_map = self.project_points_to_depth_map_current(points_tensor)

#         # Convert depth map to image message
#         depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_map.numpy().astype(np.float32), encoding="32FC1")
#         depth_msg.header = msg.header  # Use the same header as the input message

#         # Publish depth map
#         self.publisher.publish(depth_msg)

#     def project_points_to_depth_map_current(self, points):
#         depth_map = torch.norm(points, dim=-1)
#         max_depth = torch.max(depth_map)
#         depth_map = torch.where(torch.isinf(depth_map), max_depth, depth_map)
#         min_depth = torch.min(depth_map)
#         depth_range = max_depth - min_depth
#         depth_map = (depth_map - min_depth) / depth_range
#         return depth_map

# def main(args=None):
#     rclpy.init(args=args)
#     node = PointCloudTensorToDepthMap()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()