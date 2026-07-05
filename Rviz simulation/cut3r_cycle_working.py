#/home/race10/cut3r_venv/bin/python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torch.nn as nn
import sys
import os
import math
from sensor_msgs_py import point_cloud2
import std_msgs.msg
from torch.nn.functional import interpolate
from scipy.spatial.transform import Rotation
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, LivelinessPolicy

class CUT3RProcessor(Node):
    def __init__(self):
        super().__init__('cut3r_processor')
        
        # Declare parameters
        self.declare_parameter('cut3r_path', '/home/race10/CUT3R')
        self.declare_parameter('model_path', '/home/race10/CUT3R/src/cut3r_512_dpt_4_64.pth')
        self.declare_parameter('publish_current_pointcloud', True)
        self.declare_parameter('publish_aggregated_pointcloud', True)
        self.declare_parameter('publish_depth_map', True)
        self.declare_parameter('voxel_size', 0.05)  # Downsampling resolution
        self.declare_parameter('enable_colors', True)  # Enable color extraction
    
        
        cut3r_path = self.get_parameter('cut3r_path').value
        model_path = self.get_parameter('model_path').value
        
        # Add CUT3R paths to sys.path - CORRECTED PATHS
        sys.path.append(cut3r_path)
        sys.path.append(os.path.join(cut3r_path, 'src'))
        
        # Import CUT3R modules - CORRECTED IMPORTS
        try:
            # Import from the correct path based on CUT3R demo files
            from src.dust3r.model import ARCroco3DStereo
            from src.dust3r.inference import inference  # CORRECTED IMPORT PATH
            self.inference = inference  # Store as instance variable
            self.get_logger().info("Successfully imported CUT3R modules")
        except ImportError as e:
            self.get_logger().error(f"Failed to import CUT3R modules: {e}")
            return
        
        # ROS communication setup
        sensor_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            liveliness=LivelinessPolicy.AUTOMATIC,
            depth=1  # Example queue size
        )
        self.subscription = self.create_subscription(
            Image,
            #'/cam1/color/image_raw',
            '/race10/cam1/color/image_raw',
            self.image_callback,
            sensor_qos_profile
            )
        
        self.current_pose = np.eye(4)
        self.pose_increment_angle = 0.0  # Track rotation for 360° capture

        
        # Publishers
        self.pointcloud_publisher = self.create_publisher(PointCloud2, '/cut3r/aggregated_pointcloud', 10)
        self.current_pointcloud_publisher = self.create_publisher(PointCloud2, '/cut3r/current_pointcloud', 10)
        self.depth_map_publisher = self.create_publisher(Image, '/cut3r/depth_map', 10)
        
        # Initialize attributes
        self.cv_bridge = CvBridge()
        self.frame_count = 0
        
        # Load CUT3R model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.model = ARCroco3DStereo.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.get_logger().info(f"CUT3R model loaded successfully from {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load CUT3R model: {e}")
            return
        
        # CUT3R persistent state
        self.current_image = None
        self.persistent_state = None
        self.accumulated_points = []
        self.previous_frame = None
        
        self.get_logger().info(f"Initialized CUT3R Processor with continuous updating")
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
    
    def image_callback(self, msg):
        """Process single frame continuously - CUT3R approach"""
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        
        # CUT3R processes single frames continuously
        self.process_continuous_frame(cv_image)

    # def estimate_pose_increment(self):
    #     """Estimate camera pose increment for 360° reconstruction"""
    #     # Adjust these values based on your camera movement
    #     forward_speed = 0.5  # 5cm per frame
    #     rotation_speed = np.radians(20)  # 3 degrees per frame
        
    #     # Create incremental transformation
    #     increment = np.eye(4)
        
    #     # Add circular motion
    #     increment[0, 3] = forward_speed * np.cos(self.pose_increment_angle)
    #     increment[2, 3] = forward_speed * np.sin(self.pose_increment_angle)
        
    #     # Add rotation for 360° capture
    #     increment[0, 0] = np.cos(rotation_speed)
    #     increment[0, 2] = np.sin(rotation_speed)
    #     increment[2, 0] = -np.sin(rotation_speed)
    #     increment[2, 2] = np.cos(rotation_speed)
        
    #     # Update cumulative pose
    #     self.current_pose = self.current_pose @ increment
    #     self.pose_increment_angle += rotation_speed
        
    #     return self.current_pose

    def get_incremental_camera_pose(self):
        if not hasattr(self, 'pose_counter'):
            self.pose_counter = 0
        
        self.pose_counter += 1
        
        # Create circular motion for 360° capture (5° per frame)
        angle = (self.pose_counter * np.pi) / 36  # 72 frames for full circle
        
        # Position on circle (1 meter radius)
        x = 1.0 * np.cos(angle)
        z = 1.0 * np.sin(angle)
        y = 0.0  # Keep at same height
        
        # Camera looks toward center
        look_angle = angle + np.pi
        
        pose = np.eye(4)
        # Rotation to look toward center
        pose[0, 0] = np.cos(look_angle)
        pose[0, 2] = np.sin(look_angle)
        pose[2, 0] = -np.sin(look_angle)
        pose[2, 2] = np.cos(look_angle)
        
        # Position
        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = z
        
        return pose

    def voxel_grid_downsample(self, points, colors, voxel_size=0.05):
        """Downsample point cloud using voxel grid filtering"""
        if len(points) == 0:
            return points, colors
    
        # Compute voxel indices
        min_bound = points.min(axis=0)
        voxel_indices = np.floor((points - min_bound) / voxel_size).astype(np.int32)
    
        # Use dictionary to accumulate points/colors per voxel
        voxel_dict = {}
        for idx, v in enumerate(voxel_indices):
            key = tuple(v)
            if key not in voxel_dict:
                voxel_dict[key] = {'points': [], 'colors': []}
            voxel_dict[key]['points'].append(points[idx])
            voxel_dict[key]['colors'].append(colors[idx])
    
        # Average points and colors in each voxel
        downsampled_points = []
        downsampled_colors = []
        for voxel_data in voxel_dict.values():
            downsampled_points.append(np.mean(voxel_data['points'], axis=0))
            downsampled_colors.append(np.mean(voxel_data['colors'], axis=0))
    
        return np.array(downsampled_points), np.array(downsampled_colors)

    def extract_colors_from_image(self, points, image, image_shape=(224, 224)):
        """Extract RGB colors from image for corresponding 3D points"""
        if not hasattr(self, 'current_image') or self.current_image is None:
            # Return default colors if no image available
            return np.full((len(points), 3), [128, 128, 128], dtype=np.uint8)
    
        # Resize image to match point cloud resolution
        resized_image = cv2.resize(self.current_image, image_shape)
    
        # Create colors array
        colors = []
        height, width = image_shape
    
        for i, point in enumerate(points):
            # Map point index to image coordinates
            row = i // width
            col = i % width
        
            # Ensure coordinates are within bounds
            if row < height and col < width:
                # Extract RGB color from image
                color = resized_image[row, col]  # BGR format from OpenCV
                colors.append([color[2], color[1], color[0]])  # Convert BGR to RGB
            else:
                colors.append([128, 128, 128])  # Default gray color
    
        return np.array(colors, dtype=np.uint8)


    def process_continuous_frame(self, current_image):
        """CUT3R continuous processing - no image pairs needed"""
        with torch.no_grad():
            # Prepare current frame
            current_frame = self.prepare_frame(current_image)
            
            if self.previous_frame is None:
                # Initialize with first frame
                self.previous_frame = current_frame
                self.get_logger().info("Initialized CUT3R with first frame")
                return
            
            # CUT3R continuous updating with persistent state
            try:
                # Create view pair for inference (current approach in CUT3R demo)
                views = [self.previous_frame, current_frame]
                
                # Use the stored inference function - FIXED
                outputs, state_args = self.inference(views, self.model, self.device)
                
                # Update persistent state - CUT3R's key feature
                self.update_persistent_state(state_args)
                
                # Extract and process results
                if 'pred' in outputs and len(outputs['pred']) >= 2:
                    current_pred = outputs['pred'][1]  # Current view prediction
                    
                    # Publish current pointcloud
                    self.publish_current_point_cloud(current_pred)
                    
                    # Publish depth map
                    self.publish_depth_map(current_pred)
                    
                    # Accumulate for dense reconstruction
                    if self.frame_count % 5 == 0:  # More frequent updates for continuous processing
                        self.publish_accumulated_point_cloud(current_pred)
                
                # Update for next iteration - sliding window approach
                self.previous_frame = current_frame
                self.frame_count += 1
                
            except Exception as e:
                self.get_logger().error(f"CUT3R continuous processing failed: {e}")
    
    def prepare_frame(self, image):
        """Prepare single frame for CUT3R processing and store original image"""
        # Store original image for color extraction
        self.current_image = image.copy()

        # Estimate current pose instead of using identity
        current_pose = self.get_incremental_camera_pose()
    
        # Existing frame preparation code...
        H, W = 224, 224
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        img_tensor = interpolate(img_tensor, size=(H, W), mode='bilinear')
        true_shape = torch.tensor([H, W]).unsqueeze(0).to(self.device)
        #camera_pose = torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0).to(self.device)
        # Use estimated pose instead of identity
        camera_pose = torch.from_numpy(current_pose.astype(np.float32)).unsqueeze(0).to(self.device)
        img_mask = torch.tensor([True], dtype=torch.bool).to(self.device)
        ray_mask = torch.tensor([False], dtype=torch.bool).to(self.device)
        update = torch.tensor([True], dtype=torch.bool).to(self.device)
        reset = torch.tensor([False], dtype=torch.bool).to(self.device)
        instance = str(self.frame_count)
        ray_map = torch.full((1, 6, H, W), torch.nan, dtype=torch.float32).to(self.device)
    
        return {
        "img": img_tensor,
        "true_shape": true_shape,
        "camera_pose": camera_pose,
        "img_mask": img_mask,
        "ray_map": ray_map,
        "ray_mask": ray_mask,
        "update": update,
        "reset": reset,
        "instance": instance,
        }
    
    def update_persistent_state(self, state_args):
        """Update CUT3R's persistent state representation"""
        # CUT3R maintains persistent state across observations
        if self.persistent_state is None:
            self.persistent_state = state_args
        else:
            # Continuously update the persistent state
            # This is where CUT3R's continuous updating happens
            self.persistent_state = self.merge_states(self.persistent_state, state_args)
    
    def merge_states(self, previous_state, new_state):
        """Merge previous persistent state with new observations"""
        # This is a simplified merge - actual implementation depends on CUT3R's state structure
        # CUT3R's persistent state accumulates information over time
        if isinstance(new_state, dict) and isinstance(previous_state, dict):
            merged_state = previous_state.copy()
            merged_state.update(new_state)
            return merged_state
        else:
            return new_state
    
    def reset_for_new_sequence(self):
        """Reset persistent state for new sequence - CUT3R capability"""
        self.persistent_state = None
        self.previous_frame = None
        self.accumulated_points = []
        self.frame_count = 0
        self.get_logger().info("CUT3R persistent state reset for new sequence")
    
    def rotate_points(self, points):
        """Rotate points for ROS coordinate system"""
        r = Rotation.from_euler('x', -90, degrees=True)
        rotated_points = r.apply(points)
        return rotated_points
    
    def project_points_to_depth_map_current(self, points, image_shape):
        """Project 3D points to depth map - adapted from working version"""
        height, width = image_shape
        depth_map = np.full((height, width), np.inf, dtype=np.float32)
        
        def norm_pt(point):
            return math.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
        
        for i in range(height):
            for j in range(width):
                if i * width + j < len(points):
                    depth_map[i, j] = norm_pt(points[i * width + j])
        
        # Replace inf with max depth
        max_depth = np.max(depth_map[depth_map != np.inf])
        depth_map[depth_map == np.inf] = max_depth
        
        # Normalize depth map
        depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
        return depth_map
    
    #     return create_cloud(header, fields, structured_array)
    def create_colored_pointcloud2(self, points, colors, header):
        """Create a colored PointCloud2 message with proper RGB field alignment"""
        import numpy as np
        from sensor_msgs_py.point_cloud2 import create_cloud
        from sensor_msgs.msg import PointField
        
        # Ensure colors are in the right format
        if colors.dtype != np.uint8:
            colors = colors.astype(np.uint8)
        
        # Method 1: Using packed RGB (recommended for RViz2 compatibility)
        # Pack RGB values into a single 32-bit integer
        rgb_packed = np.zeros(len(points), dtype=np.uint32)
        rgb_packed = (colors[:, 0].astype(np.uint32) << 16) | \
                    (colors[:, 1].astype(np.uint32) << 8) | \
                    (colors[:, 2].astype(np.uint32))
        
        # Create structured array with XYZ and packed RGB
        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)]
        structured_array = np.zeros(len(points), dtype=dtype)
        structured_array['x'] = points[:, 0]
        structured_array['y'] = points[:, 1]
        structured_array['z'] = points[:, 2]
        structured_array['rgb'] = rgb_packed
        
        # Define fields with correct offsets
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        return create_cloud(header, fields, structured_array)

    def publish_current_point_cloud(self, pred):
        """Publish current frame point cloud with colors and downsampling"""
        if not self.get_parameter('publish_current_pointcloud').value:
            return
    
        try:
            # Extract points from CUT3R prediction
            if 'pts3d' in pred:
                points = pred['pts3d'].squeeze().cpu().numpy()
            elif 'pts3d_in_other_view' in pred:
                points = pred['pts3d_in_other_view'].squeeze().cpu().numpy()
            else:
                self.get_logger().warn("No points found in current prediction")
                return
        
            # Ensure proper shape
            if points.ndim == 1:
                points = points.reshape(-1, 3)
            elif points.ndim > 2:
                points = points.reshape(-1, 3)
        
            points = points.astype(np.float64)
        
            # Extract colors from current image
            colors = self.extract_colors_from_image(points, self.current_image)
        
            # Rotate points for ROS coordinate system
            rotated_points = self.rotate_points(points)
        
            # Downsample point cloud
            voxel_size = 0.05  # Adjust this value to control downsampling level
            downsampled_points, downsampled_colors = self.voxel_grid_downsample(
                rotated_points, colors, voxel_size
            )
        
            # Create colored point cloud message
            header = std_msgs.msg.Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "map"
        
            pc2_msg = self.create_colored_pointcloud2(downsampled_points, downsampled_colors, header)
            self.current_pointcloud_publisher.publish(pc2_msg)
        
            self.get_logger().info(f"Published {len(downsampled_points)} colored points (downsampled from {len(points)})")
        
        except Exception as e:
            self.get_logger().error(f"Failed to publish current point cloud: {e}")

    
    def publish_accumulated_point_cloud(self, pred):
        """Publish accumulated dense reconstruction with color and downsampling"""
        if not self.get_parameter('publish_aggregated_pointcloud').value:
            return

        try:
            # Extract and accumulate points for dense reconstruction
            if 'pts3d' in pred:
                points = pred['pts3d'].squeeze().cpu().numpy()
            elif 'pts3d_in_other_view' in pred:
                points = pred['pts3d_in_other_view'].squeeze().cpu().numpy()
            else:
                return

            if points.ndim == 1:
                points = points.reshape(-1, 3)
            elif points.ndim > 2:
                points = points.reshape(-1, 3)

            points = points.astype(np.float64)

            # Extract colors from the current image
            colors = self.extract_colors_from_image(points, self.current_image)

            # Rotate points for ROS coordinate system
            rotated_points = self.rotate_points(points)

            # Accumulate points and colors
            if not hasattr(self, 'accumulated_colors'):
                self.accumulated_colors = []
            self.accumulated_points.append(rotated_points)
            self.accumulated_colors.append(colors)

            # Concatenate all accumulated points and colors
            accumulated_point_cloud = np.concatenate(self.accumulated_points, axis=0)
            accumulated_colors = np.concatenate(self.accumulated_colors, axis=0)

            if accumulated_point_cloud.size == 0:
                return

            # Downsample the point cloud and colors
            voxel_size = self.get_parameter('voxel_size').value if self.has_parameter('voxel_size') else 0.05
            downsampled_points, downsampled_colors = self.voxel_grid_downsample(
                accumulated_point_cloud, accumulated_colors, voxel_size
            )

            # Create colored PointCloud2 message
            header = std_msgs.msg.Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "map"

            pc2_msg = self.create_colored_pointcloud2(downsampled_points, downsampled_colors, header)
            self.pointcloud_publisher.publish(pc2_msg)

            self.get_logger().info(
                f"Published {len(downsampled_points)} colored points (downsampled from {len(accumulated_point_cloud)})"
            )

        except Exception as e:
            self.get_logger().error(f"Failed to publish accumulated point cloud: {e}")

    
    def publish_depth_map(self, pred):
        """Publish depth map from current prediction"""
        if not self.get_parameter('publish_depth_map').value:
            return
        
        try:
            # Extract points for depth map
            if 'pts3d' in pred:
                points = pred['pts3d'].squeeze().cpu().numpy()
            elif 'pts3d_in_other_view' in pred:
                points = pred['pts3d_in_other_view'].squeeze().cpu().numpy()
            else:
                return
            
            if points.ndim == 1:
                points = points.reshape(-1, 3)
            elif points.ndim > 2:
                points = points.reshape(-1, 3)
            
            # Generate depth map
            depth_map = self.project_points_to_depth_map_current(points, (224, 224))
            
            # Convert to ROS image message
            depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_map.astype(np.float32), encoding="32FC1")
            self.depth_map_publisher.publish(depth_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish depth map: {e}")

def main(args=None):
    rclpy.init(args=args)
    cut3r_processor = CUT3RProcessor()
    rclpy.spin(cut3r_processor)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
