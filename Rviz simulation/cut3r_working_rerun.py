#!/usr/bin/env python3

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
import rerun as rr  # Add Rerun import

import sys
print("PYTHON EXECUTABLE:", sys.executable)
import rerun as rr
print("RERUN VERSION:", rr.__version__)


class CUT3RProcessor(Node):
    def __init__(self):
        super().__init__('cut3r_processor')
        
        # Initialize Rerun
        #rr.init("CUT3R_3D_Reconstruction", spawn=True)
        rr.init("CUT3R_3D_Reconstruction", spawn=False, connect="127.0.0.1:9876")  # This is the drone's local Rerun server port)

        #rr.init("CUT3R_3D_Reconstruction", spawn=True, address="YOUR_LOCAL_MACHINE_IP", port=9876)

        rr.log("description", rr.TextDocument("# CUT3R Real-time 3D Reconstruction\nContinuous 3D scene reconstruction using CUT3R", media_type=rr.MediaType.MARKDOWN))
        
        # Declare parameters
        self.declare_parameter('cut3r_path', '/home/race13/CUT3R')
        self.declare_parameter('model_path', '/home/race13/CUT3R/src/cut3r_512_dpt_4_64.pth')
        self.get_logger().info(f"🔹 Model path being used: {self.get_parameter('model_path').value}")

        self.declare_parameter('publish_current_pointcloud', True)
        self.declare_parameter('publish_aggregated_pointcloud', True)
        self.declare_parameter('publish_depth_map', True)
        
        cut3r_path = self.get_parameter('cut3r_path').value
        model_path = self.get_parameter('model_path').value
        
        # Add CUT3R paths to sys.path
        sys.path.append(cut3r_path)
        sys.path.append(os.path.join(cut3r_path, 'src'))
        sys.path.append(os.path.join(cut3r_path, 'src', 'dust3r'))
        
        # Import CUT3R modules
        try:
            from dust3r.model import ARCroco3DStereo
            from dust3r.inference import inference
            self.get_logger().info("Successfully imported CUT3R modules")
        except ImportError as e:
            self.get_logger().error(f"Failed to import CUT3R modules: {e}")
            return
        
        # ROS communication setup
        self.subscription = self.create_subscription(
            Image,
            '/race13/color/image_raw',
            self.image_callback,
            10)
        
        # Publishers (keep for ROS compatibility)
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
        self.persistent_state = None
        self.accumulated_points = []
        self.previous_frame = None
        
        # Rerun visualization setup
        self.setup_rerun_scene()
        
        self.get_logger().info(f"Initialized CUT3R Processor with Rerun visualization")
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
    
    def setup_rerun_scene(self):
        """Setup Rerun scene structure and coordinate frames"""
        # Set up coordinate frames
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
        rr.log("world/camera", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
        
        # Log camera intrinsics (adjust based on your camera)
        rr.log("world/camera/image", rr.Pinhole(
            resolution=[224, 224],
            focal_length=[112.0, 112.0],  # Adjust based on your camera
            principal_point=[112.0, 112.0]
        ), timeless=True)
        
        # Set up initial view
        rr.log("world/reconstruction", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
    
    def image_callback(self, msg):
        """Process single frame continuously with Rerun logging"""
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        
        # Set timeline for this frame
        rr.set_time_sequence("frame", self.frame_count)
        
        # Log input image to Rerun
        rr.log("world/camera/image", rr.Image(cv_image))
        
        # Process with CUT3R
        self.process_continuous_frame(cv_image)
    
    def process_continuous_frame(self, current_image):
        """CUT3R continuous processing with Rerun visualization"""
        with torch.no_grad():
            # Prepare current frame
            current_frame = self.prepare_frame(current_image)
            
            if self.previous_frame is None:
                # Initialize with first frame
                self.previous_frame = current_frame
                self.get_logger().info("Initialized CUT3R with first frame")
                return
            
            # CUT3R continuous updating
            try:
                # Create view pair for inference
                views = [self.previous_frame, current_frame]
                outputs, state_args = inference(views, self.model, self.device)
                
                # Update persistent state
                self.update_persistent_state(state_args)
                
                # Extract and process results
                if 'pred' in outputs and len(outputs['pred']) >= 2:
                    current_pred = outputs['pred'][1]  # Current view prediction
                    
                    # Visualize with Rerun
                    self.visualize_with_rerun(current_pred, current_image)
                    
                    # Publish ROS messages (keep for compatibility)
                    self.publish_current_point_cloud(current_pred)
                    self.publish_depth_map(current_pred)
                    
                    # Accumulate for dense reconstruction
                    if self.frame_count % 5 == 0:
                        self.publish_accumulated_point_cloud(current_pred)
                
                # Update for next iteration
                self.previous_frame = current_frame
                self.frame_count += 1
                
            except Exception as e:
                self.get_logger().error(f"CUT3R continuous processing failed: {e}")
    
    def visualize_with_rerun(self, pred, current_image):
        """Visualize CUT3R results with Rerun"""
        try:
            # Extract points from CUT3R prediction
            if 'pts3d' in pred:
                points = pred['pts3d'].squeeze().cpu().numpy()
            elif 'pts3d_in_other_view' in pred:
                points = pred['pts3d_in_other_view'].squeeze().cpu().numpy()
            else:
                return
            
            self.get_logger().info(f"Logging image to Rerun, shape: {current_image.shape}, dtype: {current_image.dtype}")
            print("Image shape:", current_image.shape, "dtype:", current_image.dtype)


            # Ensure proper shape
            if points.ndim == 1:
                points = points.reshape(-1, 3)
            elif points.ndim > 2:
                points = points.reshape(-1, 3)
            
            self.get_logger().info(f"Logging {points.shape[0]} points to Rerun, min: {np.min(points)}, max: {np.max(points)}")
            print("Points shape:", points.shape, "min/max:", np.min(points), np.max(points))


            # Log current frame point cloud
            rr.log("world/reconstruction/current_points", rr.Points3D(
                positions=points,
                colors=[0.2, 0.8, 0.2],  # Green for current frame
                radii=0.01
            ))
            
            # Log camera pose (simplified - you may want to extract actual pose from CUT3R)
            camera_position = np.array([0.0, 0.0, float(self.frame_count) * 0.1])  # Moving camera
            rr.log("world/camera", rr.Transform3D(
                translation=camera_position,
                from_parent=True
            ))
            
            # Accumulate points for dense reconstruction visualization
            self.accumulated_points.append(points)
            
            # Log accumulated dense reconstruction
            if len(self.accumulated_points) > 1:
                all_points = np.concatenate(self.accumulated_points, axis=0)
                
                # Color points by age (older points are more blue, newer are more red)
                colors = np.zeros((len(all_points), 3))
                points_per_frame = len(points)
                
                for i, frame_points in enumerate(self.accumulated_points):
                    start_idx = i * points_per_frame
                    end_idx = start_idx + len(frame_points)
                    if end_idx > len(all_points):
                        end_idx = len(all_points)
                    
                    # Color gradient from blue (old) to red (new)
                    age_ratio = i / len(self.accumulated_points)
                    colors[start_idx:end_idx] = [age_ratio, 0.3, 1.0 - age_ratio]
                
                rr.log("world/reconstruction/dense_points", rr.Points3D(
                    positions=all_points,
                    colors=colors,
                    radii=0.005
                ))
            
            # Generate and log depth map
            depth_map = self.project_points_to_depth_map_current(points, (224, 224))
            rr.log("world/camera/depth", rr.DepthImage(depth_map))
            
            # Log statistics
            rr.log("stats/points_count", rr.Scalar(len(points)))
            rr.log("stats/total_points", rr.Scalar(len(np.concatenate(self.accumulated_points)) if self.accumulated_points else 0))
            rr.log("stats/frame_number", rr.Scalar(self.frame_count))
            
        except Exception as e:
            self.get_logger().error(f"Rerun visualization failed: {e}")
    
    def prepare_frame(self, image):
        """Prepare single frame for CUT3R processing"""
        frame = {
            'img': torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0,
            'true_shape': torch.tensor(image.shape[:2]).unsqueeze(0).to(self.device)
        }
        
        # Resize to model input size
        frame['img'] = interpolate(frame['img'], size=(224, 224), mode='bilinear')
        frame['true_shape'] = torch.tensor([224, 224]).to(self.device)
        
        return frame
    
    def update_persistent_state(self, state_args):
        """Update CUT3R's persistent state representation"""
        if self.persistent_state is None:
            self.persistent_state = state_args
        else:
            self.persistent_state = self.merge_states(self.persistent_state, state_args)
    
    def merge_states(self, previous_state, new_state):
        """Merge previous persistent state with new observations"""
        if isinstance(new_state, dict) and isinstance(previous_state, dict):
            merged_state = previous_state.copy()
            merged_state.update(new_state)
            return merged_state
        else:
            return new_state
    
    def reset_for_new_sequence(self):
        """Reset persistent state and Rerun visualization"""
        self.persistent_state = None
        self.previous_frame = None
        self.accumulated_points = []
        self.frame_count = 0
        
        # Clear Rerun visualization
        rr.log("world/reconstruction", rr.Clear(recursive=True))
        
        self.get_logger().info("CUT3R persistent state and Rerun visualization reset")
    
    def rotate_points(self, points):
        """Rotate points for ROS coordinate system"""
        r = Rotation.from_euler('x', -90, degrees=True)
        rotated_points = r.apply(points)
        return rotated_points
    
    def project_points_to_depth_map_current(self, points, image_shape):
        """Project 3D points to depth map"""
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
    
    def publish_current_point_cloud(self, pred):
        """Publish current frame point cloud (ROS compatibility)"""
        if not self.get_parameter('publish_current_pointcloud').value:
            return
        
        try:
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
            rotated_points = self.rotate_points(points)
            
            header = std_msgs.msg.Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "map"
            
            pc2_msg = point_cloud2.create_cloud_xyz32(header, rotated_points)
            self.current_pointcloud_publisher.publish(pc2_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish current point cloud: {e}")
    
    def publish_accumulated_point_cloud(self, pred):
        """Publish accumulated dense reconstruction (ROS compatibility)"""
        if not self.get_parameter('publish_aggregated_pointcloud').value:
            return
        
        try:
            if len(self.accumulated_points) > 1:
                accumulated_point_cloud = np.concatenate(self.accumulated_points, axis=0)
                accumulated_point_cloud = accumulated_point_cloud.astype(np.float64)
                aggregated_rotated_points = self.rotate_points(accumulated_point_cloud)
                
                header = std_msgs.msg.Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = "map"
                
                pc2_msg = point_cloud2.create_cloud_xyz32(header, aggregated_rotated_points)
                self.pointcloud_publisher.publish(pc2_msg)
                
        except Exception as e:
            self.get_logger().error(f"Failed to publish accumulated point cloud: {e}")
    
    def publish_depth_map(self, pred):
        """Publish depth map (ROS compatibility)"""
        if not self.get_parameter('publish_depth_map').value:
            return
        
        try:
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
            
            depth_map = self.project_points_to_depth_map_current(points, (224, 224))
            depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_map.astype(np.float32), encoding="32FC1")
            self.depth_map_publisher.publish(depth_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish depth map: {e}")

def main(args=None):
    rclpy.init(args=args)
    cut3r_processor = CUT3RProcessor()
    
    try:
        rclpy.spin(cut3r_processor)
    except KeyboardInterrupt:
        print("Shutting down CUT3R processor...")
    finally:
        # Clean shutdown
        cut3r_processor.destroy_node()
        rr.disconnect()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
