import cv2
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge

def render_video_from_rosbag(bag_path, topic_name, output_file, fps=30):
    # ROS2 reader
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    bridge = CvBridge()
    writer = None

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == topic_name:
            # Convert ROS image message to OpenCV
            img_msg = bridge.deserialize(data, "sensor_msgs/msg/Image")
            frame = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

            if writer is None:
                h, w, _ = frame.shape
                writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            writer.write(frame)

    if writer:
        writer.release()
    print(f"Video saved at {output_file}")
