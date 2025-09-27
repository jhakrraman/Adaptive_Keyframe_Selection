It contains files to run ros package to perform reconstruction in a dynamic environment using the CUT3R model. It has basically three files that have these issues:

cut3r_cycle_working.py Frame transformation issue for the aggregated point cloud. 
Issue: As the points should be sequentially added, one after the other, and not one on the other. 
Solution: It has been fixed in the cut3r_package_working.py

cut3r_package_working.py 
Issue: As the points are added subsequently, making a 360-degree view, but here, the camera pose used for the aggregation of the point cloud is incorrect, and it's manually constructing the camera pose instead of the real-time camera pose. But it should not do that; instead, it should utilize camera poses from the frames for reconstruction when the scene changes for the aggregated_pointclouds. It should continuously do the reconstruction, respective of the scene/frame change in the environment for the aggregated_pointclouds. 
Solution: Do the scene reconstruction into the aggregated_pointcloud only when the scene changes, and identify that either based on camera angle or by scene dynamics.

cut3r_package_improved.py 
Issue: It is now only aggregating the current point cloud when there is a change in the camera pose. It is now subscribing to vehicle_altitude, vehicle_odometry, and imu, as per the code, but it is not able to get the subscribed ROS topics from the real sense camera, due to which, it is not making any accumulated_pointcloud for the continuous scene reconstruction.
Solution: It needs to subscribe to the current ros topics and utilize them for the continuous reconstruction. 
