#!/usr/bin/env python
""" Simple occupancy-grid-based mapping without localization. 

Subscribed topics:
/scan

Published topics:
/map 
/map_metadata

"""
import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import LaserScan

import numpy as np
import math

from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class Map(object):
    """ 
    The Map class stores an occupancy grid as a two dimensional
    numpy array. 
    
    Public instance variables:

        width      --  Number of columns in the occupancy grid.
        height     --  Number of rows in the occupancy grid.
        resolution --  Width of each grid square in meters. 
        origin_x   --  Position of the grid cell (0,0) in 
        origin_y   --  the map coordinate system.
        grid       --  numpy array with height rows and width columns.
        
    
    Note that x increases with increasing column number and y increases
    with increasing row number. 
    """


    def __init__(self, origin_x = -2.15, origin_y = -2.0, resolution = 0.01, 
                 width = 600, height = 600):
        """ Construct an empty occupancy grid.
        
        Arguments: origin_x, 
                   origin_y  -- The position of grid cell (0,0) in the
                                map coordinate frame.
                   resolution-- width and height of the grid cells 
                                in meters.
                   width, 
                   height    -- The grid will have height rows and width
                                columns cells.  width is the size of
                                the x-dimension and height is the size
                                of the y-dimension.
                                
         The default arguments put (0,0) in the center of the grid. 
                                
        """
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.resolution = resolution
        self.width = width 
        self.height = height 
        self.grid = np.zeros((height, width))
        
        # For the length of the occupany grid, do the following
        for i in range(width):
            # For the height of the occupancy grid, do the following
            for j in range(height):
                # Draw the initial occupancy to the map that is the occupancy grid
                self.grid[i, j] = 0.5


    def to_message(self):
        """ Return a nav_msgs/OccupancyGrid representation of this map. """
     
        grid_msg = OccupancyGrid()

        # Set up the header.
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "map"

        # .info is a nav_msgs/MapMetaData message. 
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.width
        grid_msg.info.height = self.height
        
        # Rotated maps are not supported... quaternion represents no
        # rotation. 
        grid_msg.info.origin = Pose(Point(self.origin_x, self.origin_y, 0),
                               Quaternion(0, 0, 0, 1))

        # Flatten the numpy array into a list of integers from 0-100.
        # This assumes that the grid entries are probalities in the
        # range 0-1. This code will need to be modified if the grid
        # entries are given a different interpretation (like
        # log-odds).
        
        flat_grid = self.grid.reshape((self.grid.size,)) * 100
        #print('-----------')
        
        #print (flat_grid)
        flat_grid = flat_grid.astype('int8')
      
        grid_msg.data = list(np.round(flat_grid))
       
        return grid_msg


class Mapper(object):
    """ 
    The Mapper class creates a map from laser scan data.
    """
    
    def __init__(self):
        """ Start the mapper. """

        rospy.init_node('mapper')
        self._map = Map()
        
        # Setting the queue_size to 1 will prevent the subscriber from
        # buffering scan messages.  This is important because the
        # callback is likely to be too slow to keep up with the scan
        # messages. If we buffer those messages we will fall behind
        # and end up processing really old scans.  Better to just drop
        # old scans and always work with the most recent available.
        rospy.Subscriber('scan',
                         LaserScan, self.scan_callback, queue_size=1)

        rospy.Subscriber('odom',
                         Odometry, self.odom_callback, queue_size=1)

        # Latched publishers are used for slow changing topics like
        # maps. Data will sit on the topic until someone reads it. 
        self._map_pub = rospy.Publisher('map', OccupancyGrid, latch=True, queue_size=1)
        self._map_data_pub = rospy.Publisher('map_metadata', MapMetaData, latch=True, queue_size=1)
        
        rospy.spin()


    def odom_callback(self, msg):
        # Initialise the odometry variables
        global roll, pitch, yaw, pos
        
        orientation_q = msg.pose.pose.orientation
        #print(msg.pose.pose.position) # x, y, z position
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        # print("Yaw (heading): ", yaw)
        pos = msg.pose.pose.position
        # print("Position: ", pos)


    def boundary_correction(self, value_to_correct, minimum_value, maximum_value):
        # If the value to be corrected is divisble by one and leaves no remainder, do the following
        if value_to_correct % 1 == 0:
            # While the value to be corrected is outside of the designated boundaries, do the following
            while value_to_correct > maximum_value or value_to_correct < minimum_value:
                # If the value to be corrected is greater than the maximum bound value, do the following
                if value_to_correct > maximum_value:
                    # Adjust the value to be corrected to the boundaries designated
                    value_to_correct += (minimum_value - maximum_value) - 1
                
                # Else if the value to be corrected is smaller than the minimum bound value, do the following
                elif value_to_correct < minimum_value:
                    # Adjust the value to be corrected to the boundaries designated
                    value_to_correct += (maximum_value - minimum_value) + 1
            
            # Return the corrected value
            return value_to_correct

        # Else if the value to be corrected is divisble by one and leaves a remainder, do the following
        else:
            # Increment the value to be corrected by the value of '1'
            old_value = value_to_correct + 1
            
            # While the value to be corrected is not equal to its prior value, do the following
            while value_to_correct != old_value:
                # Store the value of the value to be corrected
                old_value = value_to_correct
                
                # If the value to be corrected is smaller than the minimum bound value, do the following
                if value_to_correct < minimum_value:
                    # Adjust the value to be corrected to the boundaries designated
                    value_to_correct = (maximum_value - minimum_value) - value_to_correct
                
                # Else if the value to be corrected is greater than the maximum bound value, do the following
                elif value_to_correct > maximum_value:
                    # Adjust the value to be corrected to the boundaries designated
                    value_to_correct = (minimum_value + value_to_correct) - maximum_value
            
            # Return the corrected value
            return value_to_correct


    def bresenham_line_algorithm(self, robot_position, obstacle_position):
        # Based upon Bresenham's line algorithm, based upon Jack Elton Bresenham who developed it in 1962 at IBM 
        # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
        
        # Intialise the intersection variables
        robot_x, robot_y = robot_position
        obstacle_x, obstacle_y = obstacle_position
        difference_x = obstacle_x - robot_x
        difference_y = obstacle_y - robot_y
        
        steep_gradient = None
        position_values_swapped = False
        
        point = None
        intersecting_points = []

        # If the difference in the points Y axis is greater than the difference in the points X axis, do the following
        if abs(difference_y) > abs(difference_x):
            # The gradient of the intersecting line between the points is steep
            steep_gradient = True

        # Else if the difference in the points X axis is greater than the difference in the points Y axis, do the following
        else:
            # The gradient of the intersecting line between the points is not steep
            steep_gradient = False

        # If the gradient of the intersecting line between the points is steep, do the following
        if steep_gradient == True:
            # Rotate the intersecting line, invert the X and Y values of both points
            robot_x, robot_y = robot_y, robot_x
            obstacle_x, obstacle_y = obstacle_y, obstacle_x

        # The positions of the robot and detected obstacle have not been swapped
        position_values_swapped = False
        
        # If the robots X axis value is greater than the obstacles X axis value, do the following
        if robot_x > obstacle_x:
            # Swap the X and Y axis values of the robot for the obstacles, vice versa
            robot_x, obstacle_x = obstacle_x, robot_x
            robot_y, obstacle_y = obstacle_y, robot_y
            
            # The positions of the robot and detected obstacle have been swapped
            position_values_swapped = True

        # Recalculate the X and Y axis value differentials
        difference_x = obstacle_x - robot_x
        difference_y = obstacle_y - robot_y

        # Calculate the X axis differential error 
        x_axis_error = int(difference_x * 0.5)
        
        # If the robots Y axis value is greater than the obstacles Y axis value (robot above obstacle in the space), do the following
        if robot_y < obstacle_y:
            # The Y axis step is positively advanced, for the cells intersecting the robot and obstacle (up)
            y_axis_step = 1 

        # Else if the robots Y axis value is smaller than the obstacles Y axis value (robot below obstacle in the space), do the following
        else: 
            # The Y axis step is negatively advanced, for the cells intersecting the robot and obstacle (down)
            y_axis_step = -1

        # Start the intersection from the robots Y axis value
        y_value = robot_y
        
        # Iterate over the cells contained within the bounding box, to generate points between the robot and the obstacle
        # For the X axis distance between the robot and obstacle positions, do the following
        for x_value in range(robot_x, obstacle_x + 1):
            # If the gradient of the intersecting line between the points is steep, do the following 
            if steep_gradient == True:
                # The points X and Y values are swapped by index
                point = (y_value, x_value)
            
            # Else if the gradient of the intersecting line between the points is not steep, do the following
            else:
                # The points X and Y values are mantained by index
                point = (x_value, y_value)
            
            # Append the point to the intersecting points array
            intersecting_points.append(point)
            # Reduce the error calculated in the X axis (traverse along the X axis to become closer to the obstacles X position overtime)
            x_axis_error -= abs(difference_y)

            # If the error calculated in the X axis submerges below '0' (traverse along the Y axis to become closer the obstacles Y position overtime), do the following
            if x_axis_error < 0:
                # Step along the Y axis in the direction that nears the robot to the obstacles position (next cell)
                y_value += y_axis_step
                # Increase the error calculated in the X axis to enable further traversal in the X axis
                # e.g. robot at (0, 0) and obstacle at (5, 5): (1, 0) -> (1, 1) -> (2, 1) -> (2, 2) -> etc 
                x_axis_error += difference_x 

        # The positions of the robot and detected obstacle have not been swapped
        if position_values_swapped:
            # Reverse the list if the coordinates were swapped
            intersecting_points.reverse()
        
        # Return the points intersecting the robot and obstacle positions
        return intersecting_points


    def find_intersection_cells(self, obstacle_position, obstacle_grid, robot_position, robot_grid, sensor_offset):
        # Initialise the intersection global coordinates variables
        obstacle_x, obstacle_y = obstacle_position
        robot_x, robot_y = robot_position
        
        # Initialise the intersection grid variables
        obstacle_w, obstacle_h = obstacle_grid
        robot_w, robot_h = robot_grid
        
        # Store all of the grid cells intersecting the robot and obstacles position
        intersecting_points = self.bresenham_line_algorithm(robot_grid, obstacle_grid)
        
        # For all the intersecting cells between the robots position and the obstacles position, do the following
        for i in range(len(intersecting_points)):
            # Temporarily store the X and Y indices of the iterated intersecting point, relative to the occupancy grid array
            intersecting_point = [intersecting_points[i][0], intersecting_points[i][1]]

            # If the iterated intersecting cells indexes are not equal to either the robots position nor the obstacles position, do the following
            if intersecting_point != obstacle_grid or intersecting_point != robot_grid:
                # Restore the intersecting cell indexes to their coordinate equivalent (centre point of each cell)
                cell_w = int((intersecting_points[i][0] - self._map.origin_x + 1) / self._map.resolution)
                cell_h = self._map.height - int((intersecting_points[i][1] - self._map.origin_y + 1) / self._map.resolution)
                
                # Calculate the vector (find direction) between the position of the iterated intersecting cell and the position of the robot 
                #cell_robot_direction = math.degrees(math.atan2(cell_y - robot_y, cell_x - robot_x))
                cell_robot_direction = math.degrees(math.atan2(cell_h - robot_h, cell_w - robot_w))
        
                # Convert the sensor offset of the robot from radians to degrees
                sensor_angle = math.degrees(sensor_offset)
                
                # If the sensor passes through the centre of the intersecting cell (no error or offset), do the following
                if cell_robot_direction == sensor_angle:
                    # Draw the intersection point to the map that is the occupancy grid
                    self.bayes_theorem(intersecting_points[i][0], intersecting_points[i][1], 0.0)
                
                # Else if the sensor does not pass through the centre of the intersecting cell (there is an error and offset), do the following
                else:
                    # Calculate the half extent from the centre of a grid cell
                    half_extent = self._map.resolution / 2

                    # Calculate the vector (find direction) between the position of the iterated intersecting cells corner vertices and the position of the robot 
                    cell_top_left_corner        = abs(math.degrees(math.atan2((cell_h + half_extent) - robot_h, (cell_w - half_extent) - robot_w)))
                    cell_bottom_left_corner     = abs(math.degrees(math.atan2((cell_h - half_extent) - robot_h, (cell_w - half_extent) - robot_w)))
                    cell_top_right_corner       = abs(math.degrees(math.atan2((cell_h + half_extent) - robot_h, (cell_w + half_extent) - robot_w)))
                    cell_bottom_right_corner    = abs(math.degrees(math.atan2((cell_h - half_extent) - robot_h, (cell_w + half_extent) - robot_w)))
                   
                    # Console output
                    #print(cell_top_left_corner, cell_bottom_left_corner, cell_top_right_corner, cell_bottom_right_corner)
                   
                    # Calculate the angular error between the sensors offset and the direction of the obstacle from the robots position
                    direction_error = sensor_angle - cell_robot_direction
                    # Correct the directional error to the boundaries of '0' and '360' degrees
                    direction_error = self.boundary_correction(direction_error, 0, 2 * math.pi)
                    #print(sensor_angle, cell_robot_direction, direction_error)

                    # Calculate the largest error from all the corner vertices of the intersecting cell and its centre
                    maximum_direction_error =     max(cell_top_left_corner, cell_bottom_left_corner, cell_top_right_corner, cell_bottom_right_corner)
                    # Calculate the smallest error from all the corner vertices of the intersecting cell and its centre
                    minimum_direction_error = abs(min(cell_top_left_corner, cell_bottom_left_corner, cell_top_right_corner, cell_bottom_right_corner))

                    # If the angular error in direction is positive, do the following
                    if direction_error > 0:
                        # If the angular error is smaller than or equal to the maxiumum direction error, do the following
                        if direction_error <= maximum_direction_error:
                            # Draw the intersection point to the map that is the occupancy grid
                            self.bayes_theorem(intersecting_points[i][0], intersecting_points[i][1], (direction_error / maximum_direction_error) * 0.5)

                    # Else if the angular error in direction is negative, do the following
                    elif direction_error < 0:
                        # If the angular error is larger than or equal to the minimum direction error, do the following
                        if direction_error >= minimum_direction_error:
                            # Draw the intersection point to the map that is the occupancy grid
                            self.bayes_theorem(intersecting_points[i][0], intersecting_points[i][1], (direction_error / minimum_direction_error) * 0.5)

                    # Console output
                    #print("MIN {0}  MAX {1}".format((direction_error / minimum_direction_error) * 0.5, (direction_error / maximum_direction_error) * 0.5))


    def bayes_theorem(self, position_x, position_y, updated_probability):
        # Recursive Bayes Theorem

        # If the position resides within the boundaries of the map, do the following
        if position_x >= 0 and position_x < self._map.width and position_y >= 0 and position_y < self._map.height:
            # Store the posterior probability of the occupancy grid cell (prior probability)
            prior_probability = self._map.grid[position_x, position_y]

            # If the prior probability of the occupancy grid cell is smaller than or equal to '0.1' (remove noise - incorrect object detections), do the following
            #if prior_probability == 1.0:
                # Set the prior probability of the occupancy grid cell to '0.5' (uncertain their is an object there - allows cell to be updated)
                #prior_probability = 0.5

            # Calculate the numerator and denominator components of recursive Bayes Theorem
            numerator = updated_probability * prior_probability
            denominator = updated_probability * prior_probability + (1 - updated_probability) * (1 - prior_probability)

            # Calculate the probability of interest
            probability_of_interest = numerator / denominator
            
            # Draw the probability of interest to the map that is the occupancy grid
            self._map.grid[position_x, position_y] = probability_of_interest


    def scan_callback(self, scan):
        # Initialise the variables
        #turtlebot_radius = 0.06

        # For all of the robots sensors, do the following
        for i in range(len(scan.ranges)):
        #for i in range(1):
            # If the currently iterated sensor is offset by '5' degrees from either side of the robots heading (10 degree cone), do the following
            if i < 5 or i > 354:
                # If the currently iterated sensor has detected an object, do the following
                if scan.ranges[i] > scan.range_min and scan.ranges[i] < scan.range_max:
                    # Calculate the the angular offset of the sensor
                    heading_offset = math.radians(i)
                    sensor_offset = yaw + heading_offset
                    
                    # If the sensor detects an object, do the following
                    if scan.ranges[i] != np.Inf:
                        # Calculate the objects distance from the robots centre
                        object_distance = scan.ranges[i] #+ turtlebot_radius
                    
                    # Else if the sensor does not detect an object, do the following
                    else:
                        # Calculate the objects distance from the robots centre
                        object_distance = scan.range_max #+ turtlebot_radius
                    
                    # Find the X and Y positions of the object
                    position_x = math.cos(heading_offset) * object_distance
                    position_y = math.sin(heading_offset) * object_distance

                    # Rotate the X and Y positions into the global coordinate system
                    translate_x = position_x * math.cos(yaw) - position_y * math.sin(yaw)
                    translate_y = position_x * math.sin(yaw) + position_y * math.cos(yaw)
                
                    # Translate the X' and Y' positions into the global coordinate system
                    obstacle_x = translate_x + pos.x
                    obstacle_y = translate_y + pos.y

                    # Calculate the position of the obstacle relative to the occupancy grid array
                    point_x = int((obstacle_x - self._map.origin_x + 1) / self._map.resolution)
                    point_y = self._map.height - int((obstacle_y - self._map.origin_y + 1) / self._map.resolution)

                    # Calculate the position of the robot relative to the occupancy grid array
                    robot_x = int((pos.x - self._map.origin_x + 1) / self._map.resolution)
                    robot_y = self._map.height - int((pos.y - self._map.origin_y + 1) / self._map.resolution)

                    # Reformat the robot and obstacle position values 
                    robot_position = [pos.x, pos.y]
                    obstacle_position = [obstacle_x, obstacle_y]

                    # Reformat the robot and obstacle grid coordinates
                    robot_grid = [robot_x, robot_y]
                    obstacle_grid = [point_x, point_y]

                    # Fill intermediate cells between the robots position and a detected obstacle
                    self.find_intersection_cells(obstacle_position, obstacle_grid, robot_position, robot_grid, sensor_offset)
                    
                    # Draw the robots position to the map that is the occupancy grid
                    self.bayes_theorem(robot_x, robot_y, 0.0)

                    # Draw the obstacles position to the map that is the occupancy grid
                    self.bayes_theorem(point_x, point_y, 1.0)

                    # Console output
                    #print("===================================================================================================================================================")
                    #print("Robot position: [{0}, {1}]   Object position: [{2}, {3}]".format(robot_x, robot_y, point_x, point_y))
                        
        # Now that the map was updated, so publish it!
        rospy.loginfo("Scan is processed, publishing updated map.")
        self.publish_map()


    def publish_map(self):
        """ Publish the map. """
        grid_msg = self._map.to_message()
        self._map_data_pub.publish(grid_msg.info)
        self._map_pub.publish(grid_msg)


if __name__ == '__main__':
    try:
        m = Mapper()
    except rospy.ROSInterruptException:
        pass

"""
https://w3.cs.jmu.edu/spragunr/CS354_S14/labs/mapping/mapper.py
https://www.youtube.com/watch?v=K1ZFkR4YsRQ
"""