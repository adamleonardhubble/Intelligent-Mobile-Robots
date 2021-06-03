from geometry_msgs.msg import Pose, PoseArray, Quaternion, PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.cluster.hierarchy import linkage, fcluster, fclusterdata
from util import rotateQuaternion, getHeading
from pf_base import PFLocaliserBase
from time import time
import numpy as np
import random
import rospy
import math
import copy

class PFLocaliser(PFLocaliserBase):
    
    # Initializing Parameters   
    def __init__(self):
        # Call the Superclass Constructor
        super(PFLocaliser, self).__init__()
        
        # Set motion model parameters
        self.ODOM_ROTATION_NOISE    = np.random.uniform(0.01, 0.3) # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = np.random.uniform(0.01, 0.3) # Odometry model x axis (forward) noise
        self.ODOM_DRIFT_NOISE       = np.random.uniform(0.01, 0.3) # Odometry model y axis (side-to-side) noise

        # Sensor model parameters
        self.PARTICLE_COUNT = 200 # Count of particles comprising the particle filter that is used to localise the robot in its environment
        self.NUMBER_PREDICTED_READINGS = 90 # Number of laser sensor scan readings (observations) used to be compared with predictions when computing particle weights

        # Noise parameters for the particles comprising the particle cloud
        self.PARTICLE_POSITIONAL_NOISE = np.random.uniform(75, 100) # Particle cloud particle positional noise - Gaussian standard deviation (particle position spread)
        self.PARTICLE_ANGULAR_NOISE    = np.random.uniform(1, 120) # Particle cloud particle angular noise - von Mises standard deviation (particle rotation spread)

        # Pose estimation clustering parameters
        # Pose estimation techniques available: global mean, best particle, hac clustering
        self.POSE_ESTIMATE_TECHNIQUE = "hac clustering" # Active pose estimation technique used to appropriate the position and orientation of the robot
        self.CLUSTER_DISTANCE_THRESHOLD = 0.35 # Mean distance threshold between clusters, applied when forming flat clusters


    def roulette_wheel_selection(self, probability_weights, cumulative_probability_weight):
        # Roulette-Wheel algorithm - select particles that are weighted highly (high probability of belief) 
        # for resampling better particles that converge more-closely to the actual pose of the robot overtime
        # Probability selection technique overview: http://www.edc.ncl.ac.uk/highlight/rhjanuary2007g02.php
        pose_array = PoseArray() # Instantiate a pose array object

        # For all particles comprising the particle cloud, do the following
        for _ in range(len(self.particlecloud.poses)):
            stop_criterion = random.random() * cumulative_probability_weight # Instantiate the stop criterion for the roulette wheel selection procedure, representing a percentile of the cumulative sum of probability weights to be surpassed
            
            probability_weight_sum = 0 # Instantiate the probability weight sum
            index = 0 # Instantiate the index variable (selected individual particle)

            # While the sum of probability weights is smaller than the stop criterion, do the following
            while probability_weight_sum < stop_criterion:
			    probability_weight_sum += probability_weights[index] # Add and equal the probability weight of the selected individual to the probability weight sum
			    index += 1 # Increment the index of the selected individual
            
            pose_array.poses.append(copy.deepcopy(self.particlecloud.poses[index - 1])) # Append the pose configuration of the highly-weighted selected individual (all parameters) to the pose array
        
        return pose_array # Return the pose array object


    def particle_cloud_noise(self, pose_object):
        # Depreciate and then resample noise parameter values
        # If the odometry rotational noise parameters value is larger than '0.1', do the following
        if self.ODOM_ROTATION_NOISE > 0.1:
            self.ODOM_ROTATION_NOISE -= 0.01 # Decrement the value of the parameter (converge to the pose of the robot overtime)
        
        # Else if the odometry rotational noise parameters value is not larger than '0.1', do the following
        else:
            self.ODOM_ROTATION_NOISE = np.random.uniform(0.01, 0.1) # Odometry model rotation noise
        
        # If the odometry translational noise parameters value is larger than '0.1', do the following
        if self.ODOM_TRANSLATION_NOISE > 0.1:
            self.ODOM_TRANSLATION_NOISE -= 0.01 # Decrement the value of the parameter (converge to the pose of the robot overtime)
        
        # Else if the odometry translataional noise parameters value is not larger than '0.1', do the following
        else:
            self.ODOM_TRANSLATION_NOISE = np.random.uniform(0.01, 0.1) # Odometry model x axis (forward) noise

        # If the odometry nautical noise parameters value is larger than '0.1', do the following
        if self.ODOM_DRIFT_NOISE > 0.1:
            self.ODOM_DRIFT_NOISE -= 0.01 # Decrement the value of the parameter (converge to the pose of the robot overtime)
        
        # Else if the odometry nautical noise parameters value is larger than '0.1', do the following
        else:
            self.ODOM_DRIFT_NOISE = np.random.uniform(0.01, 0.1) # Odometry model y axis (side-to-side) noise
        
        # If the particle positional noise parameters value is larger than '2', do the following
        if self.PARTICLE_POSITIONAL_NOISE > 2.0:
            self.PARTICLE_POSITIONAL_NOISE -= 0.1 # Decrement the value of the parameter (converge to the pose of the robot overtime)
        
        # Else if the particle positional noise parameters value is not larger than '2', do the following
        else:
            self.PARTICLE_POSITIONAL_NOISE = np.random.uniform(0.01, 2) # Particle cloud particle positional noise - Gaussian standard deviation (particle position spread)

        # If the particle angular noise parameters value is larger than '90', do the following
        if self.PARTICLE_ANGULAR_NOISE > 90.0:
            self.PARTICLE_ANGULAR_NOISE -= 1.0 # Decrement the value of the parameter (converge to the pose of the robot overtime)
        
        # Else if the particle positional noise parameters value is not larger than '90', do the following
        else:
           self.PARTICLE_ANGULAR_NOISE = np.random.uniform(0.01, 90) # Particle cloud particle angular noise - von Mises standard deviation (particle rotation spread) 

        # Add positional noise to the x and y coordinate of the particle in the pose object
        pose_object.position.x += random.gauss(0, self.PARTICLE_POSITIONAL_NOISE) * self.ODOM_TRANSLATION_NOISE
        pose_object.position.y += random.gauss(0, self.PARTICLE_POSITIONAL_NOISE) * self.ODOM_DRIFT_NOISE
        
        # Add circular distribution noise (von Mises distribution - Gaussian-based sampling from circular-distributed data)
        # e.g. ([0, 270 (90 * 3 standard deviations of the mean)] - 180) * [0.01, 0.1] = [0, 9] (highest)
        angular_displacement_noise = (random.vonmisesvariate(0, self.PARTICLE_ANGULAR_NOISE) - math.pi) * self.ODOM_ROTATION_NOISE

        # Add rotational noise to the orientation (heading) of the particle in the pose object
        pose_object.orientation = rotateQuaternion(pose_object.orientation, angular_displacement_noise)
        
        return pose_object # Return the pose object of the current particle              

  
    def initialise_particle_cloud(self, initial_pose):
        """
        Called whenever an initial_pose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        
        :Args:
            | initial_pose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """

        pose_array = PoseArray() # Instantiate a pose array object

        # For the count of particles comprising the particle cloud, do the following
        for _ in range(self.PARTICLE_COUNT):                            
            pose_object = Pose() # Instantiate a pose object                                

            # Normally sample and calculate the positional noise of the robots initial observation for the x and y coordinate values
            positional_noise_x = random.gauss(0, self.PARTICLE_POSITIONAL_NOISE) * self.ODOM_TRANSLATION_NOISE
            positional_noise_y = random.gauss(0, self.PARTICLE_POSITIONAL_NOISE) * self.ODOM_DRIFT_NOISE
            
            # Add positional noise to the x and y coordinate values of the particle in the pose object
            pose_object.position.x = initial_pose.pose.pose.position.x + positional_noise_x
            pose_object.position.y = initial_pose.pose.pose.position.y + positional_noise_y

            # Add circular distribution noise (von Mises distribution - Gaussian-based sampling from circular data)
            # e.g. ([0, 270 (90 * 3std)] - 180) * [0.01, 0.1] = [0, 9] (highest)
            angular_displacement_noise = (random.vonmisesvariate(0, self.PARTICLE_ANGULAR_NOISE) - math.pi) * self.ODOM_ROTATION_NOISE
            
            # Add rotational noise to the orientation (heading) of the particle in the pose object
            pose_object.orientation = rotateQuaternion(initial_pose.pose.pose.orientation, angular_displacement_noise)

            pose_array.poses.append(pose_object) # Append the pose object to the pose array object
            
        return pose_array # Return the pose array object
 

    def update_particle_cloud(self, scan): # Function resamples based on the weights acquired by particles which is obtained from the sensor model   
        """
        This should use the supplied laser scan to update the current
        particle cloud. 
        
        i.e. self.particlecloud should be updated
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update
        """
        
        global latest_scan # Instantiate a global laser scan variable
        latest_scan = scan # Store the current observation data of the robot

        probability_weights = [] # Instantiate the probability weights array variable
        cumulative_probability_weight = 0 # Instantiate the cumulative probability weight variable                                                 
        
        # For each pose object (particle pose) in the particle cloud, do the following
        for pose_object in self.particlecloud.poses:
            probability_weight = self.sensor_model.get_weight(scan, pose_object) # Store the probability weight of the current particle accurately representing the current pose of the robot
            probability_weights.append(probability_weight) # Append the probability weight calculated for the current particle to the probability weights array variable
            cumulative_probability_weight += probability_weight # Add and equal the probability weight of the current particle to the cumulative probability weights variable

        # Task: resample the particle cloud by creating particles with a new position and orientation depending on the probabilites of the particles 
        # in the old particle cloud, using Roulette-Wheel selection to choose highly-weighted particles more often than lowly-weighted particles (converge to the current pose of the robot overtime)                      
        pose_array = self.roulette_wheel_selection(probability_weights, cumulative_probability_weight) # Resample the particles comprising the particle cloud that are believed to be the most accurate representations of the robots current pose
        
        # For each pose object (particle pose) in the pose array, do the following
        for pose_object in pose_array.poses:
            pose_object = self.particle_cloud_noise(pose_object) # Apply noise to the pose (robots observation) of the current particle in the pose array
        
        self.particlecloud = pose_array # Update the particle cloud object
        
        """
        Output the estimated position and orientation of the robot
        """
        robot_estimated_position = self.estimatedpose.pose.pose.position # Store the estimated position of the robots current pose
        robot_estimated_orientation = self.estimatedpose.pose.pose.orientation # Store the estimated orientation of the robots current pose
    
        # Cast the robots estimated orientation from quaternion to euler notation
        orientation_list = [robot_estimated_orientation.x, robot_estimated_orientation.y, robot_estimated_orientation.z, robot_estimated_orientation.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)

        # Output the robots position and orientation estimates (estimated pose)
        print("/estimatedPose: Robot Position: [{x:.2f}, {y:.2f}]    Robot Orientation: [{yaw:.2f}]".format(
                                                                                                    x=robot_estimated_position.x, 
                                                                                                    y=robot_estimated_position.y, 
                                                                                                    yaw=math.degrees(yaw)))


    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
        """

        # If the active pose estimation technique is global mean, do the following
        if self.POSE_ESTIMATE_TECHNIQUE == "global mean":
            estimated_pose = Pose() # Instantiate a pose object
            
            # Instantiate the position and orientation cumulative sum variables
            position_sum_x, position_sum_y, orientation_sum_z, orientation_sum_w = (0 for _ in range(4))
            
            # For each pose object (particle pose) in the particle cloud, do the following
            for pose_object in self.particlecloud.poses:
                position_sum_x    += pose_object.position.x # Add and equal the x positional value of the current particles pose to the sum of x positions
                position_sum_y    += pose_object.position.y # Add and equal the y positional value of the current particles pose to the sum of y positions
                orientation_sum_z += pose_object.orientation.z # Add and equal the z rotational value of the current particles pose to the sum of z orientations
                orientation_sum_w += pose_object.orientation.w # Add and equal the w rotational value of the current particles pose to the sum of w orientations
            
            estimated_pose.position.x    = position_sum_x    / self.PARTICLE_COUNT # Set the x positional value of the estimated pose of the robot to the mean particle x positional value
            estimated_pose.position.y    = position_sum_y    / self.PARTICLE_COUNT # Set the y positional value of the estimated pose of the robot to the mean particle y positional value
            estimated_pose.orientation.z = orientation_sum_z / self.PARTICLE_COUNT # Set the z rotational value of the estimated pose of the robot to the mean particle z rotational value
            estimated_pose.orientation.w = orientation_sum_w / self.PARTICLE_COUNT # Set the w rotational value of the estimated pose of the robot to the mean particle w rotational value

        # Else if the active pose estimation technique is best particle, do the following
        elif self.POSE_ESTIMATE_TECHNIQUE == "best particle":
            estimated_pose = Pose() # Instantiate a pose object
            
            # Instantiate the position, orientation and highest belief variables
            position_x, position_y, orientation_z, orientation_w, highest_belief = (0 for _ in range(5))

            # For each pose object (particle pose) in the particle cloud, do the following
            for pose_object in self.particlecloud.poses:
                probability_weight = self.sensor_model.get_weight(latest_scan, pose_object) # Store the probability weight of the current particle representing the current pose of the robot
                
                # If the highest recorded belief of a particle in the particle cloud has not been set or is smaller than the belief of the current particle, do the following
                if highest_belief == 0 or highest_belief < probability_weight:
                    position_x    = pose_object.position.x # Store the x positional value of the current particles pose
                    position_y    = pose_object.position.y # Store the y positional value of the current particles pose
                    orientation_z = pose_object.orientation.z # Store the z rotational value of the current particles pose
                    orientation_w = pose_object.orientation.w # Store the w rotational value of the current particles pose

                    highest_belief = probability_weight # Update the highest recorded belief of the particles comprising the particle cloud

            estimated_pose.position.x    = position_x # Set the x positional value of the estimated pose of the robot to the mostly-believed particles x positional value
            estimated_pose.position.y    = position_y # Set the y positional value of the estimated pose of the robot to the mostly-believed particles y positional value
            estimated_pose.orientation.z = orientation_z # Set the z rotational value of the estimated pose of the robot to the mostly-believed particles z rotational value
            estimated_pose.orientation.w = orientation_w # Set the w rotational value of the estimated pose of the robot to the mostly-believed particles w rotational value

        # Else if the active pose estimation technique is Hierarchical Agglomerative Clustering (HAC), do the following
        elif self.POSE_ESTIMATE_TECHNIQUE == "hac clustering":
            estimated_pose = Pose()	# Instantiate a pose object
            
            # Instantiate the position and orientation array variables for forming the distance matrix
            position_x, position_y, orientation_z, orientation_w = ([] for _ in range(4))
           
            # For each pose object (particle pose) in the particle cloud, do the following
            for pose_object in self.particlecloud.poses:     
                # Formulate the components of the distance matrix used by HCA
                position_x.append(pose_object.position.x) # Append the x position of the current particles pose to the position x array                
                position_y.append(pose_object.position.y) # Append the y position of the current particles pose to the position y array                        
                orientation_z.append(pose_object.orientation.z) # Append the z orientation of the current particles pose to the orientation z array 
                orientation_w.append(pose_object.orientation.w) # Append the w orientation of the current particles pose to the orientation w array 

            position_x    = np.array(position_x) # Cast the x position array to a numpy type array                                    
            position_y    = np.array(position_y) # Cast the y position array to a numpy type array                                  
            orientation_z = np.array(orientation_z) # Cast the z orientation array to a numpy type array                                            
            orientation_w = np.array(orientation_w) # Cast the w orientation array to a numpy type array                                         
            
            # Column-wise stack the position and orientation values of each particle to formulate the distance matrix
            distance_matrix = np.column_stack((position_x, position_y, orientation_z, orientation_w))
            
            # Perform hierarchical/ agglomerative clustering on the condensed distance matrix
            # Returns the hierarchical clustering encoded as a linkage matrix
            # Linkage method overview: https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
            # Median: when two clusters are combined into a new cluster, the average of centroids (two clusters) give the new centroid 
            # Also known as the WPGMC algorithm
            linkage_matrix = linkage(distance_matrix, method='median')    
                                       
            # Cluster the particles in the particle cloud (by minimising the variance between them) and assign each particle an identity 
            # corresponding to its belonging cluster (return array of numbers representing the cluster that each particle in the particle cloud belongs to)
            particle_cluster_identities = fcluster(linkage_matrix, self.CLUSTER_DISTANCE_THRESHOLD, criterion='distance')
            #print(particle_cluster_identities) # Output the particle cluster identities (particle association to a cluster)

            cluster_count = max(particle_cluster_identities) # Identify the number of clusters formulated from the particles                         
            cluster_particle_counts = [0] * cluster_count # Instantiate a cluster particle count array variable 
            cluster_probability_weight_sums = [0] * cluster_count # Instantiate a cluster probability weight sum array variable

            # For each particle cluster in the particle cloud, do the following
            for i, particle_cluster_identity in enumerate(particle_cluster_identities):                 
                pose_object = self.particlecloud.poses[i] # Assign the particle its pose object that is stored in the particle cloud (access pose data)
                
                probability_weight = self.sensor_model.get_weight(latest_scan, pose_object) # Store the probability weight of the current particle representing the current pose of the robot

                cluster_particle_counts[particle_cluster_identity - 1] += 1 # Increment the count of particles comprising the cluster
                cluster_probability_weight_sums[particle_cluster_identity - 1] += probability_weight # Store the probability weight of the current particle in the 
                #print(particle_cluster_identity, cluster_probability_weight_sums[particle_cluster_identity - 1]) # Output the cluster probability weight sums for each cluster formulated

            # Find the cluster of particles that collectively are the most accurate in comparison to all other clusters,
            # used to more-accurately represent the current pose of the robot
            cluster_highest_belief = cluster_probability_weight_sums.index(max(cluster_probability_weight_sums)) + 1 
            cluster_highest_belief_particle_count = cluster_particle_counts[cluster_highest_belief - 1] # Store the number of particles comprising the most-accurate cluster
            #print(cluster_highest_belief, cluster_probability_weight_sums) # Output the ID of the most accurate cluster and its probability weight sum

            # Instantiate the position and orientation cumlative sum variables for the estimated poses averaging operation
            position_sum_x, position_sum_y, orientation_sum_z, orientation_sum_w = (0 for _ in range(4))

            # For each particle cluster in the particle cloud, do the following
            for i, particle_cluster_identity in enumerate(particle_cluster_identities):
                # If the current particle belongs to the cluster representing the current pose of the robot most accurately, do the following
                if (particle_cluster_identity == cluster_highest_belief):
                    pose_object = self.particlecloud.poses[i] # Assign the particle its pose object that is stored in the particle cloud (access pose data)
                    
                    position_sum_x    += pose_object.position.x # Add and equal the x positional value of the current particles pose to the sum of x positions
                    position_sum_y    += pose_object.position.y # Add and equal the y positional value of the current particles pose to the sum of y positions
                    orientation_sum_z += pose_object.orientation.z # Add and equal the z rotational value of the current particles pose to the sum of z orientations
                    orientation_sum_w += pose_object.orientation.w # Add and equal the w rotational value of the current particles pose to the sum of w orientations
                
            estimated_pose.position.x    = position_sum_x    / cluster_highest_belief_particle_count # Set the x positional value of the estimated pose of the robot to the mean particle x positional value
            estimated_pose.position.y    = position_sum_y    / cluster_highest_belief_particle_count # Set the y positional value of the estimated pose of the robot to the mean particle y positional value
            estimated_pose.orientation.z = orientation_sum_z / cluster_highest_belief_particle_count # Set the z rotational value of the estimated pose of the robot to the mean particle z rotational value
            estimated_pose.orientation.w = orientation_sum_w / cluster_highest_belief_particle_count # Set the w rotational value of the estimated pose of the robot to the mean particle w rotational value

        """
        Output the estimated position and orientation of the robot
        """
        robot_estimated_position = estimated_pose.position # Store the estimated position of the robots current pose
        robot_estimated_orientation = estimated_pose.orientation # Store the estimated orientation of the robots current pose
    
        # Cast the robots estimated orientation from quaternion to euler notation
        orientation_list = [robot_estimated_orientation.x, robot_estimated_orientation.y, robot_estimated_orientation.z, robot_estimated_orientation.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)

        # Output the robots position and orientation estimates (estimated pose)
        print("{technique}: Robot Position: [{x:.2f}, {y:.2f}]    Robot Orientation: [{yaw:.2f}]".format(
                                                                                                    technique=self.POSE_ESTIMATE_TECHNIQUE.title(),
                                                                                                    x=robot_estimated_position.x, 
                                                                                                    y=robot_estimated_position.y, 
                                                                                                    yaw=math.degrees(yaw)))

        return estimated_pose # Return the estimated pose of the robot