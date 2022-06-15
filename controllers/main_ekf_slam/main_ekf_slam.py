import math
import random
import copy
from controller import Robot, Motor, DistanceSensor, Lidar, Keyboard
import numpy as np
import collections
from time import sleep
import matplotlib.pyplot as plt

import supervisor_ekf_slam

state = "sense"  # Drive along the course
USE_ODOMETRY = False  # False for ground truth pose information, True for real odometry

# create the Robot instance.
supervisor_ekf_slam.init_supervisor()
robot = supervisor_ekf_slam.supervisor

# Map Variables
MAP_BOUNDS = [1., 1.]
CELL_RESOLUTIONS = np.array([0.1, 0.1])  # 10cm per cell
NUM_X_CELLS = int(MAP_BOUNDS[0] / CELL_RESOLUTIONS[0])
NUM_Y_CELLS = int(MAP_BOUNDS[1] / CELL_RESOLUTIONS[1])

world_map = np.zeros([NUM_Y_CELLS, NUM_X_CELLS])

# Ground Sensor Measurements under this threshold are black
# measurements above this threshold can be considered white.
GROUND_SENSOR_THRESHOLD = 600  # Light intensity units
LIDAR_SENSOR_MAX_RANGE = 5.  # 3 # Meters
LIDAR_ANGLE_BINS = 21  # 21 Bins to cover the angular range of the lidar, centered at 10
LIDAR_ANGLE_RANGE = 1.5708  # 90 degrees, 1.5708 radians

# RANSAC values
MAX_TRIALS = 1000  # Max times to run algorithm
MAX_SAMPLE = 10  # Randomly select X points
MIN_LINE_POINTS = 6  # If less than 5 points left, stop algorithm
RANSAC_TOLERANCE = 0.15  # If point is within 20 cm of line, it is part of the line
RANSAC_CONSENSUS = 6  # At least 5 points required to determine if a line

# Robot Pose Values
pose_x = 0
pose_y = 0
pose_theta = 0
left_wheel_direction = 0
right_wheel_direction = 0

# Constants to help with the Odometry update
WHEEL_FORWARD = 1
WHEEL_STOPPED = 0
WHEEL_BACKWARD = -1

# GAIN Values
theta_gain = 1.0
distance_gain = 0.3

MAX_VEL_REDUCTION = 0.25
EPUCK_MAX_WHEEL_SPEED = 0.125 * MAX_VEL_REDUCTION  # m/s
EPUCK_AXLE_DIAMETER = 0.053
EPUCK_WHEEL_RADIUS = 0.0205  # ePuck's wheels are 0.041m in diameter.

# Index into ground_sensors and ground_sensor_readings for each of the 3 onboard sensors.
LEFT_IDX = 0
CENTER_IDX = 1
RIGHT_IDX = 2
WHEEL_FORWARD = 1
WHEEL_STOPPED = 0
WHEEL_BACKWARD = -1

lidar_data = np.zeros(LIDAR_ANGLE_BINS)

step = LIDAR_ANGLE_RANGE / LIDAR_ANGLE_BINS

lidar_offsets = np.linspace(step * (LIDAR_ANGLE_BINS // 2), -step * (LIDAR_ANGLE_BINS // 2), LIDAR_ANGLE_BINS)

# EKF Vars
n = 20  # number of static landmarks
mu = []
cov = []
mu_new = []
cov_new = []

# Stored global [x, y, j] for observed landmarks to check if the landmark has been seen before (is within 0.05 cm radius of previous x, y)
landmark_globals = []

SIM_TIMESTEP = int(robot.getBasicTimeStep())
max_speed = 6.28
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)
# get and enable lidar
lidar = robot.getDevice("LDS-01")
lidar.enable(SIM_TIMESTEP)
lidar.enablePointCloud()
# Initialize lidar motors
lidar_main_motor = robot.getDevice('LDS-01_main_motor')
lidar_secondary_motor = robot.getDevice('LDS-01_secondary_motor')
lidar_main_motor.setPosition(float('inf'))
lidar_secondary_motor.setPosition(float('inf'))
lidar_main_motor.setVelocity(30.0)
lidar_secondary_motor.setVelocity(60.0)

# klawiatura
keyboard = Keyboard()
keyboard.enable(SIM_TIMESTEP)


def steruj(cmd):
    left_motor.setVelocity(cmd[0])
    right_motor.setVelocity(cmd[1])


motor_cmd = {
    ord("W"): (max_speed, max_speed),
    ord("S"): (-max_speed, -max_speed),
    ord("A"): (-max_speed / 2, max_speed / 2),
    ord("D"): (max_speed / 2, -max_speed / 2),
}


def get_bounded_theta(theta):
    '''
    Returns theta bounded in [-PI, PI]
    '''
    while theta > math.pi: theta -= 2. * math.pi
    while theta < -math.pi: theta += 2. * math.pi
    return theta


def convert_lidar_reading_to_world_coord(lidar_bin, lidar_distance):
    """
    @param lidar_bin: The beam index that provided this measurement
    @param lidar_distance: The distance measurement from the sensor for that beam
    @return world_point: List containing the corresponding (x,y) point in the world frame of reference
    """

    # YOUR CODE HERE
    # print("Dist:", lidar_distance, "Ang:", lidar_bin)

    # No detection
    if (lidar_distance > LIDAR_SENSOR_MAX_RANGE):  # or lidar_distance > math.sqrt(2)):
        return None

    # Lidar centered at robot 0,0 so no translation needed
    # Convert lidar -> robot adding math.pi/2 to fix direction
    bQ_x = math.sin(lidar_bin + math.pi / 2) * lidar_distance
    bQ_y = math.cos(lidar_bin + math.pi / 2) * lidar_distance
    # print(bQ_x, bQ_y)
    # convert robot -> world
    x = math.cos(pose_theta) * bQ_x - math.sin(pose_theta) * bQ_y + pose_x
    y = math.sin(pose_theta) * bQ_x + math.cos(pose_theta) * bQ_y + pose_y
    # print(x,y)
    return [x, y]


def get_wheel_speeds(target_pose):
    '''
    @param target_pose: Array of (x,y,theta) for the destination robot pose
    @return motor speed as percentage of maximum for left and right wheel motors
    '''
    global pose_x, pose_y, pose_theta, left_wheel_direction, right_wheel_direction

    pose_x, pose_y, pose_theta = supervisor_ekf_slam.supervisor_get_robot_pose()

    bearing_error = math.atan2((target_pose[1] - pose_y), (target_pose[0] - pose_x)) - pose_theta
    distance_error = np.linalg.norm(target_pose[:2] - np.array([pose_x, pose_y]))
    heading_error = target_pose[2] - pose_theta

    BEAR_THRESHOLD = 0.06
    DIST_THRESHOLD = 0.03
    dT_gain = theta_gain
    dX_gain = distance_gain
    if distance_error > DIST_THRESHOLD:
        dTheta = bearing_error
        if abs(bearing_error) > BEAR_THRESHOLD:
            dX_gain = 0
    else:
        dTheta = heading_error
        dX_gain = 0

    dTheta *= dT_gain
    dX = dX_gain * min(3.14159, distance_error)

    phi_l = (dX - (dTheta * EPUCK_AXLE_DIAMETER / 2.)) / EPUCK_WHEEL_RADIUS
    phi_r = (dX + (dTheta * EPUCK_AXLE_DIAMETER / 2.)) / EPUCK_WHEEL_RADIUS

    left_speed_pct = 0
    right_speed_pct = 0

    wheel_rotation_normalizer = max(abs(phi_l), abs(phi_r))
    left_speed_pct = (phi_l) / wheel_rotation_normalizer
    right_speed_pct = (phi_r) / wheel_rotation_normalizer

    if distance_error < 0.05 and abs(heading_error) < 0.05:
        left_speed_pct = 0
        right_speed_pct = 0

    left_wheel_direction = left_speed_pct * MAX_VEL_REDUCTION
    phi_l_pct = left_speed_pct * MAX_VEL_REDUCTION * left_motor.getMaxVelocity()

    right_wheel_direction = right_speed_pct * MAX_VEL_REDUCTION
    phi_r_pct = right_speed_pct * MAX_VEL_REDUCTION * right_motor.getMaxVelocity()

    return phi_l_pct, phi_r_pct, (phi_l, phi_r)


def dist(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def perform_least_squares_line_estimate(lidar_world_coords, selected_points):
    """
    @param lidar_world_coords: List of world coordinates read from lidar data (tuples of the form [x, y])
    @param selected_points: Indicies of the points selected for this least squares line estimation
    @return m, b: Slope and intercept for line estimated from data - y = mx + b
    """
    sum_y = sum_yy = sum_x = sum_xx = sum_yx = 0  # Sums of y coordinates, y^2 for each coordinate, x coordinates, x^2 for each coordinate, and y*x for each point

    for point in selected_points:
        world_coord = lidar_world_coords[point]

        sum_y += world_coord[1]
        sum_yy += world_coord[1] ** 2
        sum_x += world_coord[0]
        sum_xx += world_coord[0] ** 2
        sum_yx += world_coord[0] * world_coord[1]

    num_points = len(selected_points)
    b = (sum_y * sum_xx - sum_x * sum_yx) / (num_points * sum_xx - sum_x ** 2)
    m = (num_points * sum_yx - sum_x * sum_y) / (num_points * sum_xx - sum_x ** 2)

    return m, b


def distance_to_line(x, y, m, b):
    """
    @param x: x coordinate of point to find distance to line from
    @param y: y coordinate of point to find distance to line from
    @param m: slope of line
    @param b: intercept of line
    @return dist: the distance from the given point coordinates to the given line
    """

    # Line perpendicular to input line crossing through input point - m*m_o = -1 and y=m_o*x + b_o => b_o = y - m_o*x
    m_o = -1.0 / m
    b_o = y - m_o * x

    # Intersection point between y = m*x + b and x = m_o*x + b_o
    p_x = (b - b_o) / (m_o - m)
    p_y = ((m_o * (b - b_o)) / (m_o - m)) + b_o
    print("przed bledem ", [x, y], [p_x, p_y])
    return dist([x, y], [p_x, p_y])


def get_line_landmark(line):
    global mu
    # slope perpendicular to input line
    m_o = -1.0 / line[0]

    # landmark position
    lm_x = line[1] / (m_o - line[0])
    lm_y = (m_o * line[1]) / (m_o - line[0])
    lm_j = 0

    found = False
    for [x, y, j] in landmark_globals:
        if dist([x, y], [lm_x, lm_y]) <= 0.5:
            lm_j = j
            found = True
            break
        lm_j += 1

    # If we didn't match the landmark to a previously found one and we're over the cap for new landmarks, return none to not calculate with this landmark
    if not found and len(landmark_globals) >= n:
        return None
    # Otherwise, add it to our landmarks
    elif not found and len(landmark_globals) < n:
        landmark_globals.append([lm_x, lm_y, lm_j])

    # convert to robot-relative positioning with radius from the robot and theta relative to robot
    r = dist([lm_x, lm_y], [mu[0][0], mu[1][0]])
    theta = math.atan2(lm_x, lm_y)
    theta = get_bounded_theta(theta - mu[2][0])

    return [r, theta, lm_j]


def extract_line_landmarks(lidar_world_coords):
    """
    @param lidar_world_coords: List of world coordinates read from lidar data (tuples of the form [x, y])
    @return found_landmarks: list of landmarks found through the RANSAC done on the lidar data
    """

    found_lines = []  # list of tuples of the form [m, b] of detected lines

    linepoints = []  # list of laser data points not yet associated to a found line

    found_landmarks = []  # list to keep track of found landmarks from lines, stored as [r, theta, j] relative to robot

    for i in range(len(lidar_world_coords)):
        linepoints.append(i)

    num_trials = 0
    while (num_trials < MAX_TRIALS and len(linepoints) >= MIN_LINE_POINTS):
        rand_selected_points = []

        # randomly choose up to MAX_SAMPLE points for the least squares
        for i in range(min(MAX_SAMPLE, len(linepoints))):
            temp = -1
            new_point = False
            while not new_point:
                temp = random.randint(0,
                                      len(linepoints) - 1)  # generate a random integer between 0 and our total number of remaining line points to choose from
                if linepoints[temp] not in rand_selected_points:
                    new_point = True
            rand_selected_points.append(linepoints[temp])

        # Now compute a line based on the randomly selected points
        m, b = perform_least_squares_line_estimate(lidar_world_coords, rand_selected_points)

        consensus_points = []  # points matching along the found line
        new_linepoints = []  # points not matching along the found line, if we say the line is a landmark, these are our new set of unmatched points

        for point in linepoints:
            curr_point = lidar_world_coords[point]
            # distance to line from the point
            dist = distance_to_line(curr_point[0], curr_point[1], m, b)

            if dist < RANSAC_TOLERANCE:
                consensus_points.append(point)
            else:
                new_linepoints.append(point)

        if len(consensus_points) >= RANSAC_CONSENSUS:
            # Calculate an updated line based on every point within the consensus
            m, b = perform_least_squares_line_estimate(lidar_world_coords, consensus_points)

            # add to found lines
            found_lines.append([m, b])

            # rewrite the linepoints as the linepoints that didn't match with this line to only search unmatched points for lines
            linepoints = new_linepoints.copy()

            # restart number of trials
            num_trials = 0
        else:
            num_trials += 1

    # Now we'll calculate the point closest to the origin for each line found and add these as found landmarks
    for line in found_lines:
        new_landmark = get_line_landmark(line)
        if new_landmark is not None:
            found_landmarks.append(new_landmark)

    return found_landmarks


def EKF_init(x_init):
    global Rt, Qt, mu, cov

    Rt = 5 * np.array([[0.01, 0, 0],
                       [0, 0.01, 0],
                       [0, 0, 0.01]])
    Qt = np.array([[0.01, 0],
                   [0, 0.01]])

    mu = np.append(np.array([x_init]).T, np.zeros((2 * n, 1)), axis=0)
    mu_new = mu

    cov = 1e6 * np.eye(2 * n + 3)

    cov[:3, :3] = np.eye(3, 3) * np.array(x_init).T
    cov_new = cov


def EKF_predict(u, Rt):
    # global mu
    # n = len(mu)

    # Define motion model f(mu,u)
    [dtrans, drot1, drot2] = u
    motion = np.array([[dtrans * np.cos(mu[2][0] + drot1)],
                       [dtrans * np.sin(mu[2][0] + drot1)],
                       [get_bounded_theta(drot1 + drot2)]])
    F = np.append(np.eye(3), np.zeros((3, 2 * n)), axis=1)

    # print(np.shape(F.T))
    # print(np.shape(mu))
    # print(np.shape((F.T).dot(motion)))
    # Predict new state
    mu_bar = mu + (F.T).dot(motion)

    # Define motion model Jacobian
    J = np.array([[0, 0, -dtrans * np.sin(get_bounded_theta(mu[2][0] + drot1))],
                  [0, 0, dtrans * np.cos(get_bounded_theta(mu[2][0] + drot1))],
                  [0, 0, 0]])
    G = np.eye(2 * n + 3) + (F.T).dot(J).dot(F)

    # Predict new covariance
    cov_bar = G.dot(cov).dot(G.T) + (F.T).dot(Rt).dot(F)

    print('Predicted location\t x: {0:.2f} \t y: {1:.2f} \t theta: {2:.2f}'.format(mu_bar[0][0], mu_bar[1][0],
                                                                                   mu_bar[2][0]))
    return mu_bar, cov_bar


def move(target_pose):
    lspeed, rspeed, (phi_l, phi_r) = get_wheel_speeds(target_pose)
    dtrans = np.linalg.norm(np.array(target_pose[:2]) - np.array(mu[0][0], mu[1][0]))
    u = [dtrans, phi_l, phi_r]

    print("lspeed: ", lspeed, "rspeed: ", rspeed)
    left_motor.setVelocity(lspeed)
    right_motor.setVelocity(rspeed)

    return u


def generate_obs():
    '''BIERZE DANE LIDARA I ZAMIENIA JE NA '''
    global lidar, lidar_data
    lidar_data = lidar.getRangeImage()
    # print(lidar_data)

    # convert lidar to world locations
    lidar_readings = []
    for i in range(21):
        lidar_found_loc = convert_lidar_reading_to_world_coord(i, lidar_data[i])
        if lidar_found_loc is not None:
            lidar_readings.append(lidar_found_loc)

    # print(lidar_readings)
    # Run RANSAC on lidar_data
    obs = extract_line_landmarks(lidar_readings)  # lidar readnig - dane lidar zamieniona to world coord

    return obs  # [r, theta, j]


def update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed):
    '''
    Given the amount of time passed and the direction each wheel was rotating,
    update the robot's pose information accordingly
    '''
    global pose_x, pose_y, pose_theta, EPUCK_MAX_WHEEL_SPEED, EPUCK_AXLE_DIAMETER
    # Update pose_theta
    pose_theta += (
                          right_wheel_direction - left_wheel_direction) * time_elapsed * EPUCK_MAX_WHEEL_SPEED / EPUCK_AXLE_DIAMETER
    # Update pose_x
    pose_x += math.cos(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (
            left_wheel_direction + right_wheel_direction) / 2.
    # Update pose_y
    pose_y += math.sin(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (
            left_wheel_direction + right_wheel_direction) / 2.
    print(time_elapsed, pose_theta, pose_y, pose_x)


def run_robot():
    global robot, ground_sensors, ground_sensor_readings, pose_x, pose_y, pose_theta, state
    global leftMotor, rightMotor, SIM_TIMESTEP, WHEEL_FORWARD, WHEEL_STOPPED, WHEEL_BACKWARDS
    global cov, Rt, Qt, mu

    last_odometry_update_time = None

    for i in range(10):
        robot.step(SIM_TIMESTEP)

    # Pozycja robota
    start_pose = supervisor_ekf_slam.supervisor_get_robot_pose()
    pose_x, pose_y, pose_theta = start_pose

    lidar_obs = []
    u = [0, 0, 0]  # State vector
    # Tolerances for reaching waypoint state
    x_tol = 0.06
    y_tol = 0.06
    theta_tol = 0.1
    last_EKF_update = None
    waypoint = True

    print(start_pose)
    EKF_init(start_pose)

    '''tymczasowo'''
    robot_path = [[0.30928, 0.176902, 1.046032],
                  [0.433827, 0.642393, 1.3084],
                  [0.434453, 1.702051, 1.5702],
                  [0.275088, 1.978296, 2.353669],
                  [-0.144059, 1.9779351, 3.12076],
                  [-0.484122, 1.556442, -1.83403],
                  [-0.486042, 0.158664, -1.57152],
                  [-0.345818, 0.018178, -0.783037],
                  [-0.03, -0.05, 0]]
    pos_idx = 0

    while robot.step(SIM_TIMESTEP) != -1:
        sleep(0.025)

        loop_closure_detection_time = 0

        # Zaaktualizowanie pozycji robota
        if last_odometry_update_time is None:
            last_odometry_update_time = robot.getTime()

        time_elapsed = robot.getTime() - last_odometry_update_time
        update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed)
        last_odometry_update_time = robot.getTime()
        print('superv', supervisor_ekf_slam.supervisor_get_robot_pose())
        print("Current pose mu: [%5f, %5f, %5f]" % (mu[0][0], mu[1][0], mu[2][0]))
        print("Current pose: [%5f, %5f, %5f]" % (pose_x, pose_y, pose_theta))

        if last_EKF_update is None:
            last_EKF_update = robot.getTime()

        # sterowanie klawiatura
        # key = keyboard.getKey()
        # if key in motor_cmd.keys():
        #     steruj(motor_cmd[key])
        # else:
        #     left_motor.setVelocity(0)
        #     right_motor.setVelocity(0)

        '''print("Moving")
        u_tmp = move(robot_path[pos_idx])
        pos_idx+=1
        if pos_idx == len(robot_path):
            pos_idx = 0'''
        if pos_idx == len(robot_path):
            pos_idx = 0
        target_pose = robot_path[pos_idx]
        pos_idx += 1

        # Sense
        print("Sensing")
        tmp_obs = generate_obs()
        print("TMP_OBS: ", tmp_obs)
        # for ob in tmp_obs:
        #     add = True
        #     for ob_2 in lidar_obs:
        #         print(ob[2], ob_2[2])
        #         if ob[2] == ob_2[2]:
        #             add = False
        #     if add:
        #         lidar_obs.append(ob)
        #
        # print("Lidar Obs: ", lidar_obs)
        # print("All lidar objects detected this run: ", landmark_globals)
        # Lidar - wizualizacja
        # lidar_data = lidar.getRangeImage()
        # y = lidar_data
        # x = np.linspace(math.pi * 0.8, 0, np.size(y))
        # plt.polar(x, y)
        # plt.pause(0.0000001)
        # plt.clf()

        # print("FOV", lidar.getFov())
        # print("FOV2", lidar.getVerticalFov())
        # print('min', lidar.getMinRange())
        # print('max', lidar.getMaxRange())
        # print('getRangeImage', lidar_data)

        # Sense
        print("Sensing")
        tmp_obs = generate_obs()
        # print("TMP_OBS: ", tmp_obs)
        for ob in tmp_obs:
            add = True
            for ob_2 in lidar_obs:
                # print(ob[2], ob_2[2])
                if ob[2] == ob_2[2]:
                    add = False
            if add:
                lidar_obs.append(ob)

        print("Lidar Obs: ", lidar_obs)
        print("All lidar objects detected this run: ", landmark_globals)

        if robot.getTime() - last_EKF_update > 0.5:
            print("EKF Run")
            # Predict
            # elif state == "predict":
            print("EKF Predict")
            mu_new, cov = EKF_predict(u, Rt)
            mu = mu_new
            # mu = np.append(mu,mu_new,axis=1)

            # Update
            print("EKF Update")
            if len(lidar_obs) == 0:
                print("Skipping as no new obs")
                continue

            mu_new, cov = EKF_update(lidar_obs, Qt)
            mu = mu_new
            lidar_obs = []
            # mu = np.append(mu,mu_new,axis=1)

            print("MU: ", mu)

            print("Cov: ", cov)
            last_EKF_update = robot.getTime()


if __name__ == "__main__":
    run_robot()
    plt.show()
