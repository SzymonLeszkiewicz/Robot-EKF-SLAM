import math
import random
import copy
from controller import Robot, Motor, DistanceSensor, Lidar, Keyboard
import numpy as np
import collections
from time import sleep
import matplotlib.pyplot as plt

import supervisor_ekf_slam

supervisor_ekf_slam.init_supervisor()
robot = supervisor_ekf_slam.supervisor




LIDAR_SENSOR_MAX_RANGE = 5.
LIDAR_ANGLE_BINS = 21
LIDAR_ANGLE_RANGE = 1.5708

MAX_TRIALS = 1000
MAX_SAMPLE = 10
MIN_LINE_POINTS = 6
RANSAC_TOLERANCE = 0.15
RANSAC_CONSENSUS = 6

pose_x = 0
pose_y = 0
pose_theta = 0
lw_kierunek = 0
rw_kierunek = 0

WHEEL_FORWARD = 1
WHEEL_STOPPED = 0
WHEEL_BACKWARD = -1

theta_gain = 1.0
distance_gain = 0.3

MAX_VEL_REDUCTION = 0.25
EPUCK_MAX_WHEEL_SPEED = 0.125 * MAX_VEL_REDUCTION  # m/s
EPUCK_AXLE_DIAMETER = 0.053
EPUCK_WHEEL_RADIUS = 0.0205


WHEEL_FORWARD = 1
WHEEL_STOPPED = 0
WHEEL_BACKWARD = -1

dane_lidar = np.zeros(LIDAR_ANGLE_BINS)

step = LIDAR_ANGLE_RANGE / LIDAR_ANGLE_BINS


n = 20
mu = []
cov = []
mu_new = []
cov_new = []

landmark_globals = []

SIM_TIMESTEP = int(robot.getBasicTimeStep())
max_speed = 6.28
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)
# Lidar
lidar = robot.getDevice("LDS-01")
lidar.enable(SIM_TIMESTEP)
lidar.enablePointCloud()
# lidar motory
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


komendy_kola = {
    ord("W"): (max_speed, max_speed),
    ord("S"): (-max_speed, -max_speed),
    ord("A"): (-max_speed / 2, max_speed / 2),
    ord("D"): (max_speed / 2, -max_speed / 2),
}


def get_bounded_theta(theta):

    while theta > math.pi: theta -= 2. * math.pi
    while theta < -math.pi: theta += 2. * math.pi
    return theta


def lidar_koordynaty(lidar_bin, lidar_distance):

    if (lidar_distance > LIDAR_SENSOR_MAX_RANGE):
        return None
    bQ_x = math.sin(lidar_bin + math.pi / 2) * lidar_distance
    bQ_y = math.cos(lidar_bin + math.pi / 2) * lidar_distance
    x = math.cos(pose_theta) * bQ_x - math.sin(pose_theta) * bQ_y + pose_x
    y = math.sin(pose_theta) * bQ_x + math.cos(pose_theta) * bQ_y + pose_y
    return [x, y]


def kola_predkosc(target_pose):
    global pose_x, pose_y, pose_theta, lw_kierunek, rw_kierunek

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

    lw_kierunek = left_speed_pct * MAX_VEL_REDUCTION
    phi_l_pct = left_speed_pct * MAX_VEL_REDUCTION * left_motor.getMaxVelocity()

    rw_kierunek = right_speed_pct * MAX_VEL_REDUCTION
    phi_r_pct = right_speed_pct * MAX_VEL_REDUCTION * right_motor.getMaxVelocity()

    return phi_l_pct, phi_r_pct, (phi_l, phi_r)


def dist(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def perform_least_squares_line_estimate(lidar_world_coords, selected_points):
    sum_y = sum_yy = sum_x = sum_xx = sum_yx = 0

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

    m_o = -1.0 / m
    b_o = y - m_o * x

    p_x = (b - b_o) / (m_o - m)
    p_y = ((m_o * (b - b_o)) / (m_o - m)) + b_o
    # print("przed bledem ", [x, y], [p_x, p_y])
    return dist([x, y], [p_x, p_y])


def get_line_landmark(line):
    global mu
    m_o = -1.0 / line[0]

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

    if not found and len(landmark_globals) >= n:
        return None
    elif not found and len(landmark_globals) < n:
        landmark_globals.append([lm_x, lm_y, lm_j])

    r = dist([lm_x, lm_y], [mu[0][0], mu[1][0]])
    theta = math.atan2(lm_x, lm_y)
    theta = get_bounded_theta(theta - mu[2][0])

    return [r, theta, lm_j]


def extract_line_landmarks(lidar_world_coords):


    found_lines = []

    linepoints = []

    found_landmarks = []

    for i in range(len(lidar_world_coords)):
        linepoints.append(i)

    num_trials = 0
    while (num_trials < MAX_TRIALS and len(linepoints) >= MIN_LINE_POINTS):
        rand_selected_points = []

        for i in range(min(MAX_SAMPLE, len(linepoints))):
            temp = -1
            new_point = False
            while not new_point:
                temp = random.randint(0,
                                      len(linepoints) - 1)
                if linepoints[temp] not in rand_selected_points:
                    new_point = True
            rand_selected_points.append(linepoints[temp])

        m, b = perform_least_squares_line_estimate(lidar_world_coords, rand_selected_points)

        consensus_points = []
        new_linepoints = []

        for point in linepoints:
            curr_point = lidar_world_coords[point]
            dist = distance_to_line(curr_point[0], curr_point[1], m, b)

            if dist < RANSAC_TOLERANCE:
                consensus_points.append(point)
            else:
                new_linepoints.append(point)

        if len(consensus_points) >= RANSAC_CONSENSUS:
            m, b = perform_least_squares_line_estimate(lidar_world_coords, consensus_points)

            found_lines.append([m, b])

            linepoints = new_linepoints.copy()

            num_trials = 0
        else:
            num_trials += 1

    for line in found_lines:
        new_landmark = get_line_landmark(line)
        if new_landmark is not None:
            found_landmarks.append(new_landmark)

    return found_landmarks


def EKF_init(x_init):
    global Rt, Qt, mu, cov
    global cov_new, mu_new

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
    print("PREDICT")
    print(Rt)
    print(Qt)
    print(mu)
    print(cov)



def EKF_predict(u, Rt):

    [dtrans, drot1, drot2] = u
    motion = np.array([[dtrans * np.cos(mu[2][0] + drot1)],
                       [dtrans * np.sin(mu[2][0] + drot1)],
                       [get_bounded_theta(drot1 + drot2)]])
    F = np.append(np.eye(3), np.zeros((3, 2 * n)), axis=1)


    mu_bar = mu + (F.T).dot(motion)

    J = np.array([[0, 0, -dtrans * np.sin(get_bounded_theta(mu[2][0] + drot1))],
                  [0, 0, dtrans * np.cos(get_bounded_theta(mu[2][0] + drot1))],
                  [0, 0, 0]])
    G = np.eye(2 * n + 3) + (F.T).dot(J).dot(F)

    cov_bar = G.dot(cov).dot(G.T) + (F.T).dot(Rt).dot(F)

    print('Predicted location\t x: {0:.2f} \t y: {1:.2f} \t theta: {2:.2f}'.format(mu_bar[0][0], mu_bar[1][0],
                                                                                   mu_bar[2][0]))
    return mu_bar, cov_bar


def EKF_update(obs, Qt):
    global mu, mu_new, cov_new

    for [r, theta, j] in obs:
        j = int(j)
        print([r, theta, j])
        if cov[2 * j + 3][2 * j + 3] >= 1e6 and cov[2 * j + 4][2 * j + 4] >= 1e6:
            mu[2 * j + 3][0] = mu[0][0] + r * np.cos(get_bounded_theta(theta + mu[2][0]))
            mu[2 * j + 4][0] = mu[1][0] + r * np.sin(get_bounded_theta(theta + mu[2][0]))

        delta = np.array([mu[2 * j + 3][0] - mu[0][0], mu[2 * j + 4][0] - mu[1][0]])
        q = delta.T.dot(delta)
        sq = np.sqrt(q)
        z_theta = np.arctan2(delta[1], delta[0])
        z_hat = np.array([[sq], [get_bounded_theta(z_theta - mu[2][0])]])

        F = np.zeros((5, 2 * n + 3))
        F[:3, :3] = np.eye(3)
        F[3, 2 * j + 3] = 1
        F[4, 2 * j + 4] = 1
        H_low = np.array([[-sq * delta[0], -sq * delta[1], 0, sq * delta[0], sq * delta[1]],
                          [delta[1], -delta[0], -q, -delta[1], delta[0]]], dtype='float')
        H = 1 / q * H_low.dot(F)

        K = cov.dot(H.T).dot(np.linalg.inv(H.dot(cov).dot(H.T) + Qt))
        z_dif = np.array([[r], [theta]]) - z_hat
        z_dif = (z_dif + np.pi) % (2 * np.pi) - np.pi

        mu_new = mu + K.dot(z_dif)
        cov_new = (np.eye(2 * n + 3) - K.dot(H)).dot(cov)
    print("Cov_new: ", cov_new)

    print('Updated location\t x: {0:.2f} \t y: {1:.2f} \t theta: {2:.2f}'.format(mu_new[0][0], mu_new[1][0],
                                                                                 mu_new[2][0]))
    return mu_new, cov_new


def move(target_pose):
    lspeed, rspeed, (phi_l, phi_r) = kola_predkosc(target_pose)
    dtrans = np.linalg.norm(np.array(target_pose[:2]) - np.array(mu[0][0], mu[1][0]))
    u = [dtrans, phi_l, phi_r]

    return u


def obiekty():
    global lidar, dane_lidar
    dane_lidar = lidar.getRangeImage()

    lidar_readings = []
    for i in range(21):
        lidar_found_loc = lidar_koordynaty(i, dane_lidar[i])
        if lidar_found_loc is not None:
            lidar_readings.append(lidar_found_loc)

    obs = extract_line_landmarks(lidar_readings)

    return obs


def odometria_update(left_wheel_direction, right_wheel_direction, time_elapsed):
    global pose_x, pose_y, pose_theta, EPUCK_MAX_WHEEL_SPEED, EPUCK_AXLE_DIAMETER

    pose_theta += (
                          right_wheel_direction - left_wheel_direction) * time_elapsed * EPUCK_MAX_WHEEL_SPEED / EPUCK_AXLE_DIAMETER

    pose_x += math.cos(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (
            left_wheel_direction + right_wheel_direction) / 2.

    pose_y += math.sin(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (
            left_wheel_direction + right_wheel_direction) / 2.

    print("Odometria: [%3f, %3f, %3f]" % (pose_x, pose_y, pose_theta))


def run_robot():
    global robot, ground_sensors, ground_sensor_readings, pose_x, pose_y, pose_theta
    global leftMotor, rightMotor, SIM_TIMESTEP, WHEEL_FORWARD, WHEEL_STOPPED, WHEEL_BACKWARDS
    global cov, Rt, Qt, mu
    global lw_kierunek, rw_kierunek
    m = np.zeros((50, 50))
    aktualizacja_odometria = None
    np.set_printoptions(precision=3, suppress=True)

    for i in range(10):
        robot.step(SIM_TIMESTEP)

    # Początkowa pozycji robota
    pozycja_startowa = supervisor_ekf_slam.supervisor_get_robot_pose()
    pose_x, pose_y, pose_theta = pozycja_startowa

    lidar_obs = []
    u = [0, 0, 0]  # wektor stanu
    aktualizacja_EKF = None

    EKF_init(pozycja_startowa)

    while robot.step(SIM_TIMESTEP) != -1:
        # Odometria
        if aktualizacja_odometria is None:
            aktualizacja_odometria = robot.getTime()

        uplyw_t = robot.getTime() - aktualizacja_odometria
        print("supervisor pos", supervisor_ekf_slam.supervisor_get_robot_pose())
        odometria_update(lw_kierunek, rw_kierunek, uplyw_t)
        aktualizacja_odometria = robot.getTime()
        # print("Current pose mu: [%3f, %3f, %3f]" % (mu[0][0], mu[1][0], mu[2][0]))

        if aktualizacja_EKF is None:
            aktualizacja_EKF = robot.getTime()

        kola_predkosc(supervisor_ekf_slam.supervisor_get_robot_pose())

        print("Przeszkoda:", supervisor_ekf_slam.supervisor_get_obstacle_positions())

        # sterowanie klawiatura
        klawisz = keyboard.getKey()
        if klawisz in komendy_kola.keys():
            steruj(komendy_kola[klawisz])

        elif klawisz == ord('R'):
            supervisor_ekf_slam.supervisor_reset_to_home()

        else:
            left_motor.setVelocity(0)
            right_motor.setVelocity(0)

        u = move(supervisor_ekf_slam.supervisor_get_robot_pose())

        tmp_obs = obiekty()
        for ob in tmp_obs:
            add = True
            for ob_2 in lidar_obs:
                if ob[2] == ob_2[2]:
                    add = False
            if add:
                lidar_obs.append(ob)

        if robot.getTime() - aktualizacja_EKF > 0.5:
            print("EKF Run")

            print("EKF Predict")
            mu_new, cov = EKF_predict(u, Rt)
            mu = mu_new

            print("EKF Update")
            if len(lidar_obs) == 0:
                print("Brak nowych obiektów")
                print(' ')
                continue

            mu_new, cov = EKF_update(lidar_obs, Qt)
            mu = mu_new
            lidar_obs = []

            aktualizacja_EKF = robot.getTime()

        print("Lidar: ", lidar_obs)
        print("Lidar dane ", landmark_globals)
        # for i in landmark_globals:
        #     plt.scatter(i[0], i[1])
        # plt.pause(0.0000001)
        # plt.clf()
        # Lidar - wizualizacja
        lidar_data = lidar.getRangeImage()
        y = lidar_data
        x = np.linspace(math.pi * 0.8, 0, np.size(y))
        plt.polar(x, y)
        plt.pause(0.0000001)
        plt.clf()


if __name__ == "__main__":
    run_robot()
    plt.show()
