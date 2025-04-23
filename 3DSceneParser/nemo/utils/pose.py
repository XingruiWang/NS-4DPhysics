import math
import torch
import numpy as np


def rotation_matrix(theta, elev, azum, batch_size=None):
    # Convert angles from degrees to radians for math operations
    # theta, elev, azum = map(torch.radians, [theta, elev, azum])

    # Azimuth rotation around the z-axis
    cos_azum, sin_azum = torch.cos(azum), torch.sin(azum)
    R_azum = torch.stack([
        torch.stack([cos_azum, -sin_azum, torch.zeros_like(azum)], dim=-1),
        torch.stack([sin_azum, cos_azum, torch.zeros_like(azum)], dim=-1),
        torch.stack([torch.zeros_like(azum), torch.zeros_like(azum), torch.ones_like(azum)], dim=-1)
    ], dim=-2)

    # Elevation rotation around the y-axis
    cos_elev, sin_elev = torch.cos(elev), torch.sin(elev)
    R_elev = torch.stack([
        torch.stack([cos_elev, torch.zeros_like(elev), sin_elev], dim=-1),
        torch.stack([torch.zeros_like(elev), torch.ones_like(elev), torch.zeros_like(elev)], dim=-1),
        torch.stack([-sin_elev, torch.zeros_like(elev), cos_elev], dim=-1)
    ], dim=-2)

    # Theta rotation around the z-axis (if different from azimuth)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    R_theta = torch.stack([
        torch.stack([cos_theta, -sin_theta, torch.zeros_like(theta)], dim=-1),
        torch.stack([sin_theta, cos_theta, torch.zeros_like(theta)], dim=-1),
        torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=-1)
    ], dim=-2)

    # Combine rotations
    R = torch.matmul(R_theta, torch.matmul(R_elev, R_azum))
    if batch_size is not None:
        N = R.shape[0]
        R = R.view(batch_size, N//batch_size, 3, 3)
    return R


def get_transformation_matrix(azimuth, elevation, distance):
    if distance == 0:
        # return None
        distance = 0.1

    # camera center
    C = np.zeros((3, 1))
    C[0] = distance * math.cos(elevation) * math.sin(azimuth)
    C[1] = -distance * math.cos(elevation) * math.cos(azimuth)
    C[2] = distance * math.sin(elevation)

    # rotate coordinate system by theta is equal to rotating the model by theta
    azimuth = -azimuth
    elevation = -(math.pi / 2 - elevation)

    # rotation matrix
    Rz = np.array(
        [
            [math.cos(azimuth), -math.sin(azimuth), 0],
            [math.sin(azimuth), math.cos(azimuth), 0],
            [0, 0, 1],
        ]
    )  # rotation by azimuth
    Rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(elevation), -math.sin(elevation)],
            [0, math.sin(elevation), math.cos(elevation)],
        ]
    )  # rotation by elevation
    R_rot = np.dot(Rx, Rz)
    R = np.hstack((R_rot, np.dot(-R_rot, C)))
    R = np.vstack((R, [0, 0, 0, 1]))

    return R


def rotation_theta(theta):
    # cos -sin  0
    # sin  cos  0
    # 0    0    1
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def cal_rotation_matrix(theta, elev, azum, dis):
    if dis <= 1e-10:
        dis = 0.5
    return rotation_theta(theta) @ get_transformation_matrix(azum, elev, dis)[0:3, 0:3]
