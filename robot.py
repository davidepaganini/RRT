import numpy as np
from copy import deepcopy


def get_robot_cylinders_and_spheres(joints, base_tform=np.eye(4), color="b"):
    cylinders = []
    spheres = []

    tforms = forward_kinematics_py(joints, base_tform=base_tform)
    cylinder_format = {"joint_n": None,
                       "point_a": np.array([0.0, 0.0, 0.0]),
                       "point_b": np.array([0.0, 0.0, 0.0]),
                       "cylinder_axis": np.array([0.0, 0.0, 0.0]),
                       "color": color,
                       "radius": 0.075,
                       "discrete": None}

    cylinder = deepcopy(cylinder_format)
    cylinder["point_a"] = base_tform[:3,3]
    cylinder["point_b"] = np.dot(base_tform, np.array([0, 0, 0.4, 1]))[0:3]
    cylinders.append(deepcopy(cylinder))
    spheres.append({"center": cylinder["point_b"].copy(), "color": color, "radius": 0.075})

    cylinder = deepcopy(cylinder_format)
    cylinder["joint_n"] = 0
    cylinder["point_a"] = cylinders[-1]["point_b"]
    cylinder["cylinder_axis"] = np.array([0, -0.4, 0, 1])
    cylinder["point_b"] = np.dot(tforms[cylinder["joint_n"]], cylinder["cylinder_axis"])[0:3]
    cylinders.append(deepcopy(cylinder))
    spheres.append({"center": cylinder["point_b"].copy(), "color": color, "radius": 0.075})

    cylinder = deepcopy(cylinder_format)
    cylinder["joint_n"] = 1
    cylinder["point_a"] = cylinders[-1]["point_b"]
    cylinder["cylinder_axis"] = np.array([1.2, 0, 0.4, 1])
    cylinder["point_b"] = np.dot(tforms[cylinder["joint_n"]], cylinder["cylinder_axis"])[0:3]
    cylinders.append(deepcopy(cylinder))
    spheres.append({"center": cylinder["point_b"].copy(), "color": color, "radius": 0.075})

    cylinder = deepcopy(cylinder_format)
    cylinder["joint_n"] = 2
    cylinder["point_a"] = cylinders[-1]["point_b"]
    cylinder["cylinder_axis"] = np.array([0.0, 0.0, 0.0, 1])
    cylinder["point_b"] = np.dot(tforms[cylinder["joint_n"]], cylinder["cylinder_axis"])[0:3]
    cylinders.append(deepcopy(cylinder))
    spheres.append({"center": cylinder["point_b"].copy(), "color": color, "radius": 0.075})

    cylinder = deepcopy(cylinder_format)
    cylinder["joint_n"] = 2
    cylinder["point_a"] = cylinders[-1]["point_b"]
    cylinder["cylinder_axis"] = np.array([0, 0.2, 0, 1])
    cylinder["point_b"] = np.dot(tforms[cylinder["joint_n"]], cylinder["cylinder_axis"])[0:3]
    cylinders.append(deepcopy(cylinder))
    spheres.append({"center": cylinder["point_b"].copy(), "color": color, "radius": 0.075})

    cylinder = deepcopy(cylinder_format)
    cylinder["joint_n"] = 3
    cylinder["point_a"] = cylinders[-1]["point_b"]
    cylinder["cylinder_axis"] = np.array([0, -0.2, -0.8, 1])
    cylinder["point_b"] = np.dot(tforms[cylinder["joint_n"]], cylinder["cylinder_axis"])[0:3]
    cylinders.append(deepcopy(cylinder))
    spheres.append({"center": cylinder["point_b"].copy(), "color": color, "radius": 0.075})

    cylinder = deepcopy(cylinder_format)
    cylinder["joint_n"] = 3
    cylinder["point_a"] = cylinders[-1]["point_b"]
    cylinder["cylinder_axis"] = np.array([0, -0.2, 0, 1])
    cylinder["point_b"] = np.dot(tforms[cylinder["joint_n"]], cylinder["cylinder_axis"])[0:3]
    cylinders.append(deepcopy(cylinder))
    spheres.append({"center": cylinder["point_b"].copy(), "color": color, "radius": 0.075})

    cylinder = deepcopy(cylinder_format)
    cylinder["joint_n"] = 4
    cylinder["point_a"] = cylinders[-1]["point_b"]
    cylinder["cylinder_axis"] = np.array([0.0, 0.0, 0.0, 1])
    cylinder["point_b"] = np.dot(tforms[cylinder["joint_n"]], cylinder["cylinder_axis"])[0:3]
    cylinders.append(deepcopy(cylinder))
    spheres.append({"center": cylinder["point_b"].copy(), "color": color, "radius": 0.075})

    cylinder = deepcopy(cylinder_format)
    cylinder["joint_n"] = 4
    cylinder["point_a"] = cylinders[-1]["point_b"]
    cylinder["cylinder_axis"] = np.array([0, 0.4, 0, 1])
    cylinder["point_b"] = np.dot(tforms[cylinder["joint_n"]], cylinder["cylinder_axis"])[0:3]
    cylinders.append(deepcopy(cylinder))
    spheres.append({"center": cylinder["point_b"].copy(), "color": color, "radius": 0.075})

    return cylinders, spheres


def forward_kinematics_py(theta_joints, base_tform=np.eye(4)):
    a_off = np.array([0.0, 0.0, 1.2, 0, 0.0])  # 0.0])
    d_off = np.array([0.4, 0.0, 0.0, 1, 0.0])  # 0.4])
    alpha = np.array([0, np.pi / 2, 0, -np.pi / 2, np.pi / 2])  # -np.pi / 2])
    theta_offsets = np.array([0, np.pi / 2, -np.pi / 2, 0, 0])  # 0])

    theta = theta_offsets + theta_joints
    progressive_Transforms = []
    progressive_Transform = base_tform

    for i in range(5):
        ct = np.cos(theta[i])
        st = np.sin(theta[i])
        ca = np.cos(alpha[i])
        sa = np.sin(alpha[i])
        a = a_off[i]
        d = d_off[i]

        A = np.array([
            [ct, -st, 0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0, 0, 0, 1]
        ])

        progressive_Transform = np.dot(progressive_Transform, A)
        progressive_Transforms.append(progressive_Transform)

    return np.stack(progressive_Transforms, axis=0)
