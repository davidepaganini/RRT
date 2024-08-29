import numpy as np


def collision_cylinder_sphere(cylinder, sphere):
    ab = cylinder["point_b"] - cylinder["point_a"]
    ac = sphere["center"] - cylinder["point_a"]
    bc = sphere["center"] - cylinder["point_b"]
    e = np.dot(ac, ab)
    f = np.dot(ab, ab)
    if e < 0:
        distance = np.dot(ac, ac)
    elif e > f:
        distance = np.dot(bc, bc)
    else:
        distance = np.dot(ac, ac)
        distance -= e * e / f
    radius = cylinder["radius"] + sphere["radius"]
    if distance < radius * radius:
        return True
    else:
        return False


def discretise_cylinder(cylinder):
    length_axis = np.linalg.norm(cylinder["point_b"] - cylinder["point_a"])
    num_points = 2
    if length_axis >= 2 * cylinder["radius"]:
        num_points = np.ceil((length_axis - 2 * cylinder["radius"]) / (2 * cylinder["radius"])).astype(int)
    num_points *= 2
    centers = np.linspace(cylinder["point_a"], cylinder["point_b"], num_points + 2)
    cylinder["discrete"] = []
    for index in range(centers.shape[0]):
        discrete_sphere = {"center": centers[index, :], "color": cylinder["color"], "radius": cylinder["radius"]}
        cylinder["discrete"].append(discrete_sphere)


def collision_cylinder_cylinder(cylinder_a, cylinder_b):
    if cylinder_b["discrete"] is None:
        discretise_cylinder(cylinder_b)

    for discrete_sphere in cylinder_b["discrete"]:
        in_collision = collision_cylinder_sphere(cylinder_a, discrete_sphere)
        if in_collision:
            return True
    return False


def collision_cuboid_sphere(cuboid, sphere):
    min_thickened_cuboid = cuboid["min_dim"] - sphere["radius"]
    max_thickened_cuboid = cuboid["max_dim"] + sphere["radius"]

    overlap_x = min_thickened_cuboid[0] <= sphere["center"][0] <= max_thickened_cuboid[0]
    overlap_y = min_thickened_cuboid[1] <= sphere["center"][1] <= max_thickened_cuboid[1]
    overlap_z = min_thickened_cuboid[2] <= sphere["center"][2] <= max_thickened_cuboid[2]

    return overlap_x and overlap_y and overlap_z


def collision_cylinder_cuboid(cylinder, cuboid):
    if cylinder["discrete"] is None:
        discretise_cylinder(cylinder)

    for discrete_sphere in cylinder["discrete"]:
        in_collision = collision_cuboid_sphere(cuboid, discrete_sphere)
        if in_collision:
            return True
    return False


def is_in_collision(robot_cylinders, obstacles):
    for robot_cylinder in reversed(robot_cylinders):
        if obstacles["cylinders"] is not None:
            for obstacle_cylinder in obstacles["cylinders"]:
                cylinder_cylinder_collision = collision_cylinder_cylinder(robot_cylinder, obstacle_cylinder)
                if cylinder_cylinder_collision:
                    return True
        if obstacles["spheres"] is not None:
            for obstacle_sphere in obstacles["spheres"]:
                cylinder_sphere_collision = collision_cylinder_sphere(robot_cylinder, obstacle_sphere)
                if cylinder_sphere_collision:
                    return True
        if obstacles["cuboids"] is not None:
            for obstacle_cuboid in obstacles["cuboids"]:
                cylinder_cuboid_collision = collision_cylinder_cuboid(robot_cylinder, obstacle_cuboid)
                if cylinder_cuboid_collision:
                    return True

    return False
