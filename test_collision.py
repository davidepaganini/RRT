import numpy as np
from robot import get_robot_cylinders_and_spheres
import matplotlib.pyplot as plt
from graphics import create_animation, render_scene, visualise_path
from birrt_star import birrt_star, simplify_path, TreeNode, get_nodes_distance_and_vector, check_path_valid, \
    create_spline, is_in_collision

if __name__ == '__main__':
    start_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    end_joints = np.array([0.0, -1.5 * np.pi / 2, 0.0, 0.0, 0.0])
    obstacles = {"cylinders": None, "spheres": [], "cuboids": None}
    cylinders_start, _ = get_robot_cylinders_and_spheres(start_joints)
    cylinders_goal, _ = get_robot_cylinders_and_spheres(end_joints)
    # while len(obstacles["spheres"]) < 5:
    #     possible_sphere = {"center": np.random.uniform(low=-1.5, high=1.5, size=(3,)), "color": "g", "radius": 0.25}
    #     is_collision_start = is_in_collision(cylinders_start,
    #                                          {"cylinders": None, "spheres": [possible_sphere], "cuboids": None})
    #     is_collision_goal = is_in_collision(cylinders_goal,
    #                                         {"cylinders": None, "spheres": [possible_sphere], "cuboids": None})
    #     if not is_collision_start and not is_collision_goal:
    #         obstacles["spheres"].append(possible_sphere)

    possible_sphere = {"center": np.array([0.75,0,2.5]), "color": "g", "radius": 0.25}
    is_collision_start = is_in_collision(cylinders_start,
                                         {"cylinders": None, "spheres": [possible_sphere], "cuboids": None})
    is_collision_goal = is_in_collision(cylinders_goal,
                                        {"cylinders": None, "spheres": [possible_sphere], "cuboids": None})
    if not is_collision_start and not is_collision_goal:
        obstacles["spheres"].append(possible_sphere)

    possible_sphere = {"center": np.array([-0.75,0,2.5]), "color": "g", "radius": 0.25}
    is_collision_start = is_in_collision(cylinders_start,
                                         {"cylinders": None, "spheres": [possible_sphere], "cuboids": None})
    is_collision_goal = is_in_collision(cylinders_goal,
                                        {"cylinders": None, "spheres": [possible_sphere], "cuboids": None})
    if not is_collision_start and not is_collision_goal:
        obstacles["spheres"].append(possible_sphere)

    possible_sphere = {"center": np.array([0,-0.75,2.5]), "color": "g", "radius": 0.25}
    is_collision_start = is_in_collision(cylinders_start,
                                         {"cylinders": None, "spheres": [possible_sphere], "cuboids": None})
    is_collision_goal = is_in_collision(cylinders_goal,
                                        {"cylinders": None, "spheres": [possible_sphere], "cuboids": None})
    if not is_collision_start and not is_collision_goal:
        obstacles["spheres"].append(possible_sphere)

    possible_sphere = {"center": np.array([0,0.75,2.5]), "color": "g", "radius": 0.25}
    is_collision_start = is_in_collision(cylinders_start,
                                         {"cylinders": None, "spheres": [possible_sphere], "cuboids": None})
    is_collision_goal = is_in_collision(cylinders_goal,
                                        {"cylinders": None, "spheres": [possible_sphere], "cuboids": None})
    if not is_collision_start and not is_collision_goal:
        obstacles["spheres"].append(possible_sphere)

    path = birrt_star(start_joints, end_joints, obstacles)
    simplified_path = simplify_path(path, obstacles)

    simplified_longer_path = []
    simplified_longer_path.append(simplified_path[0])
    for index1, joints in enumerate(simplified_path):
        if index1 != 0:
            from_node = TreeNode(simplified_path[index1 - 1])
            to_node = TreeNode(simplified_path[index1])
            _, distance_vector = get_nodes_distance_and_vector(from_node, to_node)
            increments = np.linspace(np.array([0, 0, 0, 0, 0]), distance_vector, 10)
            for index2 in range(1, increments.shape[0]):
                simplified_longer_path.append(from_node.joints + increments[index2])
    print("--- PATH ---")
    check_path_valid(path, obstacles)
    print("--- SIMPLIFIED PATH ---")
    check_path_valid(simplified_path, obstacles)
    print("--- SIMPLIFIED PATH ---")
    check_path_valid(simplified_longer_path, obstacles)
    # visualise_path(simplified_longer_path, obstacles)
    splined_path = create_spline(simplified_longer_path)

    create_animation(splined_path, obstacles, "spheres.gif")
