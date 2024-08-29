import numpy as np
from robot import get_robot_cylinders_and_spheres
import matplotlib.pyplot as plt
from graphics import create_animation, render_scene, visualise_path
from birrt_star import birrt_star, simplify_path, TreeNode, get_nodes_distance_and_vector, check_path_valid


if __name__ == '__main__':
    start_joints = np.array([0.0, -np.pi/2, -np.pi, 0.0, -np.pi])
    end_joints = np.array([0, -np.pi/2, 0, -np.pi/2, -np.pi/2])
    obstacles = {"cylinders": None, "spheres": [], "cuboids": None}
    cylinders_start, _ = get_robot_cylinders_and_spheres(start_joints)
    cylinders_goal, _ = get_robot_cylinders_and_spheres(end_joints)
    cylinders = []
    cylinders.extend(cylinders_start)
    cylinders.extend(cylinders_goal)

    cuboids = [{"min_dim": np.array([-0.5, -0.65, -0.3]), "max_dim": np.array([1.75, 0.35, -0.1]), "color": "c"},
               {"min_dim": np.array([-0.5, -0.65, -0.3]), "max_dim": np.array([-0.25, 0.35, 1]), "color": "c"},
               {"min_dim": np.array([-0.5, -0.75, -0.3]), "max_dim": np.array([1.75, -0.65, 1]), "color": "c"},
               {"min_dim": np.array([-0.5, 0.35, -0.3]), "max_dim": np.array([1.75, 0.45, 1]), "color": "c"},
               {"min_dim": np.array([1.5, -0.75, 1]), "max_dim": np.array([1.75, 0.45, 1.2]), "color": "c"},
               {"min_dim": np.array([1.5, -0.75, -0.3]), "max_dim": np.array([1.75, 0.45, 0.2]), "color": "c"}]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # render_scene(ax, cylinders=cylinders, cuboids=cuboids)
    # plt.show()

    obstacles = {"cylinders": None, "spheres": None, "cuboids": cuboids}
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

    create_animation(simplified_longer_path, obstacles, "demarlus.gif")