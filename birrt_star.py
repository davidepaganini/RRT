import numpy as np
from robot import get_robot_cylinders_and_spheres
from collision import is_in_collision
from graphics import create_animation
from scipy.interpolate import CubicSpline


class TreeNode:
    def __init__(self, joints):
        self.joints = joints
        self.parent = None
        self.cost = 0.0


def get_min_angular_distance_from_a_to_b(angle_a, angle_b):
    angle_a %= 2 * np.pi
    angle_b %= 2 * np.pi

    angle_a = angle_a + 2 * np.pi if angle_a < 0 else angle_a
    angle_b = angle_b + 2 * np.pi if angle_b < 0 else angle_b

    if angle_a > angle_b:
        angle_difference = angle_a - angle_b
        if angle_difference >= np.pi:
            return (2 * np.pi) - angle_difference
        else:
            return -angle_difference
    else:
        angle_difference = angle_b - angle_a
        if angle_difference >= np.pi:
            return angle_difference - (2 * np.pi)
        else:
            return angle_difference


def get_nodes_distance_and_vector(from_node, to_node):
    distance_vector = [get_min_angular_distance_from_a_to_b(angle_a, angle_b)
                       for angle_a, angle_b in zip(from_node.joints, to_node.joints)]
    return np.linalg.norm(distance_vector), np.array(distance_vector)


def get_random_node():
    return TreeNode(np.random.uniform(low=-np.pi, high=np.pi, size=(5,)))


def wrap_joints(joints):
    for index, joint in enumerate(joints):
        if joint >= np.pi:
            joints[index] = -2 * np.pi + joint
        elif joint <= -np.pi:
            joints[index] = 2 * np.pi - joint
    return joints


def steer_node(from_node, to_node, max_distance=1):
    distance, distance_vec = get_nodes_distance_and_vector(from_node, to_node)
    if distance <= max_distance:
        to_node.cost = from_node.cost + distance
        return to_node
    else:
        scaled_vector = max_distance * distance_vec / distance
        steered_joints = from_node.joints + scaled_vector
        new_node = TreeNode(wrap_joints(steered_joints))
        new_node.cost = from_node.cost + max_distance
        return new_node


def nearest_node(tree, to_node):
    distances = [get_nodes_distance_and_vector(tree_node, to_node)[0] for tree_node in tree]
    min_distance_index = np.argmin(distances)
    return tree[min_distance_index]


def find_near_nodes(tree, to_node, rewire_radius=0.8):
    return [node for node in tree if get_nodes_distance_and_vector(node, to_node)[0] < rewire_radius]


def rewire(near_nodes, new_node, obstacles):
    for near_node in near_nodes:
        cost = near_node.cost + get_nodes_distance_and_vector(near_node, new_node)[0]
        if cost < new_node.cost and is_collision_free(near_node, new_node, obstacles):
            new_node.parent = near_node
            new_node.cost = cost


def is_collision_free(from_node, to_node, obstacles, base_tform=np.eye(4), num_points=20):
    _, distance_vector = get_nodes_distance_and_vector(from_node, to_node)
    increments = np.linspace(np.array([0, 0, 0, 0, 0]), distance_vector, num_points)
    for index in range(1, increments.shape[0]):
        cylinders, _ = get_robot_cylinders_and_spheres(from_node.joints + increments[index], base_tform)
        in_collision = is_in_collision(cylinders, obstacles)
        if in_collision:
            return False
    return True


def is_problem_valid(start_node, end_node, obstacles, base_tform=np.eye(4)):
    start_cylinders, _ = get_robot_cylinders_and_spheres(start_node.joints, base_tform)
    in_collision = is_in_collision(start_cylinders, obstacles)
    end_cylinders, _ = get_robot_cylinders_and_spheres(end_node.joints, base_tform)
    in_collision = in_collision or is_in_collision(end_cylinders, obstacles)
    return not in_collision


def check_path_valid(path_joints, obstacles, base_tform=np.eye(4)):
    for config in path_joints:
        robot_cylinders, _ = get_robot_cylinders_and_spheres(config, base_tform)
        in_collision = is_in_collision(robot_cylinders, obstacles)
        if in_collision:
            print("Collision")
        else:
            print("Valid")


def birrt_star(start_joints, end_joints, obstacles, base_tform=np.eye(4), max_iter=2000, max_distance=0.5):
    start_tree = [TreeNode(np.array([0, 0, 0, 0, 0, 0])) for _ in range(max_iter)]
    end_tree = [TreeNode(np.array([0, 0, 0, 0, 0, 0])) for _ in range(max_iter)]

    start_tree[0] = TreeNode(start_joints)
    end_tree[0] = TreeNode(end_joints)

    print(f"The problem is valid: {is_problem_valid(start_tree[0], end_tree[0], obstacles)}")
    nodes_in_start = 1
    nodes_in_end = 1
    for _ in range(max_iter):
        random_node_start = get_random_node()
        nearest_start_node = nearest_node(start_tree[:nodes_in_start], random_node_start)
        new_start_node = steer_node(nearest_start_node, random_node_start, max_distance=max_distance)
        if is_collision_free(nearest_start_node, new_start_node, obstacles, base_tform):
            start_tree[nodes_in_start] = new_start_node
            new_start_node.parent = nearest_start_node
            nodes_in_start += 1
            start_near_nodes = find_near_nodes(start_tree[:nodes_in_start], new_start_node, rewire_radius=1)
            rewire(start_near_nodes, new_start_node, obstacles)
            growth_start = True
        else:
            growth_start = False

        random_node_end = get_random_node()
        nearest_end_node = nearest_node(end_tree[:nodes_in_end], random_node_end)
        new_end_node = steer_node(nearest_end_node, random_node_end, max_distance=max_distance)
        if is_collision_free(nearest_end_node, new_end_node, obstacles, base_tform):
            end_tree[nodes_in_end] = new_end_node
            new_end_node.parent = nearest_end_node
            nodes_in_end += 1
            end_near_nodes = find_near_nodes(end_tree[:nodes_in_end], new_end_node, rewire_radius=1)
            rewire(end_near_nodes, new_end_node, obstacles)
            growth_end = True
        else:
            growth_end = False

        if growth_start:
            nearest_end_node_to_new_start = nearest_node(end_tree[:nodes_in_end], new_start_node)
            distance_end_node_to_new_start = \
                get_nodes_distance_and_vector(new_start_node, nearest_end_node_to_new_start)[0]
            distance_check = distance_end_node_to_new_start < max_distance
            collision_check = is_collision_free(new_start_node, nearest_end_node_to_new_start,
                                                obstacles, base_tform, num_points=50)
            if collision_check:
                if distance_check:
                    print("Path Found!")
                    closest_start = new_start_node
                    closest_end = nearest_end_node_to_new_start
                    break
                else:
                    new_end_node = steer_node(nearest_end_node_to_new_start, new_start_node, max_distance=max_distance)
                    end_tree[nodes_in_end] = new_end_node
                    new_end_node.parent = nearest_end_node_to_new_start
                    nodes_in_end += 1

        if growth_end:
            nearest_start_node_to_new_end = nearest_node(start_tree[:nodes_in_start], new_end_node)
            distance_start_node_to_new_end = get_nodes_distance_and_vector(new_end_node, nearest_start_node_to_new_end)[
                0]
            distance_check = distance_start_node_to_new_end < max_distance
            collision_check = is_collision_free(new_end_node, nearest_start_node_to_new_end, obstacles,
                                                base_tform, num_points=50)
            if collision_check:
                if distance_check:
                    print("Path Found!")
                    closest_start = nearest_start_node_to_new_end
                    closest_end = new_end_node
                    break
                else:
                    new_start_node = steer_node(nearest_start_node_to_new_end, new_end_node, max_distance=max_distance)
                    start_tree[nodes_in_start] = new_start_node
                    new_start_node.parent = nearest_start_node_to_new_end
                    nodes_in_start += 1
        print(_)

    path_start = []
    current_node_start = closest_start
    while current_node_start is not None:
        path_start.append(np.copy(current_node_start.joints))
        current_node_start = current_node_start.parent
        if current_node_start == None:
            a = 1
    print("Backtracked to start")
    path_end = []
    current_node_end = closest_end
    while current_node_end is not None:
        path_end.append(np.copy(current_node_end.joints))
        current_node_end = current_node_end.parent
        if current_node_end is None:
            a = 2
    print("Backtracked to goal")
    path = []
    path.extend(path_start[::-1])
    path.extend(path_end)
    print("Joined the two paths!")

    return path


def simplify_path(path, obstacles, base_tform=np.eye(4)):
    max_rad_step = 0.01
    longer_path = [path[0]]
    for index, joints in enumerate(path):
        if index != 0:
            from_node = TreeNode(path[index - 1])
            to_node = TreeNode(path[index])
            _, distance_vector = get_nodes_distance_and_vector(from_node, to_node)
            max_joint_distance = np.max(np.abs(distance_vector))
            increments = np.linspace(np.array([0, 0, 0, 0, 0]), distance_vector, 10)
            for index in range(1, increments.shape[0]):
                longer_path.append(from_node.joints + increments[index])
    print("Discretised the path!")
    longer_path = np.vstack(longer_path).tolist()

    viable_moves = [0]
    while viable_moves[-1] < len(longer_path) - 1:
        from_node = TreeNode(longer_path[viable_moves[-1]])
        next_index = viable_moves[-1] + 1
        viable_moves.append(next_index)
        for index_ahead in range(next_index, len(longer_path) - 1):
            to_node = TreeNode(longer_path[index_ahead])
            _, distance_vector = get_nodes_distance_and_vector(from_node, to_node)
            max_joint_distance = np.max(np.abs(distance_vector))
            if max_joint_distance / max_rad_step < 1:
                num_discrete_points = 3
            else:
                num_discrete_points = 3 + np.ceil(max_joint_distance / max_rad_step).astype(int)
            collision_check = is_collision_free(from_node, to_node, obstacles, base_tform,
                                                num_points=num_discrete_points)
            if collision_check:
                viable_moves[-1] = index_ahead
    simplified_path = []
    for index in viable_moves:
        simplified_path.append(longer_path[index])
    print("Simplified the path!")
    return simplified_path


def create_spline(path):
    x = np.arange(len(path))
    splines = []
    for joint_idx in range(5):
        joint_path = [config[joint_idx] for config in path]
        joint_spline = CubicSpline(x, joint_path)
        joint_values = joint_spline(x)
        splines.append(joint_values)
        print(f"Spline for joint {joint_idx + 1}")
    new_path = []
    for spline_idx in range(len(path)):
        config = np.array([splines[joint_idx][spline_idx] for joint_idx in range(5)])
        new_path.append(config)
    return new_path
    print("Completed spline!")


if __name__ == '__main__':
    start_joints = np.array([1.0, -np.pi / 3, -np.pi / 3, 0.0, 0.0])
    end_joints = np.array([0, -np.pi / 2, 0, 0, 0])
    obstacles = {"cylinders": None, "spheres": [], "cuboids": None}
    cylinders_start, _ = get_robot_cylinders_and_spheres(start_joints)
    cylinders_goal, _ = get_robot_cylinders_and_spheres(end_joints)
    while len(obstacles["spheres"]) < 25:
        possible_sphere = {"center": np.random.uniform(low=-1.5, high=1.5, size=(3,)), "color": "g", "radius": 0.25}
        is_collision_start = is_in_collision(cylinders_start,
                                             {"cylinders": None, "spheres": [possible_sphere], "cuboids": None})
        is_collision_goal = is_in_collision(cylinders_goal,
                                            {"cylinders": None, "spheres": [possible_sphere], "cuboids": None})
        if not is_collision_start and not is_collision_goal:
            obstacles["spheres"].append(possible_sphere)
    # obstacles["cuboids"] = [{"min_dim": np.array([2, -0.75, 1]), "max_dim": np.array([2.25, 0.25, 1.25]), "color": "c"},
    #                         {"min_dim": np.array([2, -0.75, 0]), "max_dim": np.array([2.25, 0.25, 0.25]), "color": "c"},
    #                         {"min_dim": np.array([2, -1, 0]), "max_dim": np.array([2.25, -0.75, 1.25]), "color": "c"},
    #                         {"min_dim": np.array([2, 0.25, 0]), "max_dim": np.array([2.25, 0.5, 1.25]), "color": "c"}]
    # obstacles["cuboids"] = [{"min_dim": np.array([2, -1, -2]), "max_dim": np.array([2.25, -0.75, 2.25]), "color": "c"},
    #                         {"min_dim": np.array([2, 0.25, -2]), "max_dim": np.array([2.25, 0.5, 2.25]), "color": "c"}]

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
    create_animation(simplified_longer_path, obstacles, "test.gif")
