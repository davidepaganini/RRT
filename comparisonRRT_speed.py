from robot import get_robot_cylinders_and_spheres
from Extended_RRT import ExtendedRRT, is_in_collision
from copy import deepcopy
import numpy as np

environments = [{"cylinders": [], "spheres": [], "cuboids": []} for i in range(10)]
start_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
end_joints = np.array([0.0, -1.5 * np.pi / 2, 0.0, 0.0, 0.0])
cylinders_start, _ = get_robot_cylinders_and_spheres(start_joints)
cylinders_goal, _ = get_robot_cylinders_and_spheres(end_joints)

for i in range(10):
    found_sphere = False
    while not found_sphere:
        possible_sphere = {"center": np.random.uniform(low=-1.5, high=1.5, size=(3,)), "color": "g", "radius": 0.25}
        is_collision_start = is_in_collision(cylinders_start,{"cylinders": None, "spheres": [possible_sphere], "cuboids": None})
        is_collision_goal = is_in_collision(cylinders_goal,{"cylinders": None, "spheres": [possible_sphere], "cuboids": None})
        if not is_collision_start and not is_collision_goal:
            environments[i]["spheres"].append(possible_sphere)
            found_sphere = True
            print("Sphere!")

    found_cuboid = False
    while not found_cuboid:
        min_dim_cuboid = np.random.uniform(low=-1.5, high=1.5, size=(3,))
        max_dim_cuboid = min_dim_cuboid + np.random.uniform(low=0.0, high=0.5, size=(3,))
        possible_cuboid = {"min_dim": min_dim_cuboid, "max_dim": max_dim_cuboid}
        is_collision_start = is_in_collision(cylinders_start, {"cylinders": None, "spheres": None, "cuboids": [possible_cuboid]})
        is_collision_goal = is_in_collision(cylinders_goal, {"cylinders": None, "spheres": None, "cuboids": [possible_cuboid]})
        if not is_collision_start and not is_collision_goal:
            environments[i]["cuboids"].append(possible_cuboid)
            found_cuboid = True
            print("Cuboid!")

    found_cylinder = False
    while not found_cylinder:
        point_a = np.random.uniform(low=-1.5, high=1.5, size=(3,))
        axis = np.random.uniform(low=-1.0, high=1.0, size=(3,))
        axis = 0.75*axis / np.linalg.norm(axis)
        possible_cylinder = {"point_a": point_a, "point_b": point_a+axis, "radius":0.25, "discrete": None, "color": "b"}
        is_collision_start = is_in_collision(cylinders_start,
                                             {"cylinders": [possible_cylinder], "spheres": None, "cuboids": None})
        is_collision_goal = is_in_collision(cylinders_goal,
                                            {"cylinders": [possible_cylinder], "spheres": None, "cuboids": None})
        if not is_collision_start and not is_collision_goal:
            environments[i]["cylinders"].append(possible_cylinder)
            found_cylinder = True
            print("Cylinder!")

if __name__ == '__main__':
    successes_rrt = 0
    successes_birrt = 0
    num_nodes_rrt = 0
    num_nodes_birrt = 0
    for environment in environments:
        rrt = ExtendedRRT(environment, star=False, bidirectional=False, prune=False)
        rrt.set_problem(start_joints, end_joints)
        rrt.solve()
        num_nodes_rrt += rrt.nodes_in_start + rrt.nodes_in_end
        if rrt.solved:
            successes_rrt += 1

        birrt = ExtendedRRT(environment, star=False, bidirectional=True, prune=False)
        birrt.set_problem(start_joints, end_joints)
        birrt.solve()
        num_nodes_birrt += birrt.nodes_in_start + birrt.nodes_in_end
        if birrt.solved:
            successes_birrt += 1

    print("-----")
    print(f"RRT Successes : {successes_rrt}")
    print(f"RRT Total Nodes : {num_nodes_rrt}")
    print(f"BIRRT Successes : {successes_birrt}")
    print(f"BIRRT Total Nodes : {num_nodes_birrt}")
