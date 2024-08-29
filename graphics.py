import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
from robot import get_robot_cylinders_and_spheres


def plot_sphere(ax, sphere):
    center = sphere["center"]
    color = sphere["color"]
    radius = sphere["radius"]
    u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=0.5)


def plot_cylinder(ax, cylinder):
    radius = cylinder["radius"]
    cylinder_vector = cylinder["point_b"] - cylinder["point_a"]
    cylinder_length = np.linalg.norm(cylinder_vector)
    cylinder_vector = cylinder_vector / cylinder_length
    not_v = np.array([1, 0, 0])
    # if (cylinder_vector == not_v).all():
    #     not_v = np.array([0, 1, 0])
    # n1 = np.cross(cylinder_vector, not_v)
    # n1 /= np.linalg.norm(n1)
    found_non_parallel = False
    while not found_non_parallel:
        not_v = np.random.rand(3, )
        not_v = not_v / np.linalg.norm(not_v)
        n1 = np.cross(cylinder_vector, not_v)
        if (n1 != np.array([0.0, 0.0, 0.0])).any():
            n1 /= np.linalg.norm(n1)
            found_non_parallel = True

    n2 = np.cross(cylinder_vector, n1)
    t = np.array([0, cylinder_length])
    theta = np.linspace(0, 2 * np.pi, 30)
    t, theta = np.meshgrid(t, theta)
    # generate coordinates for surface
    X, Y, Z = [cylinder["point_a"][i] +
               cylinder_vector[i] * t +
               radius * np.sin(theta) * n1[i] +
               radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    ax.plot_surface(X, Y, Z, color=cylinder["color"], alpha=0.5)


def plot_cuboid(ax, cuboid):
    min_x, min_y, min_z = cuboid["min_dim"]
    max_x, max_y, max_z = cuboid["max_dim"]
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    X[:, :, 0] *= abs(max_x - min_x)
    X[:, :, 1] *= abs(max_y - min_y)
    X[:, :, 2] *= abs(max_z - min_z)
    X += cuboid["min_dim"]
    collection = Poly3DCollection(X, alpha=0.1, edgecolors="k")
    collection.set_color(cuboid["color"])
    collection.set_edgecolor("k")
    ax.add_collection(collection)


def render_scene(ax, cylinders=None, spheres=None, cuboids=None):
    if cylinders is not None:
        for cylinder in cylinders:
            plot_cylinder(ax, cylinder)

    if spheres is not None:
        for sphere in spheres:
            plot_sphere(ax, sphere)

    if cuboids is not None:
        for cuboid in cuboids:
            plot_cuboid(ax, cuboid)
    plt.axis('equal')


def visualise_path(path, obstacles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    render_scene(ax, obstacles["cylinders"], obstacles["spheres"], obstacles["cuboids"])
    for config in path:
        cylinders, spheres = get_robot_cylinders_and_spheres(config)
        render_scene(ax, cylinders, spheres)
    plt.show()


def create_animation(path, obstacles, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_xlim([-2, 2])
    cylinders_robot, spheres_robot = get_robot_cylinders_and_spheres(path[0])
    spheres = []
    spheres.extend(spheres_robot)
    if obstacles["spheres"] is not None:
        spheres.extend(obstacles["spheres"])
    render_scene(ax, cylinders=cylinders_robot, spheres=spheres, cuboids=obstacles["cuboids"])

    initial_xlim = ax.get_xlim()
    initial_ylim = ax.get_ylim()
    initial_zlim = ax.get_zlim()

    def update(frame):
        ax.clear()
        cylinders = []
        # cylinders_start, spheres_start = get_robot_cylinders_and_spheres(path[0], color="c")
        cylinders_end, spheres_end = get_robot_cylinders_and_spheres(path[-1], color="r")
        # cylinders.extend(cylinders_start)
        cylinders.extend(cylinders_end)
        spheres = []
        # spheres.extend(spheres_start)
        spheres.extend(spheres_end)
        cylinders_robot, spheres_robot = get_robot_cylinders_and_spheres(path[frame])
        spheres.extend(spheres_robot)
        if obstacles["spheres"] is not None:
            spheres.extend(obstacles["spheres"])
        cylinders.extend(cylinders_robot)
        ax.set_xlim(initial_xlim)
        ax.set_ylim(initial_ylim)
        ax.set_zlim(initial_zlim)
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_xlim([-2, 2])

        render_scene(ax, cylinders, spheres, obstacles["cuboids"])
        eex = []
        eey = []
        eez = []
        coord_to_plot = []
        for sphere_idx in range(len(spheres_robot)):
            coord_to_plot.append([[], [], []])

        for index in range(frame):
            cylinders_robot, spheres_robot = get_robot_cylinders_and_spheres(path[index])
            eex.append(spheres_robot[-1]["center"][0])
            eey.append(spheres_robot[-1]["center"][1])
            eez.append(spheres_robot[-1]["center"][2])
            for sphere_idx in range(len(spheres_robot)):
                coord_to_plot[sphere_idx][0].append(spheres_robot[sphere_idx]["center"][0])
                coord_to_plot[sphere_idx][1].append(spheres_robot[sphere_idx]["center"][1])
                coord_to_plot[sphere_idx][2].append(spheres_robot[sphere_idx]["center"][2])

        for sphere_idx in range(len(spheres_robot)):
            ax.plot(coord_to_plot[sphere_idx][0], coord_to_plot[sphere_idx][1],coord_to_plot[sphere_idx][2])
        # ax.plot(eex, eey, eez)

        ax.view_init(elev=12, azim=40)


    ani = FuncAnimation(fig, update, frames=len(path), repeat=False)
    ani.save(name, writer='imagemagick')
    plt.show()
