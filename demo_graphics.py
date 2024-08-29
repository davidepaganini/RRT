from robot import get_robot_cylinders_and_spheres
from collision import discretise_cylinder
from graphics import render_scene
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    robot_cylinders, _ = get_robot_cylinders_and_spheres(np.array([0, -np.pi/2, 0, 0, 0]), color="r")
    cuboids = [{"min_dim": np.array([1, -0.75, 0]), "max_dim": np.array([1.25, 0.25, 0.25]), "color": "c"},
               {"min_dim": np.array([1, -0.75, 0.75]), "max_dim": np.array([1.25, 0.25, 1]), "color": "c"},
               {"min_dim": np.array([1, -1, 0]), "max_dim": np.array([1.25, -0.75, 1]), "color": "c"},
               {"min_dim": np.array([1, 0.25, 0]), "max_dim": np.array([1.25, 0.5, 1]), "color": "c"}]
    render_scene(ax, cylinders=robot_cylinders, cuboids=cuboids)

    plt.show()
