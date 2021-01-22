from mpl_toolkits import mplot3d
# matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

def visualize(file_name, data):
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    fig = plt.figure(figsize=(9, 6))
    # Create 3D container
    ax = plt.axes(projection = '3d')
    # Visualize 3D scatter plot
    ax.scatter3D(data[0], data[1], data[2])
    # print(data[0])
    # Give labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(30, 320)
    # Save figure
    ax.set_xlim([120, 160])
    ax.set_ylim([700, 800])
    ax.set_zlim([20, 80])
    plt.savefig('visualization/' + file_name + '.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
    # plt.show()