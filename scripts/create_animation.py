#!/usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation

plt.style.use("/home/hpeter/Documents/ResearchBoos/EBM_files/EBM/plot_styles.mplstyle")

show = 'T'
print('\nCreating Animation of {}'.format(show))

# set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots(1)

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_xlabel('Latitude')
if show == 'T':
    array = np.load('simulation_data.npz')['T']
    ax.set_ylabel('$T$ (K)')
ax.set_title('EBM frame = 0')
plt.tight_layout(pad=3)

dx = 2 / (array.shape[1] - 1)
sin_lats = np.linspace(-1.0, 1.0, array.shape[1])

line, = ax.plot(sin_lats, array[0, :], 'b')

def init():
    line.set_data(sin_lats, array[0, :])
    return (line,)

def animate(i):
    if i%100 == 0: 
        print("{}/{} frames".format(i, len(array)))
    ax.set_title('EBM frame = {}'.format(i + 1))
    graph = array[i, :]
    line.set_data(sin_lats, graph)
    m = graph.min()
    M = graph.max()
    ax.set_ylim([m - 0.01*np.abs(m), M + 0.01*np.abs(M)])
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(array), interval=int(8000/len(array)), blit=True)

fname = '{}_anim.mp4'.format(show)
anim.save(fname)
print('{} created.'.format(fname))
