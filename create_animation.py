#!/usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation

# show = 'T'
# show = 'E'
# show = 'alb'
show = 'L'
print('\nCreating Animation of {}'.format(show))

# set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots(1, figsize=(9,5))

ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
ax.set_xlabel('Latitude (degrees)')
if show == 'T':
    array = np.load('T_array.npz')['arr_0']
    ax.set_ylabel('T (K)')
elif show == 'E':
    array = np.load('E_array.npz')['arr_0']
    ax.set_ylabel("W/m$^2$")
elif show == 'alb':
    array = np.load('alb_array.npz')['arr_0']
    ax.set_ylabel("$\\alpha$")
elif show == 'L':
    array = np.load('L_array.npz')['arr_0']
    ax.set_ylabel("W/m$^2$")
ax.set_title('EBM frame = 0')
plt.tight_layout(pad=3)

lats = np.linspace(-90, 90, array.shape[1])

line, = ax.plot(lats, array[0, :], 'b')

def init():
    line.set_data(lats, array[0, :])
    return (line,)

def animate(i):
    if i%100 == 0: 
        print("{}/{} frames".format(i, len(array)))
    ax.set_title('EBM frame = {}'.format(i + 1))
    graph = array[i, :]
    line.set_data(lats, graph)
    m = graph.min()
    M = graph.max()
    ax.set_ylim([m - 0.01*np.abs(m), M + 0.01*np.abs(M)])
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(array), interval=int(8000/len(array)), blit=True)

fname = '{}_anim.mp4'.format(show)
anim.save(fname)
print('{} created.'.format(fname))
