#!/usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation, rc

### STYLES
rc('animation', html='html5')
rc('lines', linewidth=2, color='b', markersize=10)
rc('axes', titlesize=20, labelsize=16, xmargin=0.01, ymargin=0.01, 
        linewidth=1.5)
rc('axes.spines', top=False, right=False)
rc('xtick', labelsize=13)
rc('xtick.major', size=5, width=1.5)
rc('ytick', labelsize=13)
rc('ytick.major', size=5, width=1.5)
rc('legend', fontsize=14)

# show = 'T'
# show = 'E'
show = 'alb'
# show = 'L'
print('\nCreating Animation of {}'.format(show))

# set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots(1, figsize=(16, 10))

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(['-90', '', '', '-60', '', '', '-30', '', '', 'EQ', '', '', '30', '', '', '60', '', '', '90'])
ax.set_xlabel('Lat')
if show == 'T':
    array = np.load('T_array.npz')['arr_0']
    ax.set_ylabel('T (K)')
elif show == 'E':
    array = np.load('E_array.npz')['arr_0']
    ax.set_ylabel("J/kg")
elif show == 'alb':
    array = np.load('alb_array.npz')['arr_0']
    ax.set_ylabel("$\\alpha$")
elif show == 'L':
    array = np.load('L_array.npz')['arr_0']
    ax.set_ylabel("W/m$^2$")
ax.set_title('EBM frame = 0')
plt.tight_layout(pad=3)

dx = 2 / array.shape[1]
sin_lats = np.linspace(-1.0 + dx/2, 1.0 - dx/2, array.shape[1])

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
