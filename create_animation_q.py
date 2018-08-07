#!/usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation

show = 'q'
print('\nCreating Animation of {}'.format(show))

# set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots(1, figsize=(9,5))

ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
ax.set_xlabel('Lat')
array = np.load('q_array.npz')['arr_0']
ax.set_ylabel("q (g / g)")
ax.set_title('EBM frame = 0')
plt.tight_layout(pad=3)

im = ax.imshow(array[0, :, :].T, extent=(-90, 90, 1000, 0), aspect=0.1, cmap='BrBG', origin='lower', animated=True)
plt.colorbar(im, ax=ax)

def animate(i):
    if i%100 == 0: 
        print("{}/{} frames".format(i, len(array)))
    ax.set_title('EBM frame = {}'.format(i + 1))
    graph = array[i, :, :].T
    im.set_array(graph)
    return im,

anim = animation.FuncAnimation(fig, animate, frames=len(array), interval=int(8000/len(array)), blit=True)

fname = '{}_anim.mp4'.format(show)
anim.save(fname)
print('{} created.'.format(fname))
