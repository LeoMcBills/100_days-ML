# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from matplotlib.widgets import Slider, Button


vol = io.imread("attention-mri.tif")
volume = vol.T
nb_frames = volume.shape[0]

fig, ax = plt.subplots(figsize=(8, 8))
img = ax.imshow(volume[0], cmap='gray', vmin=0, vmax=200)
ax.set_title('Slices in volumetric data @ MRI-Uganda')
ax.axis('off')  # Hide the axes

def update_slice(k):
    img.set_data(volume[k])
    plt.draw()
    plt.pause(0.1) 

def play_animation(event):
    for k in range(nb_frames):
        update_slice(k)
        if not is_playing:
            break
    if is_playing:
        play_animation(event)

def update(val):
    frame_index = int(val)
    update_slice(frame_index)

ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])
slider = Slider(ax_slider, 'Frame', 0, nb_frames - 1, valinit=0, valstep=1)
slider.on_changed(update)

ax_button = plt.axes([0.1, 0.06, 0.1, 0.04])
button = Button(ax_button, 'Play')

is_playing = False

def toggle_play(event):
    global is_playing
    is_playing = not is_playing
    if is_playing:
        button.label.set_text('Pause')
        play_animation(event)
    else:
        button.label.set_text('Play')

button.on_clicked(toggle_play)

plt.show()
