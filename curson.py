# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
from matplotlib.widgets import Slider, Button

# Load the data
vol = io.imread("attention-mri.tif")
volume = vol.T
r, c = volume[0].shape
nb_frames = 68

# Create a figure for 3D plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Define the z values for each frame
z_values = np.linspace(6.7, 6.7 - nb_frames * 0.1, nb_frames)

# Create a function to update the surface for each frame
def update_surface(k):
    ax.clear()  # Clear the previous surface
    z = z_values[k] * np.ones((r, c))  # Create a grid for Z values
    x = np.arange(c)  # X coordinates
    y = np.arange(r)  # Y coordinates
    x, y = np.meshgrid(x, y)  # Create meshgrid for surface plotting
    surface = ax.plot_surface(x, y, z, facecolors=plt.cm.gray(np.flipud(volume[67 - k] / 200)),
                               rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.set_title('Slices in volumetric data @ MRI-Uganda')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_zlim([-0.1, 6.8])
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    plt.draw()  # Update the figure
    plt.pause(0.01)  # Reduced pause time for faster animation

# Function to play the animation in a loop
def play_animation(event):
    for k in range(nb_frames):
        update_surface(k)
        if not is_playing:
            break

# Function to update frame based on slider value
def update(val):
    frame_index = int(val)
    update_surface(frame_index)

# Create slider for frame selection
ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])  # Slider position
slider = Slider(ax_slider, 'Frame', 0, nb_frames - 1, valinit=0, valstep=1)
slider.on_changed(update)

# Button to play the animation
ax_button = plt.axes([0.1, 0.06, 0.1, 0.04])  # Button position
button = Button(ax_button, 'Play')

# Flag for controlling play state
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
