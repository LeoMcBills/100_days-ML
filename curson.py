import sys
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MRIViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('MRI Viewer')
        self.geometry('800x600')

        self.loading_label = ttk.Label(self, text="Loading...")
        self.loading_label.pack(fill="both", expand=True)

        # Load data and create figure in the main thread
        self.load_data_and_create_figure()

    def load_data_and_create_figure(self):
        self.volume = io.imread("attention-mri.tif")
        self.slices, self.height, self.width = self.volume.shape
        self.slices = 67  # Set the number of slices to 67

        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.create_animation()
        
        # Update the UI
        self.update_ui()

    def update_ui(self):
        self.loading_label.destroy()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add play button
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X)

        self.play_button = ttk.Button(control_frame, text="Play", command=self.play_animation)
        self.play_button.pack(side=tk.LEFT)

    def create_animation(self):
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D MRI Scan Animation')

        # Initial empty plot
        self.plot = [self.ax.plot_surface(np.array([[0]]), np.array([[0]]), np.array([[0]]), 
                                          rstride=1, cstride=1, cmap='viridis', 
                                          edgecolor='none', alpha=0.8)]

        self.anim = FuncAnimation(self.fig, self.update, frames=self.slices, 
                                  interval=100, blit=False, repeat=True)
        self.anim.event_source = None  # Stop animation initially
        
        self.fig.tight_layout()

    def update(self, frame):
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'3D MRI Scan - Slice {frame+1}/{self.slices}')

        # Create meshgrid for the current slice
        x, y = np.meshgrid(range(self.width), range(self.height))
        z = frame * np.ones_like(x)

        # Plot the current slice
        self.ax.plot_surface(x, y, z, rstride=1, cstride=1, 
                             facecolors=plt.cm.viridis(self.volume[frame]/self.volume.max()), 
                             edgecolor='none', alpha=0.8)

        # Set consistent viewing angle and limits
        self.ax.view_init(elev=20, azim=45)
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_zlim(0, self.slices)

        self.canvas.draw()

    def play_animation(self):
        if self.anim.event_source is None:
            self.anim = FuncAnimation(self.fig, self.update, frames=self.slices, 
                                      interval=100, blit=False, repeat=True)
        else:
            self.anim.event_source.start()

if __name__ == '__main__':
    viewer = MRIViewer()
    viewer.mainloop()