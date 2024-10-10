import sys
import threading
import numpy as np
from skimage import io
import plotly.graph_objects as go
import tkinter as tk
from tkinter import ttk
from tkhtmlview import HTMLLabel
from plotly.offline import plot

class MRIViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('MRI Viewer')
        self.geometry('800x600')

        self.html_label = HTMLLabel(self, html="<h1>Loading...</h1>")
        self.html_label.pack(fill="both", expand=True)

        # Load data and create figure in a separate thread
        threading.Thread(target=self.load_data_and_create_figure, daemon=True).start()

    def load_data_and_create_figure(self):
        self.volume = io.imread("attention-mri.tif").T
        self.r, self.c = self.volume[0].shape

        self.fig = self.create_figure()
        html = plot(self.fig, output_type='div', include_plotlyjs='cdn')
        
        # Update the UI in the main thread
        self.after(0, self.update_html, html)

    def update_html(self, html):
        self.html_label.set_html(html)

    def create_figure(self):
        nb_frames = 68

        frames = [go.Frame(data=go.Surface(
            z=(6.7 - k * 0.1) * np.ones((self.r, self.c)),
            surfacecolor=np.flipud(self.volume[67 - k]),
            cmin=0, cmax=200
            ),
            name=str(k)
            )
            for k in range(nb_frames)]

        fig = go.Figure(frames=frames)

        fig.add_trace(go.Surface(
            z=6.7 * np.ones((self.r, self.c)),
            surfacecolor=np.flipud(self.volume[67]),
            colorscale='Gray',
            cmin=0, cmax=200,
            colorbar=dict(thickness=20, ticklen=4)
            ))

        sliders = [{
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], self.frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(frames)
            ],
        }]

        fig.update_layout(
            title='Slices in volumetric data @ MRI-Uganda',
            width=600,
            height=600,
            scene=dict(
                zaxis=dict(range=[-0.1, 6.8], autorange=False),
                aspectratio=dict(x=1, y=1, z=1),
            ),
            updatemenus = [{
                "buttons": [
                    {
                        "args": [None, self.frame_args(50)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], self.frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }],
            sliders=sliders
        )

        return fig

    @staticmethod
    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

if __name__ == '__main__':
    viewer = MRIViewer()
    viewer.mainloop()