# Import data
import sys
import time
import numpy as np
from skimage import io
import plotly.graph_objects as go
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from plotly.subplots import make_subplots
from plotly.graph_objs import Figure
from plotly.offline import plot

class MRIViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MRI Viewer')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.vol = io.imread("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif")
        self.volume = self.vol.T
        self.r, self.c = self.volume[0].shape

        self.fig = self.create_figure()
        html = plot(self.fig, output_type='div', include_plotlyjs='cdn')
        
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        web_view = QWebEngineView()
        web_view.setHtml(html)
        layout.addWidget(web_view)

    def create_figure(self):
        nb_frames = 68

        fig = go.Figure(frames=[go.Frame(data=go.Surface(
            z=(6.7 - k * 0.1) * np.ones((self.r, self.c)),
            surfacecolor=np.flipud(self.volume[67 - k]),
            cmin=0, cmax=200
            ),
            name=str(k)
            )
            for k in range(nb_frames)])

        fig.add_trace(go.Surface(
            z=6.7 * np.ones((self.r, self.c)),
            surfacecolor=np.flipud(self.volume[67]),
            colorscale='Gray',
            cmin=0, cmax=200,
            colorbar=dict(thickness=20, ticklen=4)
            ))

        sliders = [
            {
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
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]

        fig.update_layout(
            title='Slices in volumetric data @ MRI-Uganda',
            width=600,
            height=600,
            scene=dict(
                zaxis=dict(range=[-0.1, 6.8], autorange=False),
                aspectratio=dict(x=1, y=1, z=1),
            ),
            updatemenus = [
                {
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
                }
            ],
            sliders=sliders
        )

        return fig

    def frame_args(self, duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MRIViewer()
    viewer.show()
    sys.exit(app.exec_())
