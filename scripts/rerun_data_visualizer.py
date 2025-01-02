import rerun as rr
import numpy as np

import rerun.blueprint as rrb
from rerun.blueprint import Blueprint, BlueprintPanel, Horizontal, Vertical, SelectionPanel, TimeSeriesView, Tabs, Spatial2DView, TimePanel

def visualize_data(data):
    # Initialize Rerun
    rr.init("Bartender Data Visualizer", spawn=True)

    def blueprint_raw():
        blueprint = Blueprint(
            Vertical(
                Horizontal(
                    Spatial2DView(origin='camera_thunder'),
                    Spatial2DView(origin='camera_front'),
                    Spatial2DView(origin='camera_lightning')
                ),
                Horizontal(
                    TimeSeriesView(origin="thunder_q", axis_y=rrb.ScalarAxis(range=(-3.5, 3.5))),
                    TimeSeriesView(origin="lightning_q", axis_y=rrb.ScalarAxis(range=(-3.5, 3.5)))
                )
            ),
            BlueprintPanel(expanded=False),
            SelectionPanel(expanded=False),
            TimePanel(expanded=False),
            auto_space_views=False
        )
        return blueprint

    rr.send_blueprint(blueprint_raw())

    # Prepare the arrays
    lightning_angles = np.zeros((len(data.keys()), 6))
    thunder_angles = np.zeros((len(data.keys()), 6))

    for i, key in enumerate(data.keys()):
        lightning_angles[i] = data[key]['lightning_angle']
        thunder_angles[i] = data[key]['thunder_angle']

    for step in range(lightning_angles.shape[0]):
        time = step/10.0
        rr.set_time_seconds("time", time)

        camera_front_image = data[step]['camera_both_front']
        rr.log("camera_front", rr.Image(camera_front_image))

        camera_lightning_image = data[step]['camera_lightning_wrist']
        rr.log("camera_lightning", rr.Image(camera_lightning_image))

        camera_thunder_image = data[step]['camera_thunder_wrist']
        rr.log("camera_thunder", rr.Image(camera_thunder_image))
    
        for i in range(6):
            rr.log(f"lightning_q/{i}", rr.SeriesLine(name=f"Lightning {i}", width=2), static=True)
            rr.log(f"lightning_q/{i}", rr.Scalar(lightning_angles[step][i]))

            rr.log(f"thunder_q/{i}", rr.SeriesLine(name=f"Thunder {i}", width=2), static=True)
            rr.log(f"thunder_q/{i}", rr.Scalar(thunder_angles[step][i]))


if __name__ == "__main__":
    path_to_data = "raw_data/transforms_396.npy"
    data = np.load(path_to_data, allow_pickle=True).item()
    visualize_data(data)