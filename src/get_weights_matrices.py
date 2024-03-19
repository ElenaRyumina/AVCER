import numpy as np
from visualization.visualize import plot_weights_matrix

if __name__ == "__main__":
    weights_1 = np.asarray(
        [
            [0.42633145, 0.57366855],
            [0.57803352, 0.42196648],
            [0.01878466, 0.98121534],
            [0.86451425, 0.13548575],
            [0.16464752, 0.83535248],
            [0.03786653, 0.96213347],
            [0.81048546, 0.18951454],
            [0.36499999999999994, 0.22999999999999998],
        ]
    )

    plot_weights_matrix(
        weights_1.T,
        labels=[["VS", "VD"], ["Ne", "An", "Di", "Fe", "Ha", "Sa", "Su", "Mo"]],
        title="Weights for video modality fusion",
        save_path="weights1_video.pdf",
        figsize=(6, 4),
        colorbar=True,
        x_labels=False,
    )

    weights_2 = np.asarray(
        [
            [0.85806901, 0.11491265, 0.02701833],
            [0.2579578, 0.46222294, 0.27981925],
            [0.2579578, 0.62411413, 0.17148297],
            [0.72010502, 0.16716238, 0.1127326],
            [0.62082661, 0.31962795, 0.05954545],
            [0.06281922, 0.16603196, 0.77114883],
            [0.70875895, 0.24433032, 0.04691073],
            [0.060000000000000005, 0.21000000000000002, 0.01],
        ]
    )

    plot_weights_matrix(
        weights_2.T,
        labels=[["VS", "VD", "A"], ["Ne", "An", "Di", "Fe", "Ha", "Sa", "Su", "Mo"]],
        title="Weights for audio (7cl) and video modality fusion",
        save_path="weights1_av_7.pdf",
        figsize=(6, 6),
        colorbar=True,
        x_labels=False,
    )

    weights_3 = np.asarray(
        [
            [0.89900098, 0.01223291, 0.08876611],
            [0.10362151, 0.21364307, 0.68273542],
            [0.08577635, 0.66688002, 0.24734363],
            [0.04428126, 0.93791526, 0.01780348],
            [0.89679865, 0.0398964, 0.06330495],
            [0.02656456, 0.48670648, 0.48672896],
            [0.63040305, 0.22089692, 0.14870002],
            [0.16000000000000003, 0.36000000000000004, 0.01],
        ]
    )

    plot_weights_matrix(
        weights_3.T,
        labels=[["VS", "VD", "A"], ["Ne", "An", "Di", "Fe", "Ha", "Sa", "Su", "Mo"]],
        title="Weights for audio (8cl) and video modality fusion",
        save_path="weights1_av_8.pdf",
        figsize=(6, 6),
        colorbar=True,
        x_labels=True,
    )
