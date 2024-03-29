This repository introduces a new zero-short audio-visual method for compound expression recognition.

Model weights are available at [models](https://drive.google.com/drive/folders/1KMkMNKkymTVV3eJaXHU6ydvEj5UfUA0E?usp=sharing). You should download them and place them in ``src/weights``. You will also need weights for the RetinaFace detection model. Please refer to the original [repository](https://github.com/hhj1897/face_detection).

To predict compound expression by a video, you should run the command:

```shell script
python run.py --path_video <your path to a video file> --path_save <your path to save results>
```

Example of predictions obtained by static visual (VS), dynamic visual (VD), audio (A), and audio-visual (AV) models:

<div style="display:flex; flex-direction: column;">
    <img src="https://github.com/C-EXPR-DB/AVCER/blob/main/static/img/Predictions.png" alt="predictions" style="width: 100%;">
</div>
