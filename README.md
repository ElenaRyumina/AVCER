# Zero-Shot Audio-Visual Compound Expression Recognition Method based on Emotion Probability Fusion

The official repository for "Zero-Shot Audio-Visual Compound Expression Recognition Method based on Emotion Probability Fusion", as a part of [CVPRW 2024](https://affective-behavior-analysis-in-the-wild.github.io/6th/) (Accepted)

## Abstract

A Compound Expression Recognition (CER) as a part of affective computing is a novel task in intelligent human-computer interaction and multimodal user interfaces. We propose a novel audio-visual method for CER. Our method relies on emotion recognition models that fuse modalities at the emotion probability level, while decisions regarding the prediction of compound expressions are based on the pair-wise sum of weighted emotion probability distributions. Notably, our method does not use any training data specific to the target task. Thus, the problem is a zero-shot classification task. The method is evaluated in multi-corpus training and cross-corpus validation setups. We achieved F1-score values equal to 32.15% and 25.56% for the AffWild2 and C-EXPR-DB test subsets without training on target corpus and target task, respectively. Therefore, our method is on par with methods developed training target corpus or target task.

<hr />

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

## Acknowledgments

Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page.

## Website License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
