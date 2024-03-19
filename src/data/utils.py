from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch
import subprocess
import torchaudio
import os
import numpy as np
import cv2
from visualization.visualize import show_cam_on_image
from sklearn.metrics import (
    recall_score,
    f1_score,
    precision_score,
    classification_report,
)


def pth_processing(fp):
    class PreprocessInput(torch.nn.Module):
        def init(self):
            super(PreprocessInput, self).init()

        def forward(self, x):
            x = x.to(torch.float32)
            x = torch.flip(x, dims=(0,))
            x[0, :, :] -= 91.4953
            x[1, :, :] -= 103.8827
            x[2, :, :] -= 131.0912
            return x

    def get_img_torch(img, target_size=(224, 224)):
        transform = transforms.Compose([transforms.PILToTensor(), PreprocessInput()])
        img = img.resize(target_size, Image.Resampling.NEAREST)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        return img

    return get_img_torch(fp)


def convert_mp4_to_mp3(path, sampling_rate=16000):

    path_save = path[:-3] + "wav"
    if not os.path.exists(path_save):
        ff_audio = "ffmpeg -i {} -vn -acodec pcm_s16le -ar 44100 -ac 2 {}".format(
            path, path_save
        )
        subprocess.call(ff_audio, shell=True)
    wav, sr = torchaudio.load(path_save)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sampling_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        wav = transform(wav)
        sr = sampling_rate

    assert sr == sampling_rate
    return wav.squeeze(0)


def pad_wav(wav, max_length):
    current_length = len(wav)
    if current_length < max_length:
        repetitions = (max_length + current_length - 1) // current_length
        wav = torch.cat([wav] * repetitions, dim=0)[:max_length]
    elif current_length > max_length:
        wav = wav[:max_length]

    return wav


def pad_wav_zeros(wav, max_length, mode="constant"):

    if mode == "mean":
        wav = torch.nn.functional.pad(
            wav,
            (0, max(0, max_length - wav.shape[0])),
            mode="constant",
            value=torch.mean(wav),
        )

    else:
        wav = torch.nn.functional.pad(
            wav, (0, max(0, max_length - wav.shape[0])), mode=mode
        )

    return wav


def get_heatmaps(
    gradients, activations, name_layer, face_image, use_rgb=True, image_weight=0.6
):
    gradient = gradients[name_layer]
    activation = activations[name_layer]
    pooled_gradients = torch.mean(gradient[0], dim=[0, 2, 3])
    for i in range(activation.size()[1]):
        activation[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activation, dim=1).squeeze().cpu()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    heatmap = torch.unsqueeze(heatmap, -1)
    heatmap = cv2.resize(heatmap.detach().numpy(), (224, 224))
    cur_face_hm = cv2.resize(face_image, (224, 224))
    cur_face_hm = np.float32(cur_face_hm) / 255

    heatmap = show_cam_on_image(
        cur_face_hm, heatmap, use_rgb=use_rgb, image_weight=image_weight
    )

    return heatmap


def get_metrics_for_fusion(true, pred):
    dict_ = classification_report(true, pred, output_dict=True)
    metrics = np.zeros(3)
    for cl in range(1, 7):
        for idx, metric in enumerate(["precision", "f1-score", "recall"]):
            metrics[idx] += dict_[str(cl)][metric]
    precision, f1, uar = metrics / 6
    return precision, f1, uar


def softmax(matrix):
    exp_matrix = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
    return exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)


def metrics(true, pred):
    uar = recall_score(true, pred, average="macro")
    acc = recall_score(true, pred, average="micro")
    f1 = f1_score(true, pred, average="macro")
    precision = precision_score(true, pred, average="macro")
    return uar, acc, f1, precision, np.mean((uar, acc, f1, precision))


def get_weights_prob_model(ground_truth, predictions, num_weights, num_classes):
    num_predictions = len(predictions)

    weights = np.zeros(shape=(num_weights, num_predictions, num_classes))
    for i in range(num_weights):
        weights[i] = np.random.dirichlet(
            alpha=np.ones((num_predictions,)), size=num_classes
        ).T

    best = 0
    best_weights = None

    for weight_idx in tqdm(range(num_weights)):
        final_prediction = predictions[0] * weights[weight_idx, 0]
        for i in range(1, num_predictions):
            final_prediction += predictions[i] * weights[weight_idx, i]
        final_prediction = np.argmax(final_prediction, axis=-1)
        _, _, metric = get_metrics_for_fusion(ground_truth, final_prediction)
        if metric > best:
            best = metric
            best_weights = weights[weight_idx]

    print("final best metric:%f" % (best))
    print("weights:", best_weights)

    return best_weights


def get_weights_v_model(weights, ground_truth, predictions):

    pred_1 = np.array(predictions[0])
    pred_2 = np.array(predictions[1])

    best_weights = [0, 0]
    best_acc = 0

    for w_s in tqdm(weights):
        for w_d in weights:
            y_pred = np.argmax(w_s * pred_1 + w_d * pred_2, axis=1)
            _, _, acc = get_metrics_for_fusion(ground_truth, y_pred)
            if acc > best_acc:
                best_acc = acc
                best_weights = [w_s, w_d]

    print("final best metric:%f" % (best_acc))
    print("weights:", best_weights)

    return best_weights


def get_weights_av_model(weights, ground_truth, predictions):

    pred_1 = np.array(predictions[0])
    pred_2 = np.array(predictions[1])
    pred_3 = np.array(predictions[2])

    best_weights = [0, 0, 0]
    best_acc = 0

    for w_s in tqdm(weights):
        for w_d in weights:
            for w_a in weights:
                y_pred = np.argmax(w_s * pred_1 + w_d * pred_2 + w_a * pred_3, axis=1)
                _, _, acc = get_metrics_for_fusion(ground_truth, y_pred)
                if acc > best_acc:
                    best_acc = acc
                    best_weights = [w_s, w_d, w_a]

    print("final best metric:%f" % (best_acc))
    print("weights:", best_weights)

    return best_weights


def save_txt(column_names, file_names, labels, save_name):
    data_lines = [",".join(column_names)]
    for file_name, label in zip(file_names, labels):
        data_lines.append(f"{file_name},{label}")

    with open(save_name, "w") as file:
        for line in data_lines:
            file.write(line + "\n")


def get_compound_expression(pred, com_emo, dict_weights, ce_weights_type, ce_mask):
    pred = np.asarray(pred)
    prob = np.zeros((len(pred), len(com_emo)))
    for idx, (k, v) in enumerate(com_emo.items()):
        idx_1 = v[0]
        idx_2 = v[1]
        if ce_weights_type:
            s_w = dict_weights[idx_1] + dict_weights[idx_2]
            w_1 = dict_weights[idx_1] / s_w
            w_2 = dict_weights[idx_2] / s_w
            # w_1 = dict_weights[idx_1]/len(com_emo)
            # w_2 = dict_weights[idx_2]/len(com_emo)
        else:
            w_1 = 1
            w_2 = 1

        if ce_mask:
            pred = np.where(pred > 1 / 7, pred, 0)
        prob[:, idx] = pred[:, idx_1] * w_1 + pred[:, idx_2] * w_2
    return prob


def get_image_location(curr_video, frame):
    frame = int(frame.split(".")[0]) + 1
    frame = str(frame).zfill(5) + ".jpg"
    return f"{curr_video}/{frame}"
