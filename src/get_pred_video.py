import pandas as pd
import os
import numpy as np
import pickle
from tqdm import tqdm
from visualization.visualize import plot_conf_matrix
from sklearn.metrics import confusion_matrix
from data.utils import (
    metrics,
    softmax,
    get_compound_expression,
    save_txt,
    get_weights_prob_model,
    get_weights_v_model,
)


def get_metrics(
    trues, predictions, weights_1, weights_2, corpus, modality, weight_type
):

    name_emo = [
        "Neutral",
        "Anger",
        "Disgust",
        "Fear",
        "Happiness",
        "Sadness",
        "Surprise",
        "Other",
    ]

    num_predictions = len(predictions)
    final_test_prediction = predictions[0] * weights_1[0] * weights_2[0]

    for i in range(1, num_predictions):
        final_test_prediction += predictions[i] * weights_1[i] * weights_2[i]

    final_test_prediction = np.argmax(final_test_prediction, axis=-1).astype("int32")

    uar_v, acc_v, f1_v, precision_v, mean_v = metrics(trues, final_test_prediction)
    cm_v = confusion_matrix(
        trues, final_test_prediction, labels=[l for l in range(len(name_emo[:-1]))]
    )
    save_path = "src/pred_results/confusion_matrices/"
    os.makedirs(save_path, exist_ok=True)

    plot_conf_matrix(
        cm=cm_v,
        labels=name_emo[:-1],
        title="Video fusion. {0}. UAR = {1:.2f}%".format(corpus, uar_v * 100),
        save_path=os.path.join(save_path, f"{corpus}_{modality}_sd_{weight_type}.pdf"),
    )

    s_pred = np.argmax(predictions[0], axis=-1).astype("int32")
    uar_s, acc_s, f1_s, precision_s, mean_s = metrics(trues, s_pred)
    cm_stat = confusion_matrix(
        trues, s_pred, labels=[l for l in range(len(name_emo[:-1]))]
    )

    plot_conf_matrix(
        cm=cm_stat,
        labels=name_emo[:-1],
        title="Static video model. {0}. UAR = {1:.2f}%".format(corpus, uar_s * 100),
        save_path=os.path.join(
            save_path, f"{corpus}_{modality}_static_{weight_type}.pdf"
        ),
    )

    d_pred = np.argmax(predictions[1], axis=-1).astype("int32")
    uar_d, acc_d, f1_d, precision_d, mean_d = metrics(trues, d_pred)
    cm_dyn = confusion_matrix(
        trues, d_pred, labels=[l for l in range(len(name_emo[:-1]))]
    )

    plot_conf_matrix(
        cm=cm_dyn,
        labels=name_emo[:-1],
        title="Dynamic video model. {0}. UAR = {1:.2f}%".format(corpus, uar_d * 100),
        save_path=os.path.join(
            save_path, f"{corpus}_{modality}_dynamic_{weight_type}.pdf"
        ),
    )

    dict_metrics = {
        "uar_s": uar_s,
        "acc_s": acc_s,
        "f1_s": f1_s,
        "precision_s": precision_s,
        "mean_s": mean_s,
        "uar_d": uar_d,
        "acc_d": acc_d,
        "f1_d": f1_d,
        "precision_d": precision_d,
        "mean_d": mean_d,
        "uar_v": uar_v,
        "acc_v": acc_v,
        "f1_v": f1_v,
        "precision_v": precision_v,
        "mean_v": mean_v,
        "weights_1_v": weights_1,
        "weights_2_v": weights_2,
    }

    print(dict_metrics)

    save_path = "src/pred_results/metrics_dicts/"
    os.makedirs(save_path, exist_ok=True)

    filename = os.path.join(
        save_path, f"{corpus}_metrics_dict_{modality}_{weight_type}.pickle"
    )
    pickle.dump(dict_metrics, open(filename, "wb"))


def get_abaw_pred(path_ann, path_pred, ann_files):
    name_emo = [
        "Neutral",
        "Anger",
        "Disgust",
        "Fear",
        "Happiness",
        "Sadness",
        "Surprise",
        "Other",
    ]
    trues = []
    preds_stat = []
    preds_dyn = []

    for curr_ann_file in tqdm(ann_files):
        curr_df = pd.read_csv(os.path.join(path_ann, curr_ann_file))
        curr_pred_stat_df = pd.read_csv(
            os.path.join(path_pred, "static__" + curr_ann_file[:-4]) + ".csv"
        )[name_emo[:-1]]
        curr_pred_dyn_df = pd.read_csv(
            os.path.join(path_pred, "dynamic__" + curr_ann_file[:-4]) + ".csv"
        )[name_emo[:-1]]

        need_index = curr_df.loc[~curr_df["Neutral"].isin([-1, 7])].index
        true = curr_df[curr_df.index.isin(need_index)].Neutral.values.tolist()
        curr_pred_stat_df = curr_pred_stat_df[
            curr_pred_stat_df.index.isin(need_index)
        ].values.tolist()
        curr_pred_dyn_df = curr_pred_dyn_df[
            curr_pred_dyn_df.index.isin(need_index)
        ].values
        curr_pred_dyn_df = softmax(curr_pred_dyn_df).tolist()

        if len(true) > len(curr_pred_dyn_df):
            last_pred_stat = curr_pred_stat_df[-1]
            last_pred_dyn = curr_pred_dyn_df[-1]
            curr_pred_stat_df += [last_pred_stat] * (len(true) - len(curr_pred_dyn_df))
            curr_pred_dyn_df += [last_pred_dyn] * (len(true) - len(curr_pred_dyn_df))

        preds_stat.extend(curr_pred_stat_df)
        preds_dyn.extend(curr_pred_dyn_df)
        trues.extend(true)

    return preds_stat, preds_dyn, trues


def get_afew_pred(path_pred):
    df_emotion = pd.read_csv("D:/work/ABAW2024/AFEW_data.csv")
    name_videos = [i[:-4] for i in df_emotion.name_video]
    emotion = df_emotion.emotion.values

    # video model prediction order
    dict_emo = {
        "Angry": 6,
        "Disgust": 5,
        "Fear": 4,
        "Happy": 1,
        "Neutral": 0,
        "Sad": 2,
        "Surprise": 3,
    }

    trues = []
    preds_dyn = []
    preds_stat = []

    for curr_video, curr_emotion in tqdm(zip(name_videos, emotion)):
        curr_pred_stat_df = pd.read_csv(
            os.path.join(path_pred, "static__" + curr_video) + ".csv"
        ).values
        curr_pred_dyn_df = pd.read_csv(
            os.path.join(path_pred, "dynamic__" + curr_video) + ".csv"
        ).values
        curr_pred_dyn_df = softmax(curr_pred_dyn_df)

        true = dict_emo[curr_emotion]
        pred_stat = np.mean(curr_pred_stat_df, axis=0)
        pred_dyn = np.mean(curr_pred_dyn_df, axis=0)

        trues.append(true)
        preds_stat.append(pred_stat)
        preds_dyn.append(pred_dyn)

    return preds_stat, preds_dyn, trues


def get_c_expr_db_pred(
    prediction_file_format,
    path_pred,
    name_videos,
    weights_1,
    weights_2,
    modality,
    weight_type,
    ce_weights_type,
    ce_mask,
):

    name_emo = [
        "Neutral",
        "Anger",
        "Disgust",
        "Fear",
        "Happiness",
        "Sadness",
        "Surprise",
        "Other",
    ]

    df_prediction_file_format = pd.read_csv(prediction_file_format)

    preds_dyn = []
    preds_stat = []
    image_locations = []
    for curr_video in tqdm(name_videos):
        curr_pred_stat_df = pd.read_csv(
            os.path.join(path_pred, "static__" + curr_video) + ".csv"
        )
        curr_pred_stat_df["image_location"] = [
            f"{curr_video}/{str(f+1).zfill(5)}.jpg" for f in curr_pred_stat_df.index
        ]
        curr_pred_dyn_df = pd.read_csv(
            os.path.join(path_pred, "dynamic__" + curr_video) + ".csv"
        )
        curr_pred_dyn_df["image_location"] = [
            f"{curr_video}/{str(f+1).zfill(5)}.jpg" for f in curr_pred_dyn_df.index
        ]
        need_image_location = list(
            set(curr_pred_stat_df.image_location)
            & set(curr_pred_dyn_df.image_location)
            & set(df_prediction_file_format.image_location)
        )
        need_image_location = sorted(need_image_location)
        curr_pred_stat_df = curr_pred_stat_df[
            curr_pred_stat_df.image_location.isin(need_image_location)
        ][name_emo[:-1]].values.tolist()
        curr_pred_dyn_df = curr_pred_dyn_df[
            curr_pred_dyn_df.image_location.isin(need_image_location)
        ][name_emo[:-1]].values
        curr_pred_dyn_df = softmax(curr_pred_dyn_df).tolist()
        preds_stat.extend(curr_pred_stat_df)
        preds_dyn.extend(curr_pred_dyn_df)
        image_locations.extend(need_image_location)

    predictions = [preds_stat, preds_dyn]
    num_predictions = len(predictions)
    final_predictions = predictions[0] * weights_1[0] * weights_2[0]
    for i in range(1, num_predictions):
        final_predictions += predictions[i] * weights_1[i] * weights_2[i]

    # audio model prediction order
    com_emo = {
        "Fearfully Surprised": [3, 6],
        "Happily Surprised": [4, 6],
        "Sadly Surprised": [5, 6],
        "Disgustedly Surprised": [2, 6],
        "Angrily Surprised": [1, 6],
        "Sadly Fearful": [3, 5],
        "Sadly Angry": [1, 5],
    }

    dict_weights = {
        1: 5,
        2: 6,
        3: 5,
        4: 6,
        5: 4,
        6: 2,
    }

    v_prob = get_compound_expression(
        final_predictions, com_emo, dict_weights, ce_weights_type, ce_mask
    )
    s_prob = get_compound_expression(
        predictions[0], com_emo, dict_weights, ce_weights_type, ce_mask
    )
    d_prob = get_compound_expression(
        predictions[1], com_emo, dict_weights, ce_weights_type, ce_mask
    )

    save_path = "src/pred_results/DF_C_EXPR_DB/"
    os.makedirs(save_path, exist_ok=True)

    v_pred = np.argmax(v_prob[:, :7], axis=1)
    s_pred = np.argmax(s_prob[:, :7], axis=1)
    d_pred = np.argmax(d_prob[:, :7], axis=1)

    column_names = [
        "image_location",
        "Fearfully_Surprised",
        "Happily_Surprised",
        "Sadly_Surprised",
        "Disgustedly_Surprised",
        "Angrily_Surprised",
        "Sadly_Fearful",
        "Sadly_Angry",
    ]

    save_txt(
        column_names,
        image_locations,
        v_pred,
        os.path.join(
            save_path,
            f"C_EXPR_DB_{modality}_sd_{weight_type}_{ce_weights_type}_{ce_mask}.txt",
        ),
    )
    save_txt(
        column_names,
        image_locations,
        s_pred,
        os.path.join(
            save_path,
            f"C_EXPR_DB_{modality}_static_{weight_type}_{ce_weights_type}_{ce_mask_types}.txt",
        ),
    )
    save_txt(
        column_names,
        image_locations,
        d_pred,
        os.path.join(
            save_path,
            f"C_EXPR_DB_{modality}_dynamic_{weight_type}_{ce_weights_type}_{ce_mask_types}.txt",
        ),
    )


if __name__ == "__main__":

    np.random.seed(seed=42)

    path_ann = "C:/Work/Datasets/ABAW/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set/"
    path_pred_ABAW = "src/pred_results/ABAW/video/"
    path_pred_AFEW = "src/pred_results/AFEW/video/"
    path_pred_C = "src/pred_results/C-EXPR-DB/video/"
    weight_types = ["single", "double"]
    ce_weights_types = [False, True]
    ce_mask_types = [True, False]
    modalities = ["video"]

    type_of_fusion = "class_model"
    num_weights = 10000
    # num_weights=10
    num_classes = 7

    weights = np.arange(0.01, 0.5, 0.005)
    # weights = np.arange(0.01, 0.1, 0.01)

    ann_files = os.listdir(path_ann)

    prediction_file_format = "C:/Work/Datasets/ABAW/prediction_files_format/CVPR_6th_ABAW_CE_test_set_sample.txt"

    df_name_videos = pd.read_csv("D:/work/ABAW2024/C-EXPR-DB_data.csv")
    name_videos = [i[:-4] for i in df_name_videos.name_video]

    preds_stat_abaw, preds_dyn_abaw, trues_abaw = get_abaw_pred(
        path_ann, path_pred_ABAW, ann_files
    )
    print("Examples ABAW: ", len(preds_stat_abaw), len(preds_dyn_abaw), len(trues_abaw))

    preds_stat_afew, preds_dyn_afew, trues_afew = get_afew_pred(path_pred_AFEW)
    print("Examples AFEW: ", len(preds_stat_afew), len(preds_dyn_afew), len(trues_afew))

    best_weights_1 = get_weights_prob_model(
        trues_abaw, [preds_stat_abaw, preds_dyn_abaw], num_weights, num_classes
    )

    preds_stat_abaw_w = best_weights_1[0] * np.array(preds_stat_abaw)
    preds_dyn_abaw_w = best_weights_1[1] * np.array(preds_dyn_abaw)

    best_weights_2 = get_weights_v_model(
        weights, trues_abaw, [preds_stat_abaw_w, preds_dyn_abaw_w]
    )

    for modelity in modalities:
        for weight_type in weight_types:

            if weight_type == "double":
                best_weights_double = best_weights_2

            else:
                best_weights_double = [1, 1]

            print("Report ABAW")
            get_metrics(
                trues_abaw,
                [preds_stat_abaw, preds_dyn_abaw],
                best_weights_1,
                best_weights_double,
                "ABAW",
                modelity,
                weight_type,
            )

            print("Report AFEW")
            get_metrics(
                trues_afew,
                [preds_stat_afew, preds_dyn_afew],
                best_weights_1,
                best_weights_double,
                "AFEW",
                modelity,
                weight_type,
            )

            for ce_weights_type in ce_weights_types:

                for ce_mask in ce_mask_types:

                    get_c_expr_db_pred(
                        prediction_file_format,
                        path_pred_C,
                        name_videos,
                        best_weights_1,
                        best_weights_double,
                        modelity,
                        weight_type,
                        ce_weights_type,
                        ce_mask,
                    )
