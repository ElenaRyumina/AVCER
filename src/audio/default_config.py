import sys

sys.path.append("../src")

from models.audio_expr_models import *

c_config: dict = {
    "ABAW_WAV_ROOT": "",
    "ABAW_FILTERED_WAV_ROOT": "",
    "ABAW_VIDEO_ROOT": "",
    "ABAW_LABELS_ROOT": "",
    "ABAW_FEATURES_ROOT": "",
    "MELD_WAV_ROOT": "",
    "MELD_FILTERED_WAV_ROOT": "",
    "MELD_LABELS_ROOT": "",
    "MELD_VAD_ROOT": "",
    "C_WAV_ROOT": "",
    "C_FILTERED_WAV_ROOT": "",
    "C_VIDEO_ROOT": "",
    "C_LABELS_ROOT": "",
    "C_FEATURES_ROOT": "",
    ###
    "LOGS_ROOT": "",
    "MODEL_PARAMS": {
        "model_cls": ExprModelV1,
        "args": {
            "model_name": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        },
    },
    "FILTERED": False,
    "AUGMENTATION": False,
    "NUM_EPOCHS": 100,
    "BATCH_SIZE": 24,
}
