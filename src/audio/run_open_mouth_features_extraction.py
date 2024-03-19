"""
This is the script for extracting the mouth open features from video.
They are used for audio modality to detect speech segments

Author: *** # TODO
"""

import os
import glob

import cv2
import pandas as pd
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import (
    NormalizedLandmark,
    NormalizedLandmarkList,
)

from tqdm import tqdm


def calculate_triangle_area(
    landmark1: NormalizedLandmark,
    landmark2: NormalizedLandmark,
    landmark3: NormalizedLandmark,
) -> float:
    """Calculates the area of a triangle using the three dimensional coordinates of the landmarks

    Args:
        landmark1 (NormalizedLandmark): Landmark 1
        landmark2 (NormalizedLandmark): Landmark 2
        landmark3 (NormalizedLandmark): Landmark 3
    Returns:
        float: Area of a triangle
    """
    a = (landmark1.x - landmark2.x) * (landmark1.y + landmark2.y)
    b = (landmark2.x - landmark3.x) * (landmark2.y + landmark3.y)
    c = (landmark3.x - landmark1.x) * (landmark3.y + landmark1.y)
    return 0.5 * abs(a + b + c)


def calculate_surface_area(landmarks: NormalizedLandmarkList) -> float:
    """Calculates the surface area of mouth using the three dimensional coordinates of the landmarks

    Args:
        landmarks (NormalizedLandmarkList): list of landmarks

    Returns:
        float: surface area of the mouth
    """

    landmarks = landmarks.landmark

    OUTER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 78]

    # Landmark indices for the inner lips.
    INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78]

    # Calculate the surface area using the three dimensional coordinates of the landmarks.
    surface_area = 0
    for i in range(len(OUTER_LIPS) - 1):
        surface_area += calculate_triangle_area(
            landmarks[OUTER_LIPS[i]],
            landmarks[INNER_LIPS[i]],
            landmarks[OUTER_LIPS[i + 1]],
        )
        surface_area += calculate_triangle_area(
            landmarks[INNER_LIPS[i + 1]],
            landmarks[INNER_LIPS[i]],
            landmarks[OUTER_LIPS[i + 1]],
        )

    return surface_area


def extract_surface_area(
    path_to_images: str, path_to_landmarks: str, speaker_id: int = 0
) -> None:
    """Extract frame-by-frame mouth open features from images

    Args:
        path_to_images (str): Path to list of folder with images
        path_to_landmarks (str): Output path
        speaker_id (int, optional): Target speaker ID = Folder of speaker. Defaults to 0.
    """

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for folder in tqdm(os.listdir(path_to_images)):
            pd_lips = []
            for idx, file in enumerate(
                sorted(
                    glob.glob(
                        os.path.join(
                            path_to_images, folder, str(speaker_id).zfill(2), "*.jpg"
                        )
                    )
                )
            ):
                image = cv2.imread(file)
                file_name = os.path.basename(file).split(".")[0]
                # Convert the BGR image to RGB before processing.
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Print and draw face mesh landmarks on the image.
                if not results.multi_face_landmarks:
                    continue
                face_landmarks = results.multi_face_landmarks[0]

                surface_area = calculate_surface_area(face_landmarks)
                pd_lips.append([file_name, surface_area])
            pd_lips = pd.DataFrame(pd_lips, columns=["frame", "surface_area_mouth"])

            mask = pd_lips[
                pd_lips["surface_area_mouth"].rolling(window=30).mean()
                > pd_lips["surface_area_mouth"].mean()
            ]
            pd_lips["mouth_open"] = 0
            pd_lips.loc[mask.index, "mouth_open"] = 1
            save_path = os.path.join(path_to_landmarks)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pd_lips.to_csv(os.path.join(save_path, folder + ".csv"), index=True)
            print("Done with folder: {}".format(folder))


if __name__ == "__main__":
    path_to_images = "/"  # TODO
    path_to_landmarks = "/"  # TODO
    extract_surface_area(path_to_images, path_to_landmarks)
