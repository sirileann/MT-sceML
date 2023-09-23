"""
@Author: Siri Leane Rüegg
@Contact: sirrueeg@ethz.ch
@File: data_loader.py
"""
import glob
import json
import pathlib
import argparse
import random
import shutil
import re

import pandas as pd
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import display_pcd_plotly, convert_np_to_ply, convert_ply_to_np
from utils import upsample_3d_line, resample_pointclouds
from natsort import natsorted
from datetime import timedelta, datetime
from PIL import Image

pd.options.mode.chained_assignment = None  # default='warn'
plt.rcParams['font.family'] = 'serif'  # Change to your desired font family


def get_and_parse_data(DATA_DIR: str, surface_rot: bool = False) -> pd.DataFrame:
    """
    Function to save all paths from the point cloud data and to parse and check the metadata json files.
    Args:
        DATA_DIR: directory of data folder

    Returns: dataframe containing infos about the subjects and the list of incorrect age entries
    """

    data_dir = pathlib.Path(DATA_DIR)

    DIRS_BACKSCAN = natsorted(glob.glob(str(data_dir / "**/*_backscan.ply")))
    DIRS_METADATA = natsorted(glob.glob(str(data_dir / "**/*_metadata.json")))

    df_paths = pd.DataFrame(list(zip(DIRS_BACKSCAN, DIRS_METADATA)),
                            columns=['DIRS_BACKSCAN', 'DIRS_METADATA'])

    subjectIdx = np.empty(len(DIRS_METADATA), dtype=object)
    age_list = np.zeros(len(DIRS_METADATA), dtype=int)
    gender = np.empty((len(DIRS_METADATA)), dtype=str)
    recTime = np.empty((len(DIRS_METADATA)), dtype=datetime)
    ESL = np.zeros(len(DIRS_METADATA), dtype=object)
    ISL = np.zeros(len(DIRS_METADATA), dtype=object)
    fixPoints = np.zeros(len(DIRS_METADATA), dtype=dict)
    apexPoints = np.zeros(len(DIRS_METADATA), dtype=object)
    if surface_rot:
        surfaceRotation = np.zeros(len(DIRS_METADATA), dtype=float)

    for idx, dir in enumerate(DIRS_METADATA):
        # Opening JSON file
        f = open(dir)
        data = json.load(f)

        # some age entries are wrong, they need to be corrected
        if type(data['age']) == str:
            converted_date = datetime.strptime(data['age'], "%d.%m.%Y").year
            year_diff = converted_date - 2000
            age_list[idx] = year_diff
        else:
            age_list[idx] = data['age']

        subjectIdx[idx] = data['subjectIndex']
        gender[idx] = data['gender']
        recTime[idx] = data['recordingDatetime']
        ESL[idx] = np.array(data['esl_formetric'])
        ISL[idx] = np.array(data['isl_formetric'])
        fixPoints[idx] = data['fixPts']
        apexPoints[idx] = data['apexPts']
        if surface_rot:
            surfaceRotation[idx] = data['SurfaceRotationRms_deg']

        # Closing file
        f.close()

    # convert "w" to 0 and "m" to 1 in gender
    gender_list = np.asarray([0 if x == "w" else 1 for x in gender])

    df_paths.insert(0, 'Subject_ID', subjectIdx)
    df_paths = df_paths.assign(Gender=gender_list, Age=age_list, recTime=recTime, ESL=ESL, ISL=ISL, fixPoints=fixPoints,
                               apexPoints=apexPoints)

    if surface_rot:
        df_paths['surfaceRotation'] = surfaceRotation

    df_paths['recTime'] = pd.to_datetime(df_paths['recTime'], format='%d-%b-%Y %H:%M:%S')

    return df_paths


def create_dataset_italian(DATA_DIR: str, num_points: int = 1024) -> (pd.DataFrame, pd.DataFrame):
    """
    Function to create the dataset for the Italian data.
    Args:
        DATA_DIR: directory of data folder
        num_points: number of points for the resampled back surface scan

    Returns:
        df: dataframe containing infos about the subjects as well as the paths to the preprocessed data
        df_cleaned: dataframe without the subjects with invalid resampling

    """
    data_dir = pathlib.Path(DATA_DIR)
    all_paths_backscan = natsorted(glob.glob(str(data_dir / "**/*_backscan_relaxed_arms.ply")))
    all_paths_metadata = natsorted(data_dir.rglob("**/*_metadata.json"))  # **/OutputRegistration/*_metadata_*.json
    df_paths = pd.DataFrame(list(zip(all_paths_backscan, all_paths_metadata)),
                            columns=['DIRS_BACKSCAN', 'DIRS_METADATA'])

    print(len(all_paths_metadata), len(all_paths_backscan))

    subjectIdx = np.empty(len(df_paths.DIRS_METADATA), dtype=object)
    ESL = np.zeros(len(df_paths.DIRS_METADATA), dtype=object)
    ISL = np.zeros(len(df_paths.DIRS_METADATA), dtype=object)
    ISL_Formetric = np.zeros(len(df_paths.DIRS_METADATA), dtype=object)
    fixPoints = np.zeros(len(df_paths.DIRS_METADATA), dtype=object)
    apexPoints = np.zeros(len(df_paths.DIRS_METADATA), dtype=object)
    surfaceRotation = np.zeros(len(df_paths.DIRS_METADATA), dtype=float)

    for idx, dir in enumerate(df_paths.DIRS_METADATA):
        # Opening JSON file
        f = open(dir)

        # returns JSON object as a dictionary
        data = json.load(f)

        subjectIdx[idx] = data['subjectName']
        ESL[idx] = data['esl_formetric']
        ISL[idx] = np.array(data['isl_pcdicomapp'])
        ISL_Formetric[idx] = np.array(data['isl_formetric'])
        fixPoints[idx] = data['fixPts']
        apexPoints[idx] = data['apexPts']
        surfaceRotation[idx] = data['SurfaceRotationRms_deg']

        # Closing file
        f.close()

    df_paths.insert(0, 'Subject_ID', subjectIdx)
    df_paths = df_paths.assign(DIRS_ISL=ISL, DIRS_ISL_FORMETRIC=ISL_Formetric, DIRS_ESL=ESL, fixPoints=fixPoints,
                               apexPoints=apexPoints, surfaceRotation=surfaceRotation)

    df, df_cleaned = preprocess_and_save_data(df_paths, num_points=num_points, italian=True)

    return df, df_cleaned


def check_duplicated_files(df: pd.DataFrame, time_diff=7, show: bool = False):
    """
    Function to calculate the time difference between two consecutive measurements and to check whether the potentially
    duplicated files have a big root-mean-square error.
    Args:
        df: dataframe with all directories and information
        time_diff: allowed time difference, default = 7 [minutes]
        show: show all pairs that are similar, default = True

    Returns: dataframe with column "isDuplicate" added
    """

    df_sorted = df.sort_values(by=['recTime'])
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted['isDuplicate'] = False

    for i in range(1, len(df_sorted['recTime'])):
        diff = (df_sorted['recTime'][i] - df_sorted['recTime'][i - 1])
        if abs(diff) < timedelta(minutes=time_diff):

            cloud1 = np.asarray(o3d.io.read_point_cloud(df_sorted.DIRS_BACKSCAN[i]).points)
            cloud2 = np.asarray(o3d.io.read_point_cloud(df_sorted.DIRS_BACKSCAN[i - 1]).points)

            cloud1_mean = np.mean(cloud1, axis=0)
            cloud2_mean = np.mean(cloud2, axis=0)

            cloud1_demeaned = cloud1 - cloud1_mean
            cloud2_demeaned = cloud2 - cloud2_mean

            pcd1 = convert_np_to_ply(cloud1_demeaned)
            pcd2 = convert_np_to_ply(cloud2_demeaned)

            dists = pcd2.compute_point_cloud_distance(pcd1)
            dists = np.asarray(dists)

            rmse = np.sqrt(np.square(dists).mean())
            # print("Root-Mean-Square-Error of Demeaned Duplicate: ", rmse)

            df_sorted.isDuplicate[i] = True

            if rmse > 20:
                age_subject1 = df_sorted.Age[i]
                age_subject2 = df_sorted.Age[i - 1]

                if age_subject1 != age_subject2:
                    df_sorted.isDuplicate[i] = False

            if show:
                print(df_sorted['recTime'][i - 1], df_sorted['recTime'][i], "\nTime-Difference: ", diff)

    print("\nThere are %i duplicates in the dataset!" % df_sorted['isDuplicate'].sum())

    df_resorted = df_sorted.sort_values(by=['Subject_ID'])
    df_resorted = df_resorted.reset_index(drop=True)

    df_cleaned = df_resorted.drop(df_resorted[df_resorted["isDuplicate"]].index).reset_index(drop=True)

    return df_resorted, df_cleaned


def upsamling_sanity_check(pcd_backscan: np.ndarray) -> bool:
    """
    Function to check whether the upsampling was done correctly.
    If the number of points in the specific area is less than 50, the upsampling was not done correctly.
    Args:
        pcd_backscan: back surface scan to check

    Returns:
        True if the upsampling was not done correctly, False otherwise

    """
    # Define the specific area
    xmin, xmax = -50, 50
    ymin, ymax = -50, 50
    zmin, zmax = -100, 100

    # Extract the points within the specific area
    mask = (pcd_backscan[:, 0] >= xmin) & (pcd_backscan[:, 0] <= xmax) & \
           (pcd_backscan[:, 1] >= ymin) & (pcd_backscan[:, 1] <= ymax) & \
           (pcd_backscan[:, 2] >= zmin) & (pcd_backscan[:, 2] <= zmax)

    points_in_area = pcd_backscan[mask]

    # Calculate the number of points of the area
    num_points = len(points_in_area)

    if num_points < 50:
        return True
    else:
        return False


def preprocess_and_save_data(df: pd.DataFrame, num_points: int = 8192, italian=False) -> (
        pd.DataFrame, pd.DataFrame):
    """
    Function to preprocess the data and save it in the corresponding folders.
    Args:
        df: dataframe with all subjects that need to be preprocessed
        num_points: desired number of points to resample the back surface scans
        italian: bool to mark if it is an italian dataset

    Returns:
        df: dataframe containing infos about the subjects as well as the paths to the preprocessed data
        df_cleaned: dataframe without the subjects with invalid resampling
    """

    df['failedUpsampling'] = False
    df['linear_factor'] = np.zeros(len(df), dtype=object)
    df['scaling_factor'] = 0
    df['back_pp'] = None
    df['esl_pp'] = None
    df['isl_pp'] = None
    df['fixPoints_pp'] = None

    paths = [pathlib.Path("./preprocessed/backscan"), pathlib.Path("./preprocessed/esl"),
             pathlib.Path("./preprocessed/isl"), pathlib.Path("./preprocessed/fixPoints")]

    if italian:
        df['isl_formetric_pp'] = None
        paths.append(pathlib.Path("./preprocessed/isl_formetric"))

    for path in paths:
        if not path.exists():
            path.mkdir(parents=True)

    for i in tqdm(range(len(df))):
        pcd_backscan = convert_ply_to_np(df.DIRS_BACKSCAN[i])
        if italian:
            pcd_esl = np.array(df.DIRS_ESL[i])
            pcd_isl = np.array(df.DIRS_ISL[i])
            pcd_isl_formetric = np.array(df.DIRS_ISL_FORMETRIC[i])

            _, _, centered_isl_formetric, scale_factor, lin_trans_factor = crop_normalize_center_pcd(
                pcd_backscan, pcd_esl, pcd_isl_formetric, df.fixPoints[i])

            sampled_isl_formetric = upsample_3d_line(centered_isl_formetric)
        else:
            pcd_esl = df.ESL[i]
            pcd_isl = df.ISL[i]

        centered_backscan, centered_esl, centered_isl, scale_factor, lin_trans_factor = crop_normalize_center_pcd(
            pcd_backscan, pcd_esl, pcd_isl, df.fixPoints[i])
        df.scaling_factor[i] = scale_factor
        df.linear_factor[i] = lin_trans_factor

        sampled_backscan = resample_pointclouds(centered_backscan, num_points, "poisson")
        sampled_esl = upsample_3d_line(centered_esl)
        sampled_isl = upsample_3d_line(centered_isl)
        broken = upsamling_sanity_check(sampled_backscan)

        if broken:
            df.failedUpsampling[i] = True
            print("Scan", df.Subject_ID[i], "failed the upsampling")
        else:
            fixPoints_pp = {}
            for key, value in df.fixPoints[i].items():
                fixPoints_pp[key] = ((np.asarray(value) * scale_factor) - lin_trans_factor).tolist()

            back_path = "./preprocessed/backscan/backscan_pp_" + str(df.Subject_ID[i]) + ".npy"
            esl_path = "./preprocessed/esl/esl_pp_" + str(df.Subject_ID[i]) + ".npy"
            isl_path = "./preprocessed/isl/isl_pp_" + str(df.Subject_ID[i]) + ".npy"
            fixPts_path = "./preprocessed/fixPoints/fixPoints_pp_" + str(df.Subject_ID[i]) + ".json"

            if italian:
                isl_formetric_path = "./preprocessed/isl_formetric/isl_formetric_pp_" + str(df.Subject_ID[i]) + ".npy"
                np.save(isl_formetric_path, sampled_isl_formetric)
                df.isl_formetric_pp[i] = isl_formetric_path

            np.save(back_path, sampled_backscan)
            np.save(esl_path, sampled_esl)
            np.save(isl_path, sampled_isl)

            with open(fixPts_path, "w") as f:
                json.dump(fixPoints_pp, f)

            df.back_pp[i] = back_path
            df.esl_pp[i] = esl_path
            df.isl_pp[i] = isl_path
            df.fixPoints_pp[i] = fixPts_path

    print("\n %i scans failed the upsampling in the dataset!" % df['failedUpsampling'].sum())

    if italian:
        df_cleaned = df.drop(df[df["failedUpsampling"]].index).reset_index(drop=True)
        df_cleaned.to_pickle(f"italian{str(num_points)}_dataset.pkl")
    else:
        df_cleaned = df.drop(df[df["isDuplicate"]].index)
        df_cleaned = df_cleaned.drop(df_cleaned[df_cleaned["failedUpsampling"]].index)
        df_cleaned = df_cleaned.reset_index(drop=True)

    return df, df_cleaned


def crop_normalize_center_pcd(back: np.ndarray, esl: np.ndarray, isl: np.ndarray, fixPts_list: np.ndarray):
    """
    Crop, normalize and center the point clouds
    Args:
        back: numpy array of back surface scan
        esl: numpy array of ESL line
        isl: numpy array of ISL line
        fixPts_list: dict of fix points values

    Returns:
        centered_back, centered_esl, centered_isl, scale_factor, translation

    """
    pointC5 = int(fixPts_list['fix_C5'][1])  # fixPts list, y-value
    pointSP = int(fixPts_list['fix_SP'][1])  # fixPts list, y-value

    pointC7 = int(fixPts_list['fix_C7'][1])  # fixPts list, y-value
    pointDM = int(fixPts_list['fix_DM'][1])  # fixPts list, y-value

    # Create a boolean mask to select points based on their y-value
    mask1 = (back[:, 1] <= pointC5) & (back[:, 1] >= pointSP)
    mask2 = (esl[:, 1] <= pointC7) & (esl[:, 1] >= pointDM)
    mask3 = (isl[:, 1] <= pointC7) & (isl[:, 1] >= pointDM)

    # Apply the mask to the points array to remove unwanted points
    cropped_back = back[mask1]
    cropped_esl = esl[mask2]
    cropped_isl = isl[mask3]

    # Normalize the point clouds to the center
    target_distance = 1
    distance = pointC5 - pointSP
    scale_factor = target_distance / distance

    normalized_back = cropped_back * scale_factor
    normalized_esl = cropped_esl * scale_factor
    normalized_isl = cropped_isl * scale_factor
    normalized_C5 = pointC5 * scale_factor
    normalized_SP = pointSP * scale_factor

    # Translate the point clouds to the center
    max_vals = np.amax(normalized_back, axis=0)
    min_vals = np.amin(normalized_back, axis=0)
    translation = [(max_vals[0] + min_vals[0]) / 2, (normalized_C5 + normalized_SP) / 2,
                   (max_vals[2] + min_vals[2]) / 2]

    centered_back = normalized_back - translation
    centered_esl = normalized_esl - translation
    centered_isl = normalized_isl - translation

    return centered_back, centered_esl, centered_isl, scale_factor, translation


def random_augment_pcd(df: pd.DataFrame, rot_limit: float = 0, rot_angle: float = 3):
    def rot_x(back, esl, isl, fixPoints):
        # Convert the angle range from degrees to radians
        min_angle_rad = np.deg2rad(-rot_angle)
        max_angle_rad = np.deg2rad(rot_angle)

        # Generate a random rotation angle within the specified range
        rotation_angle = np.random.uniform(min_angle_rad, max_angle_rad)

        # Define the rotation matrix
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [0, np.sin(rotation_angle), np.cos(rotation_angle)]])

        # Apply the rotation to the point cloud
        back = np.dot(back, rotation_matrix.T)
        esl = np.dot(esl, rotation_matrix.T)
        isl = np.dot(isl, rotation_matrix.T)

        for key, value in fixPoints.items():
            fixPoints[key] = np.dot(np.asarray(value), rotation_matrix.T).tolist()

        print("Performing rotation around x-axis", np.rad2deg(rotation_angle))

        return back, esl, isl, fixPoints

    def rot_y(back, esl, isl, fixPoints):
        # Convert the angle range from degrees to radians
        min_angle_rad = np.deg2rad(-rot_angle)
        max_angle_rad = np.deg2rad(rot_angle)

        # Generate a random rotation angle within the specified range
        rotation_angle = np.random.uniform(min_angle_rad, max_angle_rad)

        # Define the rotation matrix for rotation about the y-axis
        rotation_matrix = np.array([[np.cos(rotation_angle), 0, np.sin(rotation_angle)],
                                    [0, 1, 0],
                                    [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]])

        # Apply the rotation to the point cloud
        back = np.dot(back, rotation_matrix.T)
        esl = np.dot(esl, rotation_matrix.T)
        isl = np.dot(isl, rotation_matrix.T)

        for key, value in fixPoints.items():
            fixPoints[key] = np.dot(np.asarray(value), rotation_matrix.T).tolist()

        print("Performing rotation around y-axis", np.rad2deg(rotation_angle))

        return back, esl, isl, fixPoints

    def rot_z(back, esl, isl, fixPoints):
        # Convert the angle range from degrees to radians
        min_angle_rad = np.deg2rad(-rot_angle)
        max_angle_rad = np.deg2rad(rot_angle)

        # Generate a random rotation angle within the specified range
        rotation_angle = np.random.uniform(min_angle_rad, max_angle_rad)

        # Define the rotation matrix for rotation about the z-axis
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                                    [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                                    [0, 0, 1]])

        # Apply the rotation to the point cloud
        back = np.dot(back, rotation_matrix.T)
        esl = np.dot(esl, rotation_matrix.T)
        isl = np.dot(isl, rotation_matrix.T)

        for key, value in fixPoints.items():
            fixPoints[key] = np.dot(np.asarray(value), rotation_matrix.T).tolist()

        print("Performing rotation around z-axis", np.rad2deg(rotation_angle))

        return back, esl, isl, fixPoints

    def mirror_yx(back, esl, isl, fixPoints):
        back[:, 0] = -back[:, 0]
        esl[:, 0] = -esl[:, 0]
        isl[:, 0] = -isl[:, 0]
        fixPoints[:, 0] = -fixPoints[:, 0]

        # print("Performing mirroring in y-z-plane")

        return back, esl, isl, fixPoints

    # Create a dictionary to map random integers to actions
    actions = {
        1: rot_x,
        2: rot_y,
        3: rot_z,
        4: mirror_yx,
    }

    average = df['surfaceRotation'].mean()
    print(average)

    paths = [pathlib.Path("./preprocessed/backscan/aug"), pathlib.Path("./preprocessed/esl/aug"),
             pathlib.Path("./preprocessed/isl/aug"), pathlib.Path("./preprocessed/fixPoints/aug")]

    for path in paths:
        if not path.exists():
            path.mkdir(parents=True)

    df_original_length = len(df)
    for i in range(2):
        for idx in tqdm(range(df_original_length)):
            if df.surfaceRotation[idx] > rot_limit:
                back = np.load(df.back_pp[idx])
                esl = np.load(df.esl_pp[idx])
                isl = np.load(df.isl_pp[idx])

                with open(df.fixPoints_pp[idx], "r") as f:
                    fixPoints = json.load(f)

                # display_pcd_plt(back, isl)

                # Generate a random integer between 1 and 4, in case without mirroring 1 to 3
                random_integer = random.randint(1, 3)

                # Execute the corresponding action based on the random integer
                if random_integer in actions:
                    back, esl, isl, fixPoints = actions[random_integer](back, esl, isl, fixPoints)
                else:
                    print("Invalid random integer")

                # display_pcd_plt(back, isl)

                row = df.iloc[idx].to_frame().T
                df = pd.concat([df, row], ignore_index=True)

                new_subjectID = f"{df.Subject_ID[idx]}_{i + 1}"

                back_path = f"./preprocessed/backscan/aug/backscan_pp_{new_subjectID}.npy"
                esl_path = f"./preprocessed/esl/aug/esl_pp_{new_subjectID}.npy"
                isl_path = f"./preprocessed/isl/aug/isl_pp_{new_subjectID}.npy"
                fixPts_path = f"./preprocessed/fixPoints/aug/fixPoints_pp_{new_subjectID}.json"

                np.save(back_path, back)
                np.save(esl_path, esl)
                np.save(isl_path, isl)

                with open(fixPts_path, "w") as f:
                    json.dump(fixPoints, f)

                df.at[df.index[-1], "back_pp"] = back_path
                df.at[df.index[-1], "esl_pp"] = esl_path
                df.at[df.index[-1], "isl_pp"] = isl_path
                df.at[df.index[-1], "fixPoints_pp"] = fixPts_path
                df.at[df.index[-1], "Subject_ID"] = new_subjectID

        print(len(df))
        average = df['surfaceRotation'].mean()
        print(average)

    df.to_pickle("balanced1024_dataset.pkl")

    # plt.boxplot(df['surfaceRotation'])
    # plt.show()

    return df


def findPointsInROI(pcd, roi):
    length = len(roi[0])
    inds = np.empty(length, dtype=object)

    for i in range(length):
        x_filter = np.logical_and(pcd[:, 0] >= roi[0][i], pcd[:, 0] <= roi[1][i])
        y_filter = np.logical_and(pcd[:, 1] >= roi[2][i], pcd[:, 1] <= roi[3][i])
        z_filter = np.logical_and(pcd[:, 2] >= roi[4][i], pcd[:, 2] <= roi[5][i])
        ind = np.where(np.logical_and(np.logical_and(x_filter, y_filter), z_filter))[0]
        inds[i] = ind
    return inds


def fixed_pcd_regularization(pcd: np.ndarray, show=False, ID="xx"):
    pcd = pcd * 640  # scale to 640 pixel since the point cloud is normalized
    pcd_ply = convert_np_to_ply(pcd)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_ply)

    # regularize the point cloud
    distSum = 0
    for i in range(pcd.shape[0]):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd[i], 2)  # find 2 points because the first one is always itself
        dist = np.linalg.norm(pcd[i] - pcd[idx[1]])
        distSum = distSum + dist
    avgDist = distSum / pcd.shape[0]
    mfactor = 1.5
    avgDist = mfactor * avgDist

    pcXGrid = np.arange(-240, 240, 1)
    pcYGrid = np.arange(-320, 320, 1)
    X, Y = np.meshgrid(pcXGrid, pcYGrid)

    # Define the upper and lower limits for z values
    z_upper_lim = 100
    z_lower_lim = -125

    # Calculate the scaling factor to map the z values to pixel values (0-1)
    scale_factor = 1 / (z_upper_lim - z_lower_lim)

    delta2 = mfactor * avgDist
    xx, yy = np.meshgrid(pcXGrid, pcYGrid)
    length = 480 * 640
    inf = np.ones(length) * np.inf
    minus_inf = np.ones(length) * (-np.inf)

    roi_inds = findPointsInROI(pcd, [np.ravel(xx) - delta2, np.ravel(xx) + delta2,
                                     np.ravel(yy) - delta2, np.ravel(yy) + delta2, minus_inf, inf])

    roi_pts = np.array([pcd[x, :] for x in roi_inds], dtype=object)
    z_median = np.array([(np.median(x[:, 2]) - z_lower_lim) * scale_factor if len(x) != 0 else 0.0 for x in roi_pts],
                        dtype=object)
    Z = z_median.reshape(X.shape)
    Z = Z.astype('float64')

    filename = "./preprocessed/depthmaps/depthmap_" + ID + ".jpeg"

    image = Image.fromarray(Z * 255)
    rotated_image = image.rotate(180)
    rotated_image = rotated_image.convert("L")  # Convert to 8-bit grayscale mode
    rotated_image.save(filename, "JPEG")
    print(f"Figure saved in directory: {filename}")

    if show:
        Z_image = Image.fromarray(Z.T)
        Z_image.show()

    return X, Y, Z


def create_dataset_formetric(DATA_DIR: str, num_points: int = 1024):
    """
    Pipeline for creating a formetric dataset
    Args:
        DATA_DIR: directory of the formetric raw data
        num_points: desired number of points for resampling the back surface scans

    Returns:
        df: dataframe containing infos about the subjects as well as the paths to the preprocessed data
        df_cleaned: dataframe without the subjects with invalid resampling

    """
    # Get all paths and information
    df_infos = get_and_parse_data(DATA_DIR)

    # Check duplicated scans
    df, df_cleaned = check_duplicated_files(df_infos)

    # Pre-Process and up sample data
    df, df_cleaned = preprocess_and_save_data(df_cleaned, num_points=num_points)

    # save dataframe to pickle
    df_cleaned.to_pickle(f"original{str(num_points)}_dataset_new.pkl")

    return df, df_cleaned


def prepare_dataset(df: pd.DataFrame, start=0, stop=5000, num_points: int = 30, depthmaps: bool = False):
    """
    Pipeline for preparing the dataset for training
    - resampling ESL / ISL to desired number of points
    - creating depthmaps
    Args:
        df: dataframe containing all subjects to prepare
        start: in case of using argparser, start index of the dataframe
        stop: in case of using argparser, stop index of the dataframe
        num_points: desired number of points for ESL / ISL resampling
        depthmaps: bool whether to convert the back surface scans to depthmaps

    """
    for i in tqdm(range(start, stop)):
        ID = str(df.iloc[i]['Subject_ID'])
        itr = int(300 / num_points)

        # Resample ISL
        isl = np.load(df.isl_pp[i])
        isl_resized = isl[::itr]
        stacked_isl = np.stack([isl_resized[:, 0], isl_resized[:, 2]], axis=0).reshape(-1, 1)

        isl_spines_path = pathlib.Path("./preprocessed/isl_30")
        if not isl_spines_path.exists():
            isl_spines_path.mkdir(parents=True)
        np.save("./preprocessed/isl_30/isl_30_" + str(ID) + ".npy", stacked_isl)

        # Resample ESL
        esl = np.load(df.esl_pp[i])
        esl_resized = esl[::itr]
        stacked_esl = np.stack([esl_resized[:, 0], esl_resized[:, 2]], axis=0).reshape(-1, 1)

        esl_spines_path = pathlib.Path("./preprocessed/esl_30")
        if not esl_spines_path.exists():
            esl_spines_path.mkdir(parents=True)
        np.save("./preprocessed/esl_30/esl_30_" + str(ID) + ".npy", stacked_esl)

        # Create depthmaps of back surface scans
        if depthmaps:
            back = np.load(df.back_pp[i])
            back_path = pathlib.Path("./preprocessed/depthmaps")
            if not back_path.exists():
                back_path.mkdir(parents=True)
            image_bs = fixed_pcd_regularization(back, ID=ID)

    print("Preparing dataset finished!")


def clean_italian(dir_dataset, dir_file, destination="raw_data/Italian_cleaned"):
    """
    Clean the Italian dataset by removing the subjects with invalid registration
    Args:
        dir_dataset: directory of raw italian data
        dir_file: directory of file containing information about registration quality
        destination: desired destination directory for the cleaned data

    """
    all_subjects_paths = natsorted(glob.glob(str(pathlib.Path(dir_dataset) / "*")))
    df = pd.read_csv(dir_file, delimiter=' ')

    for idx, path in tqdm(enumerate(all_subjects_paths)):
        path_id = re.search(f'{str(dir_dataset)}/(\d+)', str(path)).group(1)
        quality_xy = df.loc[df['SubjectID'] == int(path_id), 'QualityXY'].iloc[0]
        quality_yz = df.loc[df['SubjectID'] == int(path_id), 'QualityYZ'].iloc[0]

        if quality_xy == 1 and quality_yz == 1:
            shutil.copytree(path, f"{destination}/{path_id}")


if __name__ == "__main__":
    # # Initialize the argparse module
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int, help="Start value", nargs='?', default=0)
    parser.add_argument("stop", type=int, help="Stop value", nargs='?', default=10)
    args = parser.parse_args()

    # # to create a new (croatian) dataset uncomment this line
    # df, df_cleaned = create_dataset_formetric("./raw_data/Formetric_Croatian_data_August2023", 1024)
    # random_augment_pcd(df_cleaned, rot_limit=5)
    # prepare_dataset(pd.read_pickle("original1024_dataset.pkl"), stop=len(pd.read_pickle("original1024_datasets.pkl")))

    # # to clean invalid scans from italian dataset uncomment this line
    # clean_italian("raw_data/Italian_data_cleaned_July2023", "raw_data/evaluation_results.txt")

    # # to create a new (italian) dataset uncomment this line
    # df, df_cleaned = create_dataset_italian("./raw_data/Italian_data_cleaned_July2023", num_points=1024)
    # prepare_dataset(df_cleaned, stop=len(df_cleaned))

    # # to plot ply/json data uncomment this line
    # back = convert_ply_to_np("raw_data/Formetric_Croatian_data_July2023/croatian_457/457_backscan.ply")
    # with open("raw_data/Formetric_Croatian_data_July2023/croatian_457/457_metadata.json", "r") as f:
    #     data = json.load(f)
    #     fixPoints_dict = data["fixPts"]
    # fixPoints_np = np.array(list(fixPoints_dict.values()))
    # display_pcd_plotly(back, fixPoints_np)
