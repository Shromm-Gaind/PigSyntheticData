import json
import numpy as np
import os
from shutil import copy2
import random

def load_camera_poses(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return {str(item['Index']): item['CameraLocation'] for item in data}

def calculate_angular_difference(loc1, loc2):
    V1 = np.array([loc1['X'], loc1['Y'], loc1['Z']])
    V2 = np.array([loc2['X'], loc2['Y'], loc2['Z']])
    dot_product = np.dot(V1, V2)
    magnitude_V1 = np.linalg.norm(V1)
    magnitude_V2 = np.linalg.norm(V2)
    cos_theta = dot_product / (magnitude_V1 * magnitude_V2)
    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees
def copy_files(indices, src_base_dir, dest_base_dir, category, image_type, ext):
    for idx in indices:
        src_path = os.path.join(src_base_dir, category, image_type, f"{idx}{ext}")
        dest_path = os.path.join(dest_base_dir, category, image_type, f"{idx}{ext}")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if os.path.exists(src_path):
            copy2(src_path, dest_path)
        else:
            print(f"File {src_path} not found.")
def filter_images_with_new_threshold(base_poses, compare_poses, lower_threshold=20, upper_threshold=40, set_name="validation"):
    filtered_indices = {}
    for idx, pose in compare_poses.items():
        within_threshold = False
        for base_idx, base_pose in base_poses.items():
            angle_diff = calculate_angular_difference(pose, base_pose)
            if lower_threshold <= angle_diff <= upper_threshold:
                within_threshold = True
                break
        if within_threshold:
            filtered_indices[idx] = pose
    return filtered_indices

def refine_training_set_with_new_threshold(training_poses, validation_poses, lower_threshold=20, upper_threshold=40):
    refined_training_indices = []
    for idx, pose in training_poses.items():
        within_threshold = False
        for val_idx, val_pose in validation_poses.items():
            angle_diff = calculate_angular_difference(pose, val_pose)
            if lower_threshold <= angle_diff <= upper_threshold:
                within_threshold = True
                break
        if not within_threshold:
            refined_training_indices.append(idx)
    return {idx: training_poses[idx] for idx in refined_training_indices}

def filter_poses_exclude_close(base_poses, compare_poses, lower_threshold, upper_threshold):
    """
    Excludes poses from compare_poses that are too close to any pose in base_poses.

    Args:
    - base_poses (dict): Base poses for comparison.
    - compare_poses (dict): Poses to be filtered.
    - threshold (float): Angular difference threshold for exclusion.

    Returns:
    - dict: Poses from compare_poses that are not too close to base_poses.
    """
    filtered_indices = {}
    for idx, pose in compare_poses.items():
        too_close = False
        print(f"\nChecking pose {idx} for exclusion...")
        for base_idx, base_pose in base_poses.items():
            angle_diff = calculate_angular_difference(pose, base_pose)
            print(f"Comparing with base pose {base_idx}: Angle difference = {angle_diff} degrees")
            if lower_threshold <= angle_diff <= upper_threshold:
                too_close = True
                print(f"Pose {idx} is too close to base pose {base_idx} (within threshold). Excluding...")
                break
        if not too_close:  # Include pose if it's not too close to any base_pose
            filtered_indices[idx] = pose
            print(f"Pose {idx} is not too close to any base pose. Including in filtered set.")
    print(f"Total poses not too close to any base pose: {len(filtered_indices)}")

    return filtered_indices


def split_dataset_randomly(indices, train_fraction=0.2):
    """
    Randomly splits the indices into training and validation/test sets.

    Args:
    - indices (list): List of all indices.
    - train_fraction (float): Fraction of indices to be used for training.

    Returns:
    - tuple: (training_indices, validation_test_indices)
    """
    random.shuffle(indices)  # Randomize the order of indices
    split_point = int(len(indices) * train_fraction)  # Calculate split point
    training_indices = indices[:split_point]
    validation_test_indices = indices[split_point:]
    return training_indices, validation_test_indices


def main():
    camera_poses = load_camera_poses(r"E:\standing.json")  # Update this path
    indices = list(camera_poses.keys())

    training_indices, validation_test_indices = split_dataset_randomly(indices)


    # Initial pose sets
    training_poses = {idx: camera_poses[idx] for idx in training_indices}
    validation_test_poses = {idx: camera_poses[idx] for idx in validation_test_indices}
    lower_threshold = 1
    upper_threshold = 20

    filtered_validation_poses_with_new_threshold_20_50 = filter_poses_exclude_close(
        training_poses, validation_test_poses, lower_threshold, upper_threshold)

    refined_training_poses_with_threshold_5_10 = filter_poses_exclude_close(
        training_poses, filtered_validation_poses_with_new_threshold_20_50, lower_threshold, upper_threshold)



    base_dir = r''
    output_dir = r''
    image_types = {
        "Cage": [("RGBImages", ".bmp"), ("Val_Depth", ".png"), ("val_Segmentation_images", ".bmp")],
        "Joints": [("Val_Depth", ".png"), ("val_Segmentation_images", ".bmp")]
    }

    # Copy files for training, validation, and test sets
    for set_name, poses in [("train", refined_training_poses_with_threshold_5_10), (
            "validate", filtered_validation_poses_with_new_threshold_20_50)]:
        for category, types in image_types.items():
            for image_type, ext in types:
                dest_base_dir = os.path.join(output_dir, set_name)
                copy_files(poses.keys(), base_dir, dest_base_dir, category, image_type, ext)

    # Output counts for verification
    print(f"Filtered Validation/Test Poses: {len(filtered_validation_poses_with_new_threshold_20_50)}")
    print(f"Refined Training Poses: {len(training_poses)}")

if __name__ == "__main__":
    main()
