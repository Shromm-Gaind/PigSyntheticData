import json
import numpy as np
import cv2
from PIL import Image
import os
import re
import matplotlib.pyplot as plt

class FileMatchingException(Exception):
    pass

def extract_index_from_filename(filename):
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract index from filename: {filename}")

def get_class_for_index(index, json_data):
    for class_name, class_info in json_data.items():
        if 'start_index' in class_info and 'end_index' in class_info:
            if class_info['start_index'] <= index <= class_info['end_index']:
                return class_name, class_info['class']
    raise Exception(f"No class found for index {index}")

def extract_ids(file_list, pattern):
    ids = []
    for file in file_list:
        match = re.search(pattern, file)
        if match:
            ids.append(int(match.group(1)))
    return np.array(ids), np.array(file_list)

def match_files(joint_segmentation_files, segmentation_files, rgb_files):
    joint_segmentation_ids, _ = extract_ids(joint_segmentation_files, r"(\d+)\.bmp$")
    segmentation_ids, _ = extract_ids(segmentation_files, r"(\d+)\.bmp$")
    rgb_ids, _ = extract_ids(rgb_files, r"(\d+)\.bmp$")

    matched_files = {}
    for id in np.unique(np.concatenate([joint_segmentation_ids, segmentation_ids, rgb_ids])):
        if id in joint_segmentation_ids and id in segmentation_ids and id in rgb_ids:
            matched_files[id] = {
                "joint_segmentation": next(f for f in joint_segmentation_files if f.startswith(f"{id}")),
                "segmentation": next(f for f in segmentation_files if f.startswith(f"{id}")),
                "rgb": next(f for f in rgb_files if f.startswith(f"{id}")),
            }
    return matched_files

def load_joint_colors_from_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)
class NoJointsDetectedException(Exception):  # Exception for no joints detected
    pass

def detect_joints_in_image(image_path, joint_rgb_values):
    joint_img = Image.open(image_path)
    joint_img_np = np.array(joint_img)
    detected_joints = {}
    img_width, img_height = joint_img_np.shape[1], joint_img_np.shape[0]
    found_joints = False  # Flag to track if any joints are found

    for joint_name, rgb_value in joint_rgb_values.items():
        lower_bound = np.array(rgb_value) - 1
        upper_bound = np.array(rgb_value) + 1
        mask = cv2.inRange(joint_img_np, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            found_joints = True
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"]) / img_width
                cY = int(M["m01"] / M["m00"]) / img_height
                detected_joints[joint_name] = (cX, cY)

    if not found_joints:
        raise NoJointsDetectedException(f"No joints detected in image: {image_path}")

    return detected_joints
def calculate_2d_bbox(segmentation_map):
    non_black_color = np.any(segmentation_map != (0, 0, 0), axis=-1)
    y_idx, x_idx = np.where(non_black_color)
    if not y_idx.size or not x_idx.size:
        return None
    x_min, x_max = np.min(x_idx), np.max(x_idx)
    y_min, y_max = np.min(y_idx), np.max(y_idx)
    return x_min, y_min, x_max - x_min, y_max - y_min


# Paths
base_path_joint_segmentation = r""
base_path_rgb = r""
base_path_segmentation = r""
json_data_path = r""
save_directory = r""

json_data = load_joint_colors_from_json(json_data_path)

# Assuming you have a list of file names for each type
joint_segmentation_files = os.listdir(base_path_joint_segmentation)
segmentation_files = os.listdir(base_path_segmentation)
rgb_files = os.listdir(base_path_rgb)

matched_files = match_files(joint_segmentation_files, segmentation_files, rgb_files)


def process_images(matched_files, base_path_joint_segmentation, base_path_rgb, base_path_segmentation, save_directory, json_data, class_names):

    for id, paths in matched_files.items():
        joint_segmentation_path = os.path.join(base_path_joint_segmentation, paths["joint_segmentation"])
        segmentation_path = os.path.join(base_path_segmentation, paths["segmentation"])
        rgb_path = os.path.join(base_path_rgb, paths["rgb"])

        # Load images
        joint_segmentation_img = np.array(Image.open(joint_segmentation_path))
        segmentation_img = np.array(Image.open(segmentation_path))
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # Calculate bounding box from segmentation image
        bbox = calculate_2d_bbox(segmentation_img)
        if bbox is None:
            continue
        x_min, y_min, bbox_width, bbox_height = bbox

        # Normalize bbox coordinates
        height, width, _ = rgb_img.shape
        x_center = (x_min + bbox_width / 2) / width
        y_center = (y_min + bbox_height / 2) / height
        norm_width = bbox_width / width
        norm_height = bbox_height / height

        # Get class index from JSON data using image index
        class_name, class_index = get_class_for_index(id, json_data)

        # Start annotation with class index and bounding box
        annotation = f"{class_index} {x_center} {y_center} {norm_width} {norm_height}"

        # Detect and normalize keypoints from joint_segmentation_img
        joint_rgb_values = {joint: info["RGB"] for joint, info in json_data[class_name]["Joints"].items()}
        detected_joints = detect_joints_in_image(joint_segmentation_path, joint_rgb_values)

        # Print the number of detected joints
        num_detected_joints = len(detected_joints)
        print(f"Number of detected joints in image {id}: {num_detected_joints}")

        """
        # Assuming joint_segmentation_img is loaded and detected_joints contains normalized coordinates
        plt.figure(figsize=(10, 5))
        plt.imshow(joint_segmentation_img)  # Display the joint segmentation image

        # Iterate over detected joints to plot them
        for joint_name, (cX, cY) in detected_joints.items():
            # Convert normalized coordinates back to image coordinates for plotting
            plot_x, plot_y = cX * joint_segmentation_img.shape[1], cY * joint_segmentation_img.shape[0]
            plt.scatter([plot_x], [plot_y], label=joint_name)  # Plot each joint

        plt.title(f"Joint Segmentation Image {id} with Detected Joints")
        plt.legend()
        plt.show()
        """
        # Initialize keypoints list with placeholders. These will be replaced or used as fallbacks.
        keypoints = ["-1 -1"] * len(class_names)

        fallback_coords = None  # Initialize a variable to hold fallback coordinates

        for joint_name, coords in detected_joints.items():
            if coords:
                index = class_names.index(joint_name)
                keypoints_annotation = f"{coords[0]} {coords[1]}"
                keypoints[index] = keypoints_annotation

                # Set the first detected joint's coordinates as the fallback, if not already set
                if fallback_coords is None:
                    fallback_coords = keypoints_annotation

        # If fallback_coords is still None (no joints detected), you can decide on a default strategy
        # For the purpose of demonstration, let's use "0 0" as a universal fallback if no joints were detected at all.
        if fallback_coords is None:
            fallback_coords = "0 0"

        # Now, replace any remaining "-1 -1" placeholders with the fallback_coords
        keypoints = [kp if kp != "-1 -1" else fallback_coords for kp in keypoints]

        # Append keypoints to the annotation. No need to replace placeholders now.
        annotation += ' ' + ' '.join(keypoints)

        # Save annotation to file
        annotation_filename = f"{id}.txt"
        annotation_path = os.path.join(save_directory, annotation_filename)
        with open(annotation_path, 'w') as file:
            file.write(annotation + "\n")




def create_dataset_yaml(save_directory, json_data_path, train_images_dir, val_images_dir, class_names, flip_pairs):
    dataset_config = {
        'path': save_directory,
        'train': train_images_dir,
        'val': val_images_dir,
        'names': class_names,
        'kpt_shape': [len(class_names), 2],  # Assuming 2D keypoints (x, y)
        'flip_idx': flip_pairs
    }

    yaml_path = os.path.join(save_directory, 'dataset_config.yaml')
    with open(yaml_path, 'w') as yaml_file:
        json.dump(dataset_config, yaml_file, indent=2)

    print(f"Dataset YAML created at {yaml_path}")
    return yaml_path


class_names = [
    'Snout', 'Neck',
    'Base left ear', 'Base right ear', 'Tip left ear', 'Tip right ear',
    'Left shoulder', 'Right shoulder', 'Left elbow', 'Right elbow',
    'Left hand', 'Right hand', 'Left flank', 'Right flank', 'Left hip',
    'Right hip', 'Left knee', 'Right knee', 'Left foot', 'Right foot',
    'Base tail', 'Tip tail'
]
process_images(matched_files, base_path_joint_segmentation, base_path_rgb, base_path_segmentation, save_directory, json_data, class_names)

# Define flip pairs based on the symmetrical keypoints
flip_pairs = [
    (2, 3),  # 'Base left ear', 'Base right ear'
    (4, 5),  # 'Tip left ear', 'Tip right ear'
    (6, 7),  # 'Left shoulder', 'Right shoulder'
    (8, 9),  # 'Left elbow', 'Right elbow'
    (10, 11), # 'Left hand', 'Right hand'
    (12, 13), # 'Left flank', 'Right flank'
    (14, 15), # 'Left hip', 'Right hip'
    (16, 17), # 'Left knee', 'Right knee'
    (18, 19)  # 'Left foot', 'Right foot'
]

train_images_dir = ""
val_images_dir = ""

# Assuming paths and other variables are defined as in the previous example
create_dataset_yaml(save_directory, json_data_path, train_images_dir, val_images_dir, class_names, flip_pairs)