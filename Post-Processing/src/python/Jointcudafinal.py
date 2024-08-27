import numpy as np
import imageio as iio
import open3d as o3d
import cv2
from numba import cuda
import math
import os
import re
import json
import pickle


@cuda.jit
def depth_conversion_kernel(PointDepth, DistanceFromCenter, PlaneDepth, f):
    i, j = cuda.grid(2)
    if i < PointDepth.shape[0] and j < PointDepth.shape[1]:
        PlaneDepth[i, j] = PointDepth[i, j] / math.sqrt(1 + (DistanceFromCenter[i, j] / f) ** 2)


@cuda.jit
def create_point_cloud_kernel(pixel_depth, x, y, z, FX_DEPTH, FY_DEPTH, CX_DEPTH, CY_DEPTH):
    i, j = cuda.grid(2)
    if i < pixel_depth.shape[0] and j < pixel_depth.shape[1]:
        depth = pixel_depth[i, j] / 1000  # Convert to meters

        # Skip points with depth greater than 5 meters
        if depth > 5.0:
            return

        x[i, j] = ((j - CX_DEPTH) * depth / FX_DEPTH)
        y[i, j] = ((i - CY_DEPTH) * depth / FY_DEPTH)
        z[i, j] = depth


def depth_conversion(PointDepth, f):
    H, W = PointDepth.shape
    i_c = float(H) / 2 - 1
    j_c = float(W) / 2 - 1
    cols = np.linspace(0, W - 1, num=W)
    rows = np.linspace(0, H - 1, num=H).reshape(-1, 1)
    DistanceFromCenter = np.sqrt((rows - i_c) ** 2 + (cols - j_c) ** 2)

    # Allocate memory on the device
    PointDepth_device = cuda.to_device(PointDepth)
    DistanceFromCenter_device = cuda.to_device(DistanceFromCenter)
    PlaneDepth_device = cuda.device_array(PointDepth.shape, np.float32)

    # Configure the blocks and grid sizes more optimally
    threadsperblock = (32, 32)
    blockspergrid_x = max(1, math.ceil(PointDepth.shape[0] / threadsperblock[0]))
    blockspergrid_y = max(1, math.ceil(PointDepth.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Launch the kernel
    depth_conversion_kernel[blockspergrid, threadsperblock](PointDepth_device, DistanceFromCenter_device,
                                                            PlaneDepth_device, f)

    # Copy the result back to the host
    return PlaneDepth_device.copy_to_host()


def load_depth_image(file_path):
    depth_image = iio.v3.imread(file_path).astype(np.float32)
    return depth_image


def create_point_cloud_from_depth(depth_image, rgb_path, FX_DEPTH, FY_DEPTH, CX_DEPTH, CY_DEPTH):
    # Calculate pixel depth in millimeters
    pixel_depth_mm = (depth_image[:, :, 0] + depth_image[:, :, 1] * 256 + depth_image[:, :, 2] * 256 * 256)
    pixel_depth = depth_conversion(pixel_depth_mm, FX_DEPTH)

    H, W = pixel_depth.shape

    # Allocate memory on the device
    pixel_depth_device = cuda.to_device(pixel_depth)
    x_device = cuda.device_array((H, W), dtype=np.float32)
    y_device = cuda.device_array((H, W), dtype=np.float32)
    z_device = cuda.device_array((H, W), dtype=np.float32)

    # Configure the blocks
    threads_per_block = (32, 32)  # Adjust based on your GPU's architecture
    blocks_per_grid_x = math.ceil(W / threads_per_block[0])
    blocks_per_grid_y = math.ceil(H / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch the kernel
    create_point_cloud_kernel[blocks_per_grid, threads_per_block](pixel_depth_device, x_device, y_device, z_device,
                                                                  FX_DEPTH, FY_DEPTH, CX_DEPTH, CY_DEPTH)

    # Copy the result back to the host
    x = x_device.copy_to_host()
    y = y_device.copy_to_host()
    z = z_device.copy_to_host()

    # Explicitly deallocate device memory
    del x_device, y_device, z_device

    # Read the RGB image using OpenCV
    rgb_image = cv2.imread(rgb_path)
    if rgb_image is None:
        raise FileNotFoundError(f"RGB image not found at path: {rgb_path}")
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Compute coordinates u and v (filter out points where z=0)
    non_zero_z_indices = np.nonzero(z)  # Change to np.nonzero
    x_nz, y_nz, z_nz = x[non_zero_z_indices], y[non_zero_z_indices], z[non_zero_z_indices]
    u = np.clip(np.round(FX_DEPTH * x_nz / z_nz + CX_DEPTH).astype(int), 0, rgb_image.shape[1] - 1)
    v = np.clip(np.round(FY_DEPTH * y_nz / z_nz + CY_DEPTH).astype(int), 0, rgb_image.shape[0] - 1)

    # Fetch RGB values based on u and v
    rgb_values = rgb_image[v, u, :].astype(np.float32) / 255.0

    # Combine XYZ and RGB values
    point_cloud_with_color = np.zeros((x_nz.size, 6), dtype=np.float32)  # Adjusted array size
    point_cloud_with_color[:, :3] = np.column_stack((x_nz, y_nz, z_nz))  # Use non-zero xyz values
    point_cloud_with_color[:, 3:] = rgb_values

    return point_cloud_with_color


class IncompleteJointsException(Exception):
    pass


def find_and_count_valid_joints(depth_image, joint_segmentation_image, joint_colors, FX_DEPTH, FY_DEPTH, CX_DEPTH,
                                CY_DEPTH):
    # Validate input images
    if depth_image is None or joint_segmentation_image is None:
        raise InvalidImageException("Invalid or None depth/joint segmentation image provided.")

    joint_locations = {}
    valid_joint_count = 0
    depth_mm = depth_image[:, :, 0] + depth_image[:, :, 1] * 256 + depth_image[:, :, 2] * 256 * 256
    converted_depth = depth_conversion(depth_mm, FX_DEPTH)

    for joint_name, color in joint_colors.items():
        lower_bound = np.array(color) - 2
        upper_bound = np.array(color) + 2
        mask = cv2.inRange(joint_segmentation_image, lower_bound, upper_bound)
        y, x = np.where(mask > 0)

        if len(x) > 0 and len(y) > 0:
            valid_depths = converted_depth[y, x] > 0  # Check for valid depth values
            if np.any(valid_depths):
                avg_depth_mm = np.mean(converted_depth[y[valid_depths], x[valid_depths]])
                avg_x = (np.mean(x[valid_depths]) - CX_DEPTH) * avg_depth_mm / FX_DEPTH / 1000
                avg_y = (np.mean(y[valid_depths]) - CY_DEPTH) * avg_depth_mm / FY_DEPTH / 1000
                avg_z = avg_depth_mm / 1000
                joint_locations[joint_name] = (avg_x, avg_y, avg_z)
                valid_joint_count += 1
        # No exception is raised here for missing joints

    # Print the number of detected joints
    print(f"Number of joints detected in the image: {valid_joint_count}")

    if valid_joint_count == 0:
        raise NoJointsDetectedException("No joints were detected in the image.")

    return joint_locations


# helper function
def are_points_in_bbox(points, bbox, min_points=5):
    bbox_min = np.asarray(bbox.get_min_bound())
    bbox_max = np.asarray(bbox.get_max_bound())
    in_bbox = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
    points_in_bbox = points[in_bbox]
    return len(points_in_bbox) >= min_points

def create_bounding_box_from_joints(joint_locations, point_cloud, min_points=5):
    if not joint_locations:
        print("No joint locations provided.")
        return None, None, None

    hardcoded_radii = {
        'Snout': 0.08, 'Neck': 0.048, 'Base left ear': 0.048, 'Base right ear': 0.048,
        'Tip left ear': 0.048, 'Tip right ear': 0.048, 'Left shoulder': 0.096,
        'Right shoulder': 0.096, 'Left elbow': 0.096, 'Right elbow': 0.096,
        'Left hand': 0.064, 'Right hand': 0.064, 'Left flank': 0.048,
        'Right flank': 0.048, 'Left hip': 0.096, 'Right hip': 0.096,
        'Left knee': 0.096, 'Right knee': 0.096, 'Left foot': 0.096,
        'Right foot': 0.096, 'Base tail': 0.048, 'Tip tail': 0.032
    }


    points_for_overall_bbox = []  # Store points for the overall bounding box
    individual_bboxes = {}
    individual_bbox_data = {}

    for joint_name, location in joint_locations.items():
        sphere_radius = hardcoded_radii.get(joint_name, 0.02)
        joint_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius).translate(location)
        joint_sphere_bbox = joint_sphere.get_axis_aligned_bounding_box()

        # Prepare point cloud data for are_points_in_bbox
        point_cloud_np = np.asarray(point_cloud.points) if isinstance(point_cloud, o3d.geometry.PointCloud) else point_cloud

        # Dynamic adjustment for min_points based on specific joints
        adjusted_min_points = 1 if joint_name in ['Base left ear', 'Base right ear', 'Tip left ear', 'Tip right ear', 'Left flank', 'Tip tail'] else min_points

        if are_points_in_bbox(point_cloud_np, joint_sphere_bbox, adjusted_min_points):
            joint_centroid = joint_sphere_bbox.get_center()
            joint_dimensions = joint_sphere_bbox.get_extent()
            individual_bboxes[joint_name] = joint_sphere_bbox
            individual_bbox_data[joint_name] = {'centroid': joint_centroid, 'dimensions': joint_dimensions}
            points_for_overall_bbox.append(joint_centroid)

    # Check if the number of joints found is 0 or 1
    if len(individual_bboxes) <= 1:
        raise Exception("Insufficient valid joints found. Need at least two joints with enough points in their bounding boxes.")

    if points_for_overall_bbox:
        overall_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(points_for_overall_bbox)
        )
        overall_centroid = overall_bbox.get_center()
        overall_dimensions = overall_bbox.get_extent()
    else:
        print("No valid joints found for overall bounding box.")
        return None, {}, {}

    return (overall_centroid, overall_dimensions), individual_bboxes, individual_bbox_data

def print_rgb_values_in_joint_bounding_boxes(joint_locations, rgb_image, sphere_radius=0.02):
    for joint_name, joint_location in joint_locations.items():
        # Convert joint location to pixel coordinates
        u = int(FX_DEPTH * joint_location[0] / joint_location[2] + CX_DEPTH)
        v = int(FY_DEPTH * joint_location[1] / joint_location[2] + CY_DEPTH)

        # Define bounding box around the joint
        radius_in_pixels = int(sphere_radius * FX_DEPTH / joint_location[2])  # Assuming sphere radius in meters
        bbox_min_u, bbox_max_u = max(0, u - radius_in_pixels), min(rgb_image.shape[1], u + radius_in_pixels)
        bbox_min_v, bbox_max_v = max(0, v - radius_in_pixels), min(rgb_image.shape[0], v + radius_in_pixels)

        # Check if the bounding box is valid and contains data
        if bbox_min_u < bbox_max_u and bbox_min_v < bbox_max_v:
            rgb_values = rgb_image[bbox_min_v:bbox_max_v, bbox_min_u:bbox_max_u, :]
            if rgb_values.size > 0:
                # Print the RGB values
                print(f"RGB values near joint '{joint_name}':")
                for row in rgb_values:
                    for pixel in row:
                        print(pixel)
            else:
                print(f"No RGB data found in bounding box for joint '{joint_name}'.")
        else:
            print(f"Joint '{joint_name}' has an invalid or empty bounding box.")


def extract_ids(file_list, pattern):
    ids = []
    for file in file_list:
        match = re.search(pattern, file)
        if match:
            ids.append(int(match.group(1)))
    return np.array(ids), np.array(file_list)


def match_files(depth_files, joint_depth_files, joint_segmentation_files, segmentation_files):
    # Extract IDs and filenames as NumPy arrays
    depth_ids, depth_filenames = extract_ids(depth_files, r"(\d+)\.bmp$")
    joint_depth_ids, joint_depth_filenames = extract_ids(joint_depth_files, r"(\d+)\.bmp$")
    joint_segmentation_ids, joint_segmentation_filenames = extract_ids(joint_segmentation_files, r"(\d+)\.bmp$")
    rgb_ids, rgb_filenames = extract_ids(rgb_files, r"(\d+)\.bmp$")  # Adjust the regex as needed
    segmentation_ids, segmentation_filenames = extract_ids(segmentation_files, r"(\d+)\.bmp$")

    matched_files = {}
    for depth_id, depth_file in zip(depth_ids, depth_filenames):
        try:
            # Find matching joint depth file
            joint_match_idx = np.where(joint_depth_ids == depth_id)[0]
            if joint_match_idx.size == 0:
                raise FileMatchingException(f"No matching joint depth file found for depth ID {depth_id}")

            # Find matching joint segmentation file
            joint_seg_match_idx = np.where(joint_segmentation_ids == depth_id)[0]
            if joint_seg_match_idx.size == 0:
                raise FileMatchingException(f"No matching joint segmentation file found for depth ID {depth_id}")

            # Find matching RGB file
            rgb_match_idx = np.where(rgb_ids == depth_id)[0]
            if rgb_match_idx.size == 0:
                raise FileMatchingException(f"No matching RGB file found for depth ID {depth_id}")

            # Find matching segmentation file
            seg_match_idx = np.where(segmentation_ids == depth_id)[0]
            if seg_match_idx.size == 0:
                raise FileMatchingException(f"No matching segmentation file found for depth ID {depth_id}")

            # Add matched files to the dictionary
            joint_depth_file = joint_depth_filenames[joint_match_idx[0]]
            joint_seg_file = joint_segmentation_filenames[joint_seg_match_idx[0]]
            rgb_file = rgb_filenames[rgb_match_idx[0]]
            seg_file = segmentation_filenames[seg_match_idx[0]]

            matched_files[(depth_file, joint_depth_file, joint_seg_file, rgb_file)] = seg_file

        except FileMatchingException as e:
            print(e)
            # Optionally, handle the exception further (e.g., logging, skipping the current iteration)

    return matched_files


def calculate_2d_bbox(segmentation_map):
    # Check for non-black pixels (i.e., any color other than black)
    non_black_color = np.any(segmentation_map != (0, 0, 0), axis=-1)
    y_idx, x_idx = np.where(non_black_color)

    if len(x_idx) == 0 or len(y_idx) == 0:
        print("No non-black pixels found. No bounding box can be calculated.")
        return None

    x_min, x_max = np.min(x_idx), np.max(x_idx)
    y_min, y_max = np.min(y_idx), np.max(y_idx)

    x_size = x_max - x_min
    y_size = y_max - y_min

    bbox_info = f"Bounding Box: Top-Left Corner = ({x_min}, {y_min}), Width = {x_size}, Height = {y_size}"
    print(bbox_info)

    return x_min, y_min, x_size, y_size


def point_cloud_with_joints(main_point_cloud_with_color, joint_locations, joint_color=None):
    if joint_color is None:
        joint_color = [238, 240, 248]

    # Prepare containers for joint points and colors
    joint_points = np.empty((0, 3), dtype=np.float32)
    joint_colors = np.empty((0, 3), dtype=np.float32)

    for joint_name, loc in joint_locations.items():
        sphere_points, _ = create_sphere_points(loc, 0.02, 100, joint_color)
        # Ensure sphere_points is a 2D array
        if sphere_points.ndim == 1:
            sphere_points = np.expand_dims(sphere_points, axis=0)
        joint_points = np.vstack([joint_points, sphere_points])
        joint_colors = np.vstack([joint_colors, np.tile(joint_color, (len(sphere_points), 1))])

    # Combine main point cloud and joint points
    all_points = np.vstack([main_point_cloud_with_color[:, :3], joint_points])
    all_colors = np.vstack([main_point_cloud_with_color[:, 3:], joint_colors])

    combined_point_cloud_with_color = np.hstack((all_points, all_colors))
    return combined_point_cloud_with_color


def save_pointcloud_as_bin(point_cloud_with_color, base_save_path, current_index):
    # Check if the original point cloud has XYZ and RGB data
    if point_cloud_with_color.shape[1] != 6:
        raise ValueError("Point cloud data does not have the correct format. Expected XYZRGB.")

    # Check color data before saving
    if np.all(point_cloud_with_color[:, 3:6] < 0.01):
        raise ValueError("Color data in the point cloud is near zero.")

    # Construct the binary file path
    bin_filename = f"{current_index}.bin"
    bin_path = os.path.join(base_save_path, bin_filename)

    # Convert point cloud data to the correct dtype before saving
    point_cloud_with_color = point_cloud_with_color.astype(np.float32)

    # Save the point cloud data directly to a binary file
    point_cloud_with_color.tofile(bin_path)
    print(f"Saved pointcloud to {bin_path}")


def print_sample_points_in_bboxes(point_cloud, bounding_boxes, sample_size=5):
    points = np.asarray(point_cloud.points)
    for joint_name, bbox in bounding_boxes.items():
        bbox_min = np.asarray(bbox.get_min_bound())
        bbox_max = np.asarray(bbox.get_max_bound())

        in_bbox = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
        points_in_bbox = points[in_bbox]

        sample_points = points_in_bbox[:sample_size]
        print(f"Joint '{joint_name}': Sample points in the bounding box: {sample_points}")


class JSONLoadingException(Exception):
    pass


class ClassDeterminationException(Exception):
    pass


class JointColorsNotFoundException(Exception):
    pass


class JointCountMismatchException(Exception):
    pass


class InvalidSegmentationImageException(Exception):
    pass


class FileMatchingException(Exception):
    pass


class InvalidImageException(Exception):
    pass


class NoJointsDetectedException(Exception):
    pass


def load_joint_colors_from_json(json_path):
    try:
        with open(json_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        raise JSONLoadingException(f"Error loading JSON file: {e}")


def get_class_for_index(index, json_data):
    adjusted_index = index  # Adjust the index to match your data's indexing
    for class_name, class_info in json_data.items():
        if class_info['start_index'] <= adjusted_index <= class_info['end_index']:
            print(f"Current index: {adjusted_index}, Class: {class_name}, Object class: {class_info['class']}")
            return class_name, class_info['class']
    raise ClassDeterminationException(f"No class found for index {adjusted_index}")


def get_joint_info_for_class(class_name, json_data):
    if class_name in json_data:
        joint_info_dict = json_data[class_name]['Joints']
        joint_colors = {joint_name: tuple(info['RGB']) for joint_name, info in joint_info_dict.items()}
        joint_classes = {joint_name: info['Joint_class'] for joint_name, info in joint_info_dict.items()}
        return joint_colors, joint_classes
    raise JointColorsNotFoundException(f"No joint information found for class '{class_name}'")


def extract_index_from_filename(filename):
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract index from filename: {filename}")


def load_segmentation_image(image_path):
    # Load the image
    image = iio.v3.imread(image_path).astype(np.uint8)
    if image is None:
        raise IOError(f"Failed to load segmentation image at {image_path}")
    return image


# Function to create a point cloud for a joint sphere
def create_sphere_points(center, radius, points_per_sphere, color):
    # Normalize the color values if they are in the range of 0-255
    normalized_color = np.array(color) / 255.0

    # Generate theta and phi for uniform point distribution
    phi = np.random.uniform(0, 2 * np.pi, points_per_sphere)
    costheta = np.random.uniform(-1, 1, points_per_sphere)
    theta = np.arccos(costheta)

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Translate points to the given center
    points = np.vstack((x, y, z)).T + center

    # Assign the same color to all points
    colors = np.array([normalized_color for _ in range(points_per_sphere)])

    return points, colors


# Camera parameters
FX_DEPTH = 320
FY_DEPTH = 320
CX_DEPTH = 320
CY_DEPTH = 240

# Define the base paths for each type of image
base_path_depth = r""
base_path_joint_depth = r""
base_path_joint_segmentation = r"s"
base_path_rgb = r""
base_path_segmentation = r"s"
bin_save_directory = r""
json_data = load_joint_colors_from_json(r"")

# Gather lists of filenames
depth_files = os.listdir(base_path_depth)
joint_depth_files = os.listdir(base_path_joint_depth)
joint_segmentation_files = os.listdir(base_path_joint_segmentation)
rgb_files = os.listdir(base_path_rgb)
segmentation_files = os.listdir(base_path_segmentation)

# Match files
matched_files = match_files(depth_files, joint_depth_files, joint_segmentation_files, segmentation_files)
info_list = []

K = np.array([[FX_DEPTH, 0, CX_DEPTH],
              [0, FY_DEPTH, CY_DEPTH],
              [0, 0, 1]])
Rt = np.identity(4)

# Process each set of matched files
for (depth_file, joint_depth_file, joint_seg_file, rgb_file), seg_file in matched_files.items():
    # Load and process files
    file_index = extract_index_from_filename(depth_file)

    class_name, object_class = get_class_for_index(file_index, json_data)
    joint_colors, joint_classes = get_joint_info_for_class(class_name, json_data)
    depth_image_path = os.path.join(base_path_depth, depth_file)
    joint_depth_image_path = os.path.join(base_path_joint_depth, joint_depth_file)
    joint_segmentation_image_path = os.path.join(base_path_joint_segmentation, joint_seg_file)
    rgb_image_path = os.path.join(base_path_rgb, rgb_file)  # Assuming you have a base path for RGB images
    segmentation_image_path = os.path.join(base_path_segmentation, seg_file)

    # Load the images
    depth_image = load_depth_image(depth_image_path)
    joint_depth_image = load_depth_image(joint_depth_image_path)
    joint_segmentation_image = iio.v3.imread(joint_segmentation_image_path).astype(np.uint8)
    print(f"Joint segmentation image for index {file_index} loaded successfully.")

    rgb_image = cv2.imread(rgb_image_path)  # Load RGB image here
    segmentation_image = iio.v3.imread(segmentation_image_path).astype(np.uint8)
    joint_locations = find_and_count_valid_joints(joint_depth_image, joint_segmentation_image, joint_colors, FX_DEPTH,
                                                  FY_DEPTH, CX_DEPTH, CY_DEPTH)

    point_cloud_with_colour = create_point_cloud_from_depth(
        depth_image,
        rgb_image_path,
        FX_DEPTH, FY_DEPTH, CX_DEPTH, CY_DEPTH
    )

    bbox_2d = calculate_2d_bbox(segmentation_image)

    numpy_points = point_cloud_with_colour[:, :3]  # Extracting only the XYZ coordinates

    overall_bbox_data, individual_bboxes, individual_bbox_data = create_bounding_box_from_joints(joint_locations, numpy_points)
    # print_rgb_values_in_joint_bounding_boxes(joint_locations, rgb_image)

    # Visualize the point cloud and joints

    # combined_point_cloud = point_cloud_with_joints(point_cloud_with_colour, joint_locations)

    save_pointcloud_as_bin(point_cloud_with_colour, bin_save_directory, file_index)

    # Annotations data
    names = []
    locations = []
    dimensions = []
    rotation_y = []
    indices = []
    classes = []
    gt_boxes = []

    joint_index = 1  # Starting index for joints
    for joint_name, joint_location in joint_locations.items():
        if joint_name in individual_bbox_data:  # Check if the joint is in the dictionary
            joint_class = joint_classes[joint_name]
            bbox_data = individual_bbox_data[joint_name]
            centroid = bbox_data['centroid']
            dimension = bbox_data['dimensions']
            full_info = np.concatenate((centroid, dimension, np.array([0])))

            names.append(joint_name)
            locations.append(centroid)
            dimensions.append(dimension)
            rotation_y.append(0)
            indices.append(joint_index)
            classes.append(joint_class)
            gt_boxes.append(full_info)

            joint_index += 1


    # Add overall bounding box
    overall_centroid, overall_dimensions = overall_bbox_data
    overall_bbox_info = np.concatenate((overall_centroid, overall_dimensions, np.array([0])))


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_with_colour[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud_with_colour[:, 3:])
        # Prepare bounding boxes for visualization
    bbox_visuals = []
    for joint_name, bbox in individual_bboxes.items():
        bbox.color = [1, 0, 0]  # Set color to red
        bbox_visuals.append(bbox)

    # Prepare the overall bounding box for visualization
    overall_bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=overall_centroid - overall_dimensions / 2,
        max_bound=overall_centroid + overall_dimensions / 2
    )

    # Visualize everything together
    o3d.visualization.draw_geometries([pcd, overall_bbox, *bbox_visuals])




    names.append(class_name)
    locations.append(overall_centroid)
    dimensions.append(overall_dimensions)
    rotation_y.append(0)
    indices.append(joint_index)
    classes.append(object_class)
    gt_boxes.append(overall_bbox_info)

    # Convert lists to numpy arrays
    names = np.array(names)
    locations = np.vstack(locations)
    dimensions = np.vstack(dimensions)
    rotation_y = np.array(rotation_y)
    indices = np.array(indices)
    classes = np.array(classes)
    gt_boxes = np.vstack(gt_boxes)
    gt_num = len(names)

    # Prepare the final annotation dictionary
    annotations = {
        'gt_num': gt_num,
        'name': names,
        'bbox': np.array([bbox_2d]),  # Assuming bbox_2d is calculated elsewhere
        'location': locations,
        'dimensions': dimensions,
        'rotation_y': rotation_y,
        'index': indices,
        'class': classes,
        'gt_boxes_upright_depth': gt_boxes
    }

    # Add to info_list
    info = {
        'point_cloud': {'num_features': 23, 'lidar_idx': file_index},
        'pts_path': os.path.join("", f"{file_index}.bin"),
        'image': {
            'image_idx': file_index,
            'image_shape': np.array(depth_image.shape),
            'image_path': os.path.join("", f"{file_index}.bmp")
        },
        'calib': {'K': K, 'Rt': Rt},
        'annos': annotations
    }

    info_list.append(info)

# Serialize and save the info_list to a pickle file
pickle_filename = 'sunrgbd_infos_val.pkl'
with open(pickle_filename, 'wb') as file:
    pickle.dump(info_list, file, protocol=4)

print(f"Data saved to {pickle_filename}")

"""
    # Visualize the point cloud and joints
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_with_colour[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud_with_colour[:, 3:])

    # Prepare point clouds for each joint location
    for joint_name, loc in joint_locations.items():
        #joint_color = joint_colors[joint_name]  # RGB color from joint_colors dictionary
        joint_color = [238, 240, 248]  # Red color for joints in the range 0-255
        sphere_points = create_sphere_points(loc, 0.02, 100, joint_color)  # Adjust radius and points as needed
        pcd += sphere_points

    # Prepare bounding boxes for visualization
    bbox_visuals = []
    for joint_name, bbox in individual_bboxes.items():
        bbox_visuals.append(bbox)

    # Prepare the overall bounding box for visualization
    overall_bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=overall_centroid - overall_dimensions / 2,
        max_bound=overall_centroid + overall_dimensions / 2
    )

    # Visualize everything together
    o3d.visualization.draw_geometries([pcd, overall_bbox, *bbox_visuals])
"""  # visualisation
