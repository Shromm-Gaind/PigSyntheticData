import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.interpolate import splprep, splev

def calculate_look_at_orientation(camera_point, target_point):
    # Calculate direction vector from camera to target
    direction = np.array(target_point) - np.array(camera_point)
    dx, dy, dz = direction

    # Calculate pitch1
    distance_horizontal = np.sqrt(dx ** 2 + dy ** 2)
    pitch = np.arctan2(-dz, distance_horizontal) * (180 / np.pi)

    # Calculate yaw
    yaw = np.arctan2(dy, dx) * (180 / np.pi)

    # No need to adjust yaw by target's yaw orientation in this approach
    # Ensure yaw is normalized to a 0 - 360 range if necessary
    yaw = yaw % 360
    if yaw > 180:  # Convert yaw to -180 to 180 range for easier interpretation
        yaw -= 360

    return pitch, yaw


def generate_hemisphere_points_with_orientation(center, start_radius, end_radius, step):
    # Parameters for point spacing
    degrees_per_point = 20
    degrees_per_circle = 20

    # Initialize a list to store points with their index, location, and orientation
    points_info = []
    point_index = 0

    for radius_cm in range(start_radius, end_radius + 1, step):
        num_points = int(360 / degrees_per_point)
        num_circles = int(90 / degrees_per_circle) + 1

        for i in range(num_circles):
            phi = np.pi / 2 * i / (num_circles - 1)

            if i == 0:  # Special case for the zenith
                x, y, z = center[0], center[1], center[2] + radius_cm
                pitch, yaw = calculate_look_at_orientation((x, y, z), center)
                points_info.append({
                    'index': point_index,
                    'location': (x, y, z),
                    'orientation': (pitch, yaw)
                })
                point_index += 1
                continue  # Skip the rest of the loop for the zenith case

            for j in range(num_points):
                theta = 2 * np.pi * j / num_points
                x = center[0] + radius_cm * np.sin(phi) * np.cos(theta)
                y = center[1] + radius_cm * np.sin(phi) * np.sin(theta)
                z = center[2] + radius_cm * np.cos(phi)

                pitch, yaw = calculate_look_at_orientation((x, y, z), center)
                points_info.append({
                    'index': point_index,
                    'location': (x, y, z),
                    'orientation': (pitch, yaw)
                })
                point_index += 1

    return points_info


# Hemisphere parameters
center = np.array([-510, -620, 70])  # Center of the hemisphere in cm
start_radius = 500  # Starting radius in cm
end_radius = 500  # Ending radius in cm
step = 1  # Step size in cm

points_info = generate_hemisphere_points_with_orientation(center, start_radius, end_radius, step)

# Plotting (optional)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for point in points_info:
    x, y, z = point['location']
    ax.scatter(x, y, z)
ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Z (cm)')
ax.set_title('Hemisphere with Indexed Points and Orientations')
plt.show()

# Print the first few points for verification (optional)
for point in points_info:
    print(f"Index: {point['index']}, Location: {point['location']}, Orientation: {point['orientation']}")

# Convert points_info to the specified JSON format
points_json = [{
    "Index": int(point['index']),
    "CameraLocation": {
        "X": float(point['location'][0]),
        "Y": float(point['location'][1]),
        "Z": float(point['location'][2])
    },
    "CameraRotation": {
        "P": -abs(float(point['orientation'][0])),
        "Y": float(point['orientation'][1]),
        "R": 0.0  # Roll is fixed as 0.0, already a Python float
    }
} for point in points_info]

# Define the JSON file name
json_file_name = r'E:\easytest.json'

# Write to JSON file
with open(json_file_name, 'w') as json_file:
    json.dump(points_json, json_file, indent=4)



# Extract x, y, z coordinates from points_info
x = np.array([point['location'][0] for point in points_info])
y = np.array([point['location'][1] for point in points_info])
z = np.array([point['location'][2] for point in points_info])

# Fit a 3D spline to the extracted coordinates. The s parameter controls the amount of smoothing.
tck, u = splprep([x, y, z], s=0.0)

# Evaluate the spline over a finer set of points to get a smooth curve
u_fine = np.linspace(0, 1, 300)
x_smooth, y_smooth, z_smooth = splev(u_fine, tck)

# Plot the original points and the smooth spline curve
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Recalculate the center coordinates in the plot's coordinate system
center_x, center_y, center_z = center

# Radius for the large circle to be displayed at the center
circle_radius = 50  # cm, adjust as needed

# Generating points for the circle in the XY plane at the center's Z height
circle_points = np.linspace(0, 2 * np.pi, 100)  # 100 points around the circle
circle_x = center_x + circle_radius * np.cos(circle_points)
circle_y = center_y + circle_radius * np.sin(circle_points)
circle_z = np.full_like(circle_x, center_z)


# Calculate vector components (u, v, w) pointing towards the center for each point
u = center[0] - x
v = center[1] - y
w = center[2] - z

# Normalize the vectors so they have a uniform length for visual clarity
norm = np.sqrt(u**2 + v**2 + w**2)
u_norm = u / norm * 20  # Scale to a fixed length for the arrows, adjust as needed
v_norm = v / norm * 20
w_norm = w / norm * 20

# Plot the original points, the smooth spline curve, and the large circle at the center
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Original points
ax.scatter(x, y, z, color='red', label='Camera Pose')

# Spline curve
ax.plot(x_smooth, y_smooth, z_smooth, color='blue', label='Spline Curve')

# Large circle at the center
ax.plot(circle_x, circle_y, circle_z, color='green', linewidth=2, label='Center Indicator')

# Vectors pointing towards the center
ax.quiver(x, y, z, u_norm, v_norm, w_norm, color='black', length=2, arrow_length_ratio=0.3, label='Orientation Vectors')


ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Z (cm)')
ax.set_title('Spline Generation for Easy Dataset')
ax.legend()

#plt.savefig(r'E:\Figure3.svg')

plt.show()
