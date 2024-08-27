from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def unpack_depth(packed_image_path):

    img = Image.open(packed_image_path)
    img = img.convert('RGBA')

    # Convert the image to a numpy array
    img_data = np.array(img)

    # Split into channels
    R = img_data[:, :, 0].astype(np.float32)
    G = img_data[:, :, 1].astype(np.float32)
    B = img_data[:, :, 2].astype(np.float32)
    A = img_data[:, :, 3].astype(np.float32)

    # Reconstruct the depth value
    depth = (R + G / 256.0 + B / (256.0 * 256.0) + A / (256.0 * 256.0 * 256.0))
    depth *= (256.0 * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)

    return depth

# Put the path to the Depth Image.
depth_array = unpack_depth("")


# Visualize the depth
plt.imshow(depth_array, cmap='gray')
plt.colorbar()
plt.show()
