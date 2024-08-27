import numpy as np


def view_npy_as_text(file_path):
    data = np.load(file_path)

    print(data)


def save_npy_as_text(file_path, output_path):
    data = np.load(file_path)

    # Save the contents to a text file
    np.savetxt(output_path, data, fmt='%0.8f')


if __name__ == "__main__":
    # Replace with your .npy file path
    npy_file_path = 'unpacked_depth.npy'

    view_npy_as_text(npy_file_path)

    #save the contents to a text file
    text_output_path = 'output_DepthText_file.txt'
    save_npy_as_text(npy_file_path, text_output_path)
