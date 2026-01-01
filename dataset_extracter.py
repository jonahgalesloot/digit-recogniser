import os
import struct
import numpy as np

def read_mnist_images(file_path):
    """
    Reads MNIST images from an idx3-ubyte file.
    Returns a numpy array of shape (num_images, rows, cols).
    """
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError("Magic number mismatch in image file: expected 2051, got {}".format(magic))
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        return images

def read_mnist_labels(file_path):
    """
    Reads MNIST labels from an idx1-ubyte file.
    Returns a numpy array of shape (num_labels,).
    """
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError("Magic number mismatch in label file: expected 2049, got {}".format(magic))
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels

def binarize_image(image, threshold=100):
    """Convert grayscale image to binary (0 or 1) using the given threshold."""
    return (image > threshold).astype(int)

dataset_dir = "datasets/train"
images_file = os.path.join(dataset_dir, "MNIST/images.idx3-ubyte")
labels_file = os.path.join(dataset_dir, "MNIST/labels.idx1-ubyte")

images = read_mnist_images(images_file)
labels = read_mnist_labels(labels_file)
num_images = images.shape[0]
print("Number of images found:", num_images)

output_base_dir = os.path.join(dataset_dir, "extracted")
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

label_counters = {str(digit): 0 for digit in range(10)}

for i in range(num_images):
    label_str = str(labels[i])
    label_dir = os.path.join(output_base_dir, label_str)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    label_counters[label_str] += 1
    count = label_counters[label_str]
    txt_path = os.path.join(label_dir, f"image_{count}.txt")
    
    img_array = binarize_image(images[i])
    
    with open(txt_path, "w") as f:
        for row in img_array:
            row_str = " ".join(map(str, row.tolist()))
            f.write(row_str + "\n")
    
    print(f"Extracted image {i+1} to '{label_dir}' as image_{count}.txt")

print(f"Extraction complete. Check the '{output_base_dir}' directory for the output.")
