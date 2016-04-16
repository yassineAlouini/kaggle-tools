import cv2

IMAGE_DIM = (100, 100)


def process_image(img_file):
    """
    Load and resize a color image file
    """
    img = cv2.imread(img_file)
    img = cv2.resize(img, IMAGE_DIM).transpose(
        (2, 0, 1)).astype('float32') / 255.0
    return img
