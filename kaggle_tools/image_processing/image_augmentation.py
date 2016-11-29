from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)


def augment_img(input_file, output_folder, img_format='jpg',
                number_imgs=10):
    """
    Generate number_imgs new images from a given image.
    This function is inspired from the following blog post:
    https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    """
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    img = load_img(input_file)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=output_folder,
                              save_format=img_format):
        i += 1
        if i > number_imgs:
            break
