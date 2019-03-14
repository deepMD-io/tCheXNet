from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np

# given a csv file
# return a panda dataframe
# path -> image path
def load_image_info(csv_file_path):
    # read the csv file
    dataset_df = pd.read_csv(csv_file_path)
    # return the file
    return dataset_df

# given a image path list
# return a numpy array vertically stacking the images
def get_image_numpy_array(image_path_list):
    #A list of image paths
    #image_path_list = [
    #    'images/elephant.jpg',
    #    'images/2017-Honda-Civic-sedan-front-three-quarter.jpg',
    #]

    #A list of images
    image_list = []

    # load a file from each path
    for image_path in image_path_list:
        # PIL image
        img = load_img(image_path, target_size=(224, 224))
        # convert the image pixels to a numpy array
        img = img_to_array(img)
        # reshape data for the model
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        # add into image list
        image_list.append(img)

    # convert into a numpy by vertically stack the list
    x = np.vstack(image_list)

    # return the numpy array
    return x

if __name__ == '__main__':
    csv_file_path = 'chexpert/valid_frontal_6_classes.csv'
    dataset_df = load_image_info(csv_file_path)

    image_path_list = 'chexpert/' + dataset_df['Path']
    images = get_image_numpy_array(image_path_list)

    print(images.shape)
