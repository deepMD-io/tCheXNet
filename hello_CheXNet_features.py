from chexnet import get_chexnet_model
from chexnet import chexnet_preprocess_input
from chexnet import chexnet_class_index_to_name
from preprocess import get_image_numpy_array


def main():
    # get model
    base_model, model = get_chexnet_model()

    # prepare a list of images
    image_path_list = [
        'hello_world_images/view1_frontal.jpg',
    ]

    # convert the image into numpy array
    image_numpy_array = get_image_numpy_array(image_path_list)

    # preprocess images
    x_test = chexnet_preprocess_input(image_numpy_array)

    # obtain the feature
    yhat = base_model.predict(x_test)

    # for each image, print the feature
    for image_index in range(len(yhat)):
        print(image_path_list[image_index])
        print(len(yhat[image_index]), yhat[image_index])


if __name__ == '__main__':
    '''
    What this script is doing is that
    1. It first loads a chexnet nodel (DenseNet121),
       where the weight is from https://github.com/brucechou1983/CheXNet-Keras
    2. It then outputs the activation output of the layer before the output layer
    '''
    main()
