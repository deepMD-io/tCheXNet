from train_tCheXNet import get_model
from chexnet import chexnet_preprocess_input
from preprocess import get_image_numpy_array
import os
from test_CheXNet import target_classes


# For Mac users
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    save_dir = os.path.join(
        os.getcwd(),
        'saved_models'
    )

    model_name = '94482_23620_keras_cw_noDropOut_chexpert_pretrained_chexnet_512_1_001_0.54064.h5'
    model_path = os.path.join(save_dir, model_name)

    base_model, model = get_model()

    # load weights
    model.load_weights(model_path)

    # prepare a list of images
    image_path_list = [
        'hello_world_images/view1_frontal.jpg',
    ]

    # convert the image into numpy array
    image_numpy_array = get_image_numpy_array(image_path_list)

    # preprocess images
    x_test = chexnet_preprocess_input(image_numpy_array)

    # predict the probability across all output classes
    yhat = model.predict(x_test)

    # for each image, print the class name and the probability
    for image_index in range(len(yhat)):
        print(image_path_list[image_index])
        print('Pneumothorax', yhat[image_index][0])


if __name__ == '__main__':
    '''
    What this script is doing is that
    1. It first loads a tCheXNet nodel
    2. It then makes predictions on the images on the image_path_list
    3. The probability of prediction across Pneumothorax is computed
    '''
    main()
