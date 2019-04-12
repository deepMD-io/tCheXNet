from chexnet import get_chexnet_model
from chexnet import chexnet_preprocess_input
from chexnet import chexnet_class_index_to_name
from preprocess import get_image_numpy_array


def main():
    # get model
    base_model, model = get_chexnet_model()

    # prepare a list of images
    image_path_list = [
        'hello_world_images/patient64541_view1_frontal.jpg',
    ]

    # convert the image into numpy array
    image_numpy_array = get_image_numpy_array(image_path_list)

    # preprocess images
    x_test = chexnet_preprocess_input(image_numpy_array)

    # predict the probability across all output classes
    yhat = model.predict(x_test)

    # for each image, print the class names and the probability
    for image_index in range(len(yhat)):
        print(image_path_list[image_index])
        for i in range(len(yhat[image_index])):
            print(chexnet_class_index_to_name[i], yhat[image_index][i])


if __name__ == '__main__':
    '''
    What this script is doing is that
    1. It first loads a chexnet nodel (DenseNet121),
       where the weight is from https://github.com/brucechou1983/CheXNet-Keras
    2. It then made predictions on the images on the image_path_list
    3. The probability of prediction across all 14 classes is computed
        'Atelectasis',  # 0
        'Cardiomegaly',  # 1
        'Effusion',  # 2
        'Infiltration',  # 3
        'Mass',  # 4
        'Nodule',  # 5
        'Pneumonia',  # 6
        'Pneumothorax',  # 7
        'Consolidation',  # 8
        'Edema',  # 9
        'Emphysema',  # 10
        'Fibrosis',  # 11
        'Pleural_Thickening',  # 12
        'Hernia',  # 13
    '''
    main()
