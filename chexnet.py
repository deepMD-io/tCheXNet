from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.utils import print_summary

# chexNet weights
# https://github.com/brucechou1983/CheXNet-Keras
chexnet_weights = 'chexnet/best_weights.h5'

# chexnet class names
chexnet_class_index_to_name = [
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
]

# chexnet class indexes
chexnet_class_name_to_index = {
    'Atelectasis': 0,
    'Cardiomegaly': 1,
    'Effusion': 2,
    'Infiltration': 3,
    'Mass': 4,
    'Nodule': 5,
    'Pneumonia': 6,
    'Pneumothorax': 7,
    'Consolidation': 8,
    'Edema': 9,
    'Emphysema': 10,
    'Fibrosis': 11,
    'Pleural_Thickening': 12,
    'Hernia': 13,
}


def chexnet_preprocess_input(value):
    return preprocess_input(value)


def get_chexnet_model():
    input_shape = (224, 224, 3)
    img_input = Input(shape=input_shape)
    base_weights = 'imagenet'

    # create the base pre-trained model
    base_model = DenseNet121(
        include_top=False,
        input_tensor=img_input,
        input_shape=input_shape,
        weights=base_weights,
        pooling='avg'
    )

    x = base_model.output
    # add a logistic layer -- let's say we have 14 classes
    predictions = Dense(
        14,
        activation='sigmoid',
        name='predictions')(x)

    # this is the model we will use
    model = Model(
        inputs=img_input,
        outputs=predictions,
    )

    # load chexnet weights
    model.load_weights(chexnet_weights)

    # return model
    return base_model, model


if __name__ == '__main__':
    '''
    What this script is doing is that
    1. It first loads a chexnet nodel (DenseNet121),
       where the weight is from https://github.com/brucechou1983/CheXNet-Keras
    2. It then print a summary of the model architecture
    '''
    # get model
    base_model, model = get_chexnet_model()
    # print a model summary
    print_summary(model)
