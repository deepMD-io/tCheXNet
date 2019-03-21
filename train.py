
from chexnet import get_chexnet_model
from keras.layers import Input, Dense, Dropout
from keras.utils import print_summary
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import os
import pandas as pd
from generator import AugmentedImageSequence
from main import target_classes
from weights import get_class_weights

def get_class_weight(csv_file_path, target_class):
    #total_counts - int
    #class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    #multiply - int, positve weighting multiply
    '''
    df = pd.read_csv(csv_file_path)
    total_counts = df.shape[0]
    class_positive_counts = {}
    multiply = 1

    for target_class in target_classes:
        class_positive_counts[target_class] = df.loc[(df[target_class] == 1)].shape[0]

    class_weight = get_class_weights(
        total_counts,
        class_positive_counts,
        multiply
    )
    '''

    df = pd.read_csv(csv_file_path)
    total_counts = df.shape[0]
    class_weight = []
    for target_class in target_classes:
        weight_dict = {}
        weight_dict[0] = df.loc[(df[target_class] == 0)].shape[0] / total_counts
        weight_dict[1] = df.loc[(df[target_class] == 1)].shape[0] / total_counts
        class_weight.append(weight_dict)

    print(class_weight)
    return class_weight


def get_model():
    # get base model, model
    base_model, chexnet_model = get_chexnet_model()
    # print a model summary
    # print_summary(base_model)

    x = base_model.output
    # Dropout layer
    #x = Dropout(0.2)(x)
    # one more layer (relu)
    x = Dense(512, activation='relu')(x)
    # Dropout layer
    #x = Dropout(0.2)(x)
    #x = Dense(256, activation='relu')(x)
    # Dropout layer
    #x = Dropout(0.2)(x)
    # add a logistic layer -- let's say we have 6 classes
    predictions = Dense(
        1,
        activation='sigmoid')(x)

    # this is the model we will use
    model = Model(
        inputs=base_model.input,
    	outputs=predictions,
    )

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all base_model layers
    for layer in base_model.layers:
        layer.trainable = False

    # initiate an Adam optimizer
    opt = Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.0,
        amsgrad=False
    )

    # Let's train the model using Adam
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    return base_model, model

def main():
    batch_size = 16
    epochs = 50

    save_dir = os.path.join(
        os.getcwd(),
        'saved_models'
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    filepath = "saved_models/94482_23620_keras_cw_noDropOut_chexpert_pretrained_chexnet_512_1_{epoch:03d}_{val_loss:.5f}.h5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        mode='min'
    )
    callbacks_list = [checkpoint]

    #new_model_name = '94482_23620_keras_chexpert_pretrained_chexnet_512_6_epochs_1.h5'

    base_model, model = get_model()

    # load old weights
    #old_model_name = 'keras_chexpert_pretrained_chexnet_512_6_epochs_10.h5'
    #model_path = os.path.join(save_dir, old_model_name)
    #model.load_weights(model_path)

    # print a model summary
    #print_summary(model)

    csv_file_path = 'chexpert/train_94482_frontal_6_classes_real_no_zeros_preprocessed.csv'
    #train_df = pd.read_csv(csv_file_path)

    class_weight = get_class_weight(
                    csv_file_path,
                    target_classes)

    train_generator = AugmentedImageSequence(
                        dataset_csv_file=csv_file_path,
                        class_names=target_classes,
                        source_image_dir='./chexpert/',
                        batch_size=batch_size)

    csv_file_path = 'chexpert/train_23620_frontal_6_classes_real_no_zeros_preprocessed.csv'
    #valid_df = pd.read_csv(csv_file_path)

    valid_generator = AugmentedImageSequence(
                        dataset_csv_file=csv_file_path,
                        class_names=target_classes,
                        source_image_dir='./chexpert/',
                        batch_size=batch_size)

    STEP_SIZE_TRAIN=train_generator.steps
    STEP_SIZE_VALID=valid_generator.steps

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        class_weight=class_weight,
                        use_multiprocessing=True)

    # Save model and weights
    #if not os.path.isdir(save_dir):
    #    os.makedirs(save_dir)
    #model_path = os.path.join(save_dir, new_model_name)
    #model.save(model_path)
    #print('Saved trained model at %s ' % model_path)



if __name__ == '__main__':
    main()
