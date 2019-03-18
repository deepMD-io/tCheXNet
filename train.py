
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

def get_model():
    # get base model, model
    base_model, chexnet_model = get_chexnet_model()
    # print a model summary
    # print_summary(base_model)

    x = base_model.output
    # Dropout layer
    x = Dropout(0.2)(x)
    # one more layer (relu)
    x = Dense(512, activation='relu')(x)
    # Dropout layer
    x = Dropout(0.2)(x)
    # add a logistic layer -- let's say we have 6 classes
    predictions = Dense(
        6,
        activation='sigmoid',
        name='predictions')(x)

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
    batch_size = 128
    epochs = 100

    save_dir = os.path.join(
        os.getcwd(),
        'saved_models'
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    filepath = "saved_models/94482_23620_keras_chexpert_pretrained_chexnet_512_6_{epoch:02d}_{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=False,
        mode='max'
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
    train_df = pd.read_csv(csv_file_path)

    train_generator = AugmentedImageSequence(
                        dataset_csv_file=csv_file_path,
                        class_names=target_classes,
                        source_image_dir='./chexpert/',
                        batch_size=batch_size)

    csv_file_path = 'chexpert/train_23620_frontal_6_classes_real_no_zeros_preprocessed.csv'
    valid_df = pd.read_csv(csv_file_path)

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
                        use_multiprocessing=True)


    # Save model and weights
    #if not os.path.isdir(save_dir):
    #    os.makedirs(save_dir)
    #model_path = os.path.join(save_dir, new_model_name)
    #model.save(model_path)
    #print('Saved trained model at %s ' % model_path)



if __name__ == '__main__':
    main()
