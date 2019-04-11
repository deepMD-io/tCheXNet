from train import get_model
from chexnet import chexnet_preprocess_input
from preprocess import load_data_from_csv
import os
import pandas as pd
from main import target_classes
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import average_precision_score


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

    # take a look
    model.summary()

    # load data from csv
    csv_file_path = 'chexpert/valid_frontal_6_classes.csv'
    valid_df = pd.read_csv(csv_file_path)
    x_test, y_test = load_data_from_csv(csv_file_path)

    # preprocess images
    x_test = chexnet_preprocess_input(x_test)

    # predict the probability across all output classes
    yhat = model.predict(x_test)

    # evaluation
    for i, target_class in enumerate(target_classes):
        y_true = valid_df[target_class]
        y_scores = yhat[:, i]
        # print out y_scores
        for y_score in y_scores:
            print(y_score)
        roc_score = roc_auc_score(y_true, y_scores)
        #prc_score = average_precision_score(y_true, y_scores)
        #print(target_class, roc_score, prc_score)
        print(target_class, roc_score)


if __name__ == '__main__':
    '''
    What this script is doing is that
    1. It first loads a tCheXNet nodel
    2. It then made predictions on the Chexpert validation dataset (frontal only)
       The validation data contains 202 frontal chest x-ray images,
       from 200 unique patients.
       It is the testing dataset for this study
    3. The performance was then evaluated on one of the common classes, which is
       'Pneumothorax'. Area under ROC curve is computed.
    '''
    main()
