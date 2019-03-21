from train import get_model
from chexnet import chexnet_preprocess_input
from preprocess import load_data_from_csv
import os
import pandas as pd
from main import target_classes
from sklearn.metrics import roc_auc_score

def main():
    save_dir = os.path.join(
        os.getcwd(),
        'saved_models'
    )

    model_name = '94482_23620_keras_cw_noDropOut_chexpert_pretrained_chexnet_512_1_003_0.59832.h5'
    model_path = os.path.join(save_dir, model_name)

    base_model, model = get_model()

    # load weights
    model.load_weights(model_path)

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
        score = roc_auc_score(y_true, y_scores)
        print(target_class, score)

if __name__ == '__main__':
    main()
