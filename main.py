from chexnet import get_chexnet_model
from chexnet import chexnet_preprocess_input
from chexnet import chexnet_class_name_to_index
from preprocess import get_image_numpy_array
from preprocess import load_data_from_csv
from sklearn.metrics import roc_auc_score
import pandas as pd

target_classes = [
    #'Cardiomegaly',
    #'Edema',
    #'Consolidation',
    #'Pneumonia',
    #'Atelectasis',
    'Pneumothorax'
]

def main():
    # load data from csv
    csv_file_path = 'chexpert/valid_frontal_6_classes.csv'
    dataset_df = pd.read_csv(csv_file_path)
    x_test, y_test = load_data_from_csv(csv_file_path)

    # get model
    base_model, model = get_chexnet_model()

    # preprocess images
    x_test = chexnet_preprocess_input(x_test)

    # predict the probability across all output classes
    yhat = model.predict(x_test)

    # evaluation
    for target_class in target_classes:
        y_true = dataset_df[target_class]
        y_scores = yhat[:, chexnet_class_name_to_index[target_class]]
        score = roc_auc_score(y_true, y_scores)
        print(target_class, score)


if __name__ == '__main__':
    '''
    What this script is doing is that
    1. It first loads a chexnet nodel (DenseNet121),
       where the weight is from https://github.com/brucechou1983/CheXNet-Keras
    2. It then made predictions on the Chexpert validation dataset (frontal only)
       The validation data contains 202 frontal chest x-ray images,
       from 200 unique patients.
    3. The performance was then evaluated on the 6 common classes, which are
       'Atelectasis',
       'Cardiomegaly',
       'Consolidation',
       'Edema',
       'Pneumonia',
       'Pneumothorax',
    '''
    main()
