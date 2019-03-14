from chexnet import get_chexnet_model
from chexnet import chexnet_preprocess_input
from chexnet import chexnet_class_name_to_index
from preprocess import load_image_info
from preprocess import get_image_numpy_array
from sklearn.metrics import roc_auc_score

target_classes = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Pneumonia',
    'Pneumothorax',
]

def main():
    # load csv file
    csv_file_path = 'chexpert/valid_frontal_6_classes.csv'
    dataset_df = load_image_info(csv_file_path)

    # load images
    image_path_list = 'chexpert/' + dataset_df['Path']
    images = get_image_numpy_array(image_path_list)

    # get model
    model = get_chexnet_model()

    # preprocess images
    x = chexnet_preprocess_input(images)

    # predict the probability across all output classes
    yhat = model.predict(x)

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
    3. The performance was then evaluated on the 6 common classes, which are
       'Atelectasis',
       'Cardiomegaly',
       'Consolidation',
       'Edema',
       'Pneumonia',
       'Pneumothorax',
    '''
    main()
