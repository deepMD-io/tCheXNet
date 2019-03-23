from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_y_score_csv_file(csv_file_path):
    output = []
    with open (csv_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            line = float(line)
            #print(line)
            output.append(line)
    return output

csv_file_path = 'chexnet/chexnet_chexpert_valid_frontal_6_classes_Pneumothorax_prediction_score.csv'
y_score_chexnet = read_y_score_csv_file(csv_file_path)
y_score_chexnet = np.asarray(y_score_chexnet)

csv_file_path = 'chexpert/tchexnet_chexpert_valid_frontal_6_classes_Pneumothorax_prediction_score.csv'
y_score_tchexnet = read_y_score_csv_file(csv_file_path)
y_score_tchexnet = np.asarray(y_score_tchexnet)

csv_file_path = 'chexpert/valid_frontal_6_classes.csv'
dataset_df = pd.read_csv(csv_file_path)
y_true = dataset_df['Pneumothorax']

# Compute ROC curve and ROC area for each classifier
fpr = dict()
tpr = dict()
roc_auc = dict()

# ROC for CheXNet
fpr[0], tpr[0], _ = roc_curve(y_true, y_score_chexnet)
roc_auc[0] = auc(fpr[0], tpr[0])

# ROC for tCheXNet
fpr[1], tpr[1], _ = roc_curve(y_true, y_score_tchexnet)
roc_auc[1] = auc(fpr[1], tpr[1])

plt.figure()
lw = 1
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='CheXNet ROC curve (area = %0.3f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], color='darkblue',
         lw=lw, label='tCheXNet ROC curve (area = %0.3f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()
