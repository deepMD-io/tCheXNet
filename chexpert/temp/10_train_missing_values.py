import pandas as pd

# read the csv file
dataset_df = pd.read_csv("train_valid_frontal_6_classes_real_no_zeros.csv")

# replace "-1" by "0"
dataset_df = dataset_df.replace(-1, 0)

#fill the nan values with 0
dataset_df = dataset_df.fillna(value = 0)

#select the first 1000
#dataset_df = dataset_df.head(1001)

dataset_df.columns=[
    '',
    'Path',
    'Sex',
    'Age',
    'Frontal/Lateral',
    'AP/PA',
    'Cardiomegaly',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',]

dataset_df['Cardiomegaly'] = dataset_df['Cardiomegaly'].astype('int32')
dataset_df['Edema'] = dataset_df['Edema'].astype('int32')
dataset_df['Consolidation'] = dataset_df['Consolidation'].astype('int32')
dataset_df['Pneumonia'] = dataset_df['Pneumonia'].astype('int32')
dataset_df['Atelectasis'] = dataset_df['Atelectasis'].astype('int32')
dataset_df['Pneumothorax'] = dataset_df['Pneumothorax'].astype('int32')

#output, without writting out row numbers
dataset_df.to_csv('train_valid_frontal_6_classes_real_no_zeros_preprocessed.csv', index=False)
