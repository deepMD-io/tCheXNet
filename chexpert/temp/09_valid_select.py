import pandas as pd

# read the csv file
dataset_df = pd.read_csv("valid.csv")

# selecting only the following columns
new_df = dataset_df[
    ['Path',
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
]

#Frontal/Lateral
# select only the frontal
new_df = new_df.loc[
    new_df['Frontal/Lateral'] == 'Frontal'
]

# print the shape
print(new_df.shape)

new_df['Cardiomegaly'] = new_df['Cardiomegaly'].astype('int32')
new_df['Edema'] = new_df['Edema'].astype('int32')
new_df['Consolidation'] = new_df['Consolidation'].astype('int32')
new_df['Pneumonia'] = new_df['Pneumonia'].astype('int32')
new_df['Atelectasis'] = new_df['Atelectasis'].astype('int32')
new_df['Pneumothorax'] = new_df['Pneumothorax'].astype('int32')

# written to an output file
new_df.to_csv('valid_frontal_6_classes.csv', index=False)
