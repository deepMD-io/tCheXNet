import pandas as pd

# read the csv file
dataset_df = pd.read_csv("train.csv")

# the csv file contains the following columns
# Path
# Sex
# Age
# Frontal/Lateral
# AP/PA
# No Finding 0
# Enlarged Cardiomediastinum 1
# Cardiomegaly 2
# Lung Opacity 3
# Lung Lesion 4
# Edema 5
# Consolidation 6
# Pneumonia 7
# Atelectasis 8
# Pneumothorax 9
# Pleural Effusion 10
# Pleural Other 11
# Fracture 12
# Support Devices 13

# first record
# new_df = dataset_df.loc[0]
# print the record
#print(new_df)

# two records
# new_df = dataset_df.loc[[0,1]]
# print the shape
#print(new_df.shape)

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

# select only the Cardiomegaly with label not equal to 0
new_df = new_df.loc[
    #(new_df['Cardiomegaly'].notnull())
    #&
    (new_df['Cardiomegaly'] != 0)
]

# select only the Edema with label not equal to 0
new_df = new_df.loc[
    #(new_df['Edema'].notnull())
    #&
    (new_df['Edema'] != 0)
]

# select only the Consolidation with label not equal to 0
new_df = new_df.loc[
    #(new_df['Consolidation'].notnull())
    #&
    (new_df['Consolidation'] != 0)
]

# select only the Pneumonia with label not equal to 0
new_df = new_df.loc[
    #(new_df['Pneumonia'].notnull())
    #&
    (new_df['Pneumonia'] != 0)
]

# select only the Atelectasis with label not equal to 0
new_df = new_df.loc[
    #(new_df['Atelectasis'].notnull())
    #&
    (new_df['Atelectasis'] != 0)
]

# select only the Pneumothorax with label not equal to 0
new_df = new_df.loc[
    #(new_df['Pneumothorax'].notnull())
    #&
    (new_df['Pneumothorax'] != 0)
]

'''
# select only the records with at least one entry
new_df = new_df.loc[
    (new_df['Cardiomegaly'].notnull())
    |
    (new_df['Edema'].notnull())
    |
    (new_df['Consolidation'].notnull())
    |
    (new_df['Pneumonia'].notnull())
    |
    (new_df['Atelectasis'].notnull())
    |
    (new_df['Pneumothorax'].notnull())
]
'''

# print the shape
print(new_df.shape)

# written to an output file
new_df.to_csv('train_valid_frontal_6_classes_real_no_zeros.csv')
