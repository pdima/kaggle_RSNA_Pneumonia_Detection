import pandas as pd
from sklearn.model_selection import StratifiedKFold

nb_folds = 4

training_samples = pd.read_csv('../input/stage_1_detailed_class_info.csv')
training_samples = training_samples.drop_duplicates().reset_index(drop=True)
X = training_samples['patientId']
y = training_samples['class']
training_samples['fold'] = -1

skf = StratifiedKFold(n_splits=nb_folds, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    training_samples.loc[test_index, 'fold'] = fold

training_samples.to_csv('../input/folds.csv', index=False)

for cls in training_samples['class'].unique():
    print(cls)
    cls_samples = training_samples[training_samples['class'] == cls].reset_index(drop=True)
    for fold in range(nb_folds):
        print(fold, len(cls_samples[cls_samples.fold == fold]))
