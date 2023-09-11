# %% imports
from defect_level_classification_object import *

# %% training&test and export a model
defect_classifier = defect_classifier()

# define the training path, the path to the training data
defect_classifier.training_path = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\training_data\80k\2022-09-15-09-00-01_advanced example - multi_level_L_datasetID_0.csv"

# train the model
defect_classifier.train_export_model(use_PCA=True)

# %% testing
# define the object
defect_classifier = defect_classifier()

# define the scaler and maechine learnign model path
defect_classifier.trained_model_path = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\model_defect_classification.joblib"
defect_classifier.trained_scaler_path = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\scaler_for_defect_classification.joblib"
defect_classifier.test_data_path = r"D:\study\thesis_data_storage\journal\defect_classification\basic_model\testing_data\outputs\2023-07-30-18-52-48_advanced example - multi_level_L_datasetID_0.csv"

# testing
defect_classifier.test_model()

# %% testing the trained model varying Et1

# define the object
defect_classifier = defect_classifier()

# define the scaler and maechine learnign model path
defect_classifier.trained_model_path = r'G:\study\thesis_data_storage\journal\defect_classification\basic_model\model_defect_classification.joblib'
print(os.path.exists(defect_classifier.trained_model_path))
defect_classifier.trained_scaler_path = r'G:\study\thesis_data_storage\journal\defect_classification\basic_model\scaler_for_defect_classification.joblib'

# define a list of path to test csv

# vary Et1
test_path_list = [
    r'G:\study\thesis_data_storage\journal\defect_classification\Et1_vs_acc\test_data\2023-01-20-21-03-20_advanced example - multi_level_L_datasetID_0.csv',
    r'G:\study\thesis_data_storage\journal\defect_classification\Et1_vs_acc\test_data\2023-01-20-21-08-57_advanced example - multi_level_L_datasetID_0.csv',
    r'G:\study\thesis_data_storage\journal\defect_classification\Et1_vs_acc\test_data\2023-01-20-21-14-35_advanced example - multi_level_L_datasetID_0.csv',
    r'G:\study\thesis_data_storage\journal\defect_classification\Et1_vs_acc\test_data\2023-01-20-21-20-20_advanced example - multi_level_L_datasetID_0.csv',
    r'G:\study\thesis_data_storage\journal\defect_classification\Et1_vs_acc\test_data\2023-01-20-21-25-50_advanced example - multi_level_L_datasetID_0.csv',
    r'G:\study\thesis_data_storage\journal\defect_classification\Et1_vs_acc\test_data\2023-01-20-21-31-38_advanced example - multi_level_L_datasetID_0.csv',
    r'G:\study\thesis_data_storage\journal\defect_classification\Et1_vs_acc\test_data\2023-01-20-21-37-35_advanced example - multi_level_L_datasetID_0.csv',
    r'G:\study\thesis_data_storage\journal\defect_classification\Et1_vs_acc\test_data\2023-01-20-21-43-28_advanced example - multi_level_L_datasetID_0.csv',
    r'G:\study\thesis_data_storage\journal\defect_classification\Et1_vs_acc\test_data\2023-01-20-21-49-03_advanced example - multi_level_L_datasetID_0.csv',
    r'G:\study\thesis_data_storage\journal\defect_classification\Et1_vs_acc\test_data\2023-01-20-21-54-37_advanced example - multi_level_L_datasetID_0.csv',
    r'G:\study\thesis_data_storage\journal\defect_classification\Et1_vs_acc\test_data\2023-01-21-11-30-17_advanced example - multi_level_L_datasetID_0.csv'
]

# vary Et2
# test_path_list = [
#     r'G:\study\thesis_data_storage\journal\defect_classification\Et2_vs_acc\test_data\2023-01-22-13-42-10_advanced example - multi_level_L_datasetID_0.csv',
#     r'G:\study\thesis_data_storage\journal\defect_classification\Et2_vs_acc\test_data\2023-01-22-13-47-50_advanced example - multi_level_L_datasetID_0.csv',
#     r'G:\study\thesis_data_storage\journal\defect_classification\Et2_vs_acc\test_data\2023-01-22-13-53-35_advanced example - multi_level_L_datasetID_0.csv',
#     r'G:\study\thesis_data_storage\journal\defect_classification\Et2_vs_acc\test_data\2023-01-22-13-59-21_advanced example - multi_level_L_datasetID_0.csv',
#     r'G:\study\thesis_data_storage\journal\defect_classification\Et2_vs_acc\test_data\2023-01-22-14-04-58_advanced example - multi_level_L_datasetID_0.csv',
#     r'G:\study\thesis_data_storage\journal\defect_classification\Et2_vs_acc\test_data\2023-01-22-14-10-24_advanced example - multi_level_L_datasetID_0.csv',
#     r'G:\study\thesis_data_storage\journal\defect_classification\Et2_vs_acc\test_data\2023-01-22-14-15-55_advanced example - multi_level_L_datasetID_0.csv',
#     r'G:\study\thesis_data_storage\journal\defect_classification\Et2_vs_acc\test_data\2023-01-22-14-21-01_advanced example - multi_level_L_datasetID_0.csv',
#     r'G:\study\thesis_data_storage\journal\defect_classification\Et2_vs_acc\test_data\2023-01-22-14-26-40_advanced example - multi_level_L_datasetID_0.csv',
#     r'G:\study\thesis_data_storage\journal\defect_classification\Et2_vs_acc\test_data\2023-01-22-14-32-15_advanced example - multi_level_L_datasetID_0.csv',
#     r'G:\study\thesis_data_storage\journal\defect_classification\Et2_vs_acc\test_data\2023-01-22-14-37-42_advanced example - multi_level_L_datasetID_0.csv'
# ]


Et_range = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
counter = 0
for test_path in test_path_list:
    print(Et_range[counter])
    counter = counter + 1
    # define the test path
    defect_classifier.test_data_path = test_path
    # test the model
    defect_classifier.test_model()

# %% Using gridsearch to train&test and exprot the model
defect_classifier = defect_classifier()

# define the training path
defect_classifier.training_path = r"G:\study\thesis_data_storage\journal\defect_classification\basic_model\training_data\2022-09-15-09-00-01_advanced example - multi_level_L_datasetID_0.csv"

# train the model
defect_classifier.train_Gridsearch()

# email reminding
defect_classifier.email_reminder()

# %% apply transfer learning with bert pre-trained model
# %% Using gridsearch to train&test and exprot the model
defect_classifier = defect_classifier()

# define the training path
defect_classifier.training_path = r"G:\study\thesis_data_storage\journal\defect_classification\basic_model\training_data\2022-09-15-09-00-01_advanced example - multi_level_L_datasetID_0.csv"

# train the model
defect_classifier.transfer_learning_VGG16()