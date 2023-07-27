
# %% imports
from multi_classification_object import *

# define the object
multi_classification = multi_classifier()
# define the training path
# four sets
multi_classification.set11_path = r"D:\study\thesis_data_storage\unordered\set11\p\outputs\set11_80k.csv"
multi_classification.set10_path = r"D:\study\thesis_data_storage\unordered\set10\p\80k\2022_10_25\2022-10-25-13-57-56_advanced example - multi_level_L_datasetID_0.csv"
multi_classification.set01_path = r"D:\study\thesis_data_storage\unordered\set01\p\outputs\outputs\set01_80k.csv"
multi_classification.set00_path = r"D:\study\thesis_data_storage\unordered\set00\p\set00_80k\outputs\set00_80k.csv"
# combined dataset
multi_classification.combined_path = r"D:\study\thesis_data_storage\unordered\classification\set_classification\80k"
multi_classification.combined_path_file = r"D:\study\thesis_data_storage\unordered\classification\set_classification\80k\set_classification.csv"
# %% combine the data
multi_classification.data_combiner()

# %% train test the model
# load the combined data
multi_classification.combined_path_file = r'G:\study\thesis_data_storage\unordered\classification\set_classification\80k\set_classification.csv'
multi_classification.load_combined_data()
# train the model
multi_classification.train_test_model()

# %% GridSearch
# load the combined data
multi_classification.load_combined_data()
# Gridsearch to optimize the model
multi_classification.train_Gridsearch()

# %% Test the model
# load the model
trained_model_path = r"D:\study\thesis_data_storage\journal\set_classification\set_classification.joblib"
trained_scaler_path = r"D:\study\thesis_data_storage\journal\set_classification\set_classficiation_scaler.joblib"
trained_model = joblib.load(trained_model_path)
scaler = joblib.load(trained_scaler_path)
# load the test set
test_data = pd.read_csv(r"D:\study\thesis_data_storage\journal\set_classification\test_set\outputs\2023-07-07-11-09-19_advanced example - multi_level_L_datasetID_0.csv")
# go through the data pre-processing
X, y, param = multi_classification.pre_processor(test_data, return_param=True)
# scale the x data
X_scaled = scaler.transform(X)
# make hte prediction
y_pred = trained_model.predict(X_scaled)
# report the output
print(classification_report(y, y_pred))
# confusion matrix
print(confusion_matrix(y, y_pred, normalize='all'))
# find the x values that give the wrong prediction: 10 into 11
index_11_10 = np.where((np.array(y) == 2) & (np.array(y_pred) == 1))[0]
param_11_10 = param.iloc[index_11_10, :]
# visualize the mistakes
plt.figure()
plt.scatter(param_11_10['Et_eV_1'], param_11_10['Et_eV_2'])
plt.show()
# %% email reminder
multi_classification.email_reminder()