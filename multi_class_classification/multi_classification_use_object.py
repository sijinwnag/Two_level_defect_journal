
# %% imports
from multi_classification_object import *
# define the object
multi_classification = multi_classifier()
# define the training path
# four sets
multi_classification.set11_path = r'G:\study\thesis_data_storage\unordered\set11\p\outputs\set11_80k.csv'
multi_classification.set10_path = r'G:\study\thesis_data_storage\unordered\set10\p\80k\2022_10_25\2022-10-25-13-57-56_advanced example - multi_level_L_datasetID_0.csv'
multi_classification.set01_path = r'G:\study\thesis_data_storage\unordered\set01\p\outputs\outputs\set01_80k.csv'
multi_classification.set00_path = r'G:\study\thesis_data_storage\unordered\set00\p\set00_80k\outputs\set00_80k.csv'
# combined dataset
multi_classification.combined_path = r'G:\study\thesis_data_storage\unordered\classification\set_classification\80k'
# %% combine the data
multi_classification = multi_classifier()
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

# %% email reminder
multi_classification.email_reminder()