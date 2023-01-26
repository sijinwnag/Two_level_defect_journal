
# %% imports
from multi_classification_object import *

# %% combine the data
multi_classification = multi_classifier()
multi_classification.data_combiner()
# %% train test the model
# define the object
multi_classification = multi_classifier()
# load the combined data
multi_classification.load_combined_data()
# train the model
multi_classification.train_test_model()
