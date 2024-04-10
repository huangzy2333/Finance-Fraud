from code.baseline import baseline_models
from code.cm5 import stacking_cm5
from code.cm6 import stacking_cm6

data_path = '../data/TFN/'
save_path = '../results/'

baseline_models(data_path, save_path)
stacking_cm5(data_path, save_path)
stacking_cm6(data_path, save_path)

