import json
import joblib
import numpy as np
import pandas as pd
from dispatcher import get_explainer



with open('src/config.json') as f:
  config = json.load(f)

model = joblib.load(config.get("model_path",None))
x_train = np.load(config.get("x_train_path",None))
x_test = np.load(config.get("x_test_path",None))
data_type =config.get("datatype",None)
feature_names = list(np.load(config.get("feature_names",None),allow_pickle=True))
output_dir = config.get("output_dir","output/")
indices_list = config.get("indices",[])
if data_type == "text":
  vectorizer = np.load(config.get("vectorizer_path"))
  feature_names = vectorizer.get_feature_names()

algorithm = config.get("algorithm","shap")
explainer = get_explainer(algorithm)(model,x_train,x_test,data_type,feature_names,output_dir)
explainer.global_explain()
explainer.local_explain(indices_list)