import json
import joblib
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from dispatcher import get_explainer
from tensorflow.keras.models import load_model

from tensorflow.compat.v1.keras.backend import get_session
tf.compat.v1.disable_v2_behavior()

with open('src/config.json') as f:
  config = json.load(f)


model = None
tokenizer = None
vectorizer = None
feature_names = None
model_path = config.get("model_path",None)
if model_path.endswith("h5"):
  model = load_model(model_path)
else:
  model = joblib.load(model_path,None)
x_train = np.load(config.get("x_train_path",None),allow_pickle=True)
x_test = np.load(config.get("x_test_path",None),allow_pickle=True)
data_type =config.get("datatype",None)
feature_path = config.get("feature_names",None)
if feature_path is not None:
  feature_names = list(np.load(feature_path,allow_pickle=True))
output_dir = config.get("output_dir","output/")
indices_list = config.get("indices",[])
vectorizer_path = config.get("vectorizer_path",None)
tokenizer_path = config.get("tokenizer_path",None)

if data_type == "text":
  if tokenizer_path is not None:
    with open(tokenizer_path, 'rb') as handle:
      tokenizer = pickle.load(handle)

  else:
    with open(vectorizer_path, 'rb') as handle:
      vectorizer = pickle.load(handle)

    feature_names = vectorizer.get_feature_names()

algorithm = config.get("algorithm","shap")
explainer = get_explainer(algorithm)(model,x_train,x_test,data_type,output_dir,feature_names,vectorizer,tokenizer)
if algorithm == 'shap':
  explainer.global_explain()
  print("done")
explainer.local_explain(indices_list)