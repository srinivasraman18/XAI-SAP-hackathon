import lime
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

class LimeExplainer:

	def __init__(self,model,x_train,x_test,data_type,output_dir,feature_names=None,vectorizer=None,tokenizer=None):
		self.model = model
		self.x_train = x_train 
		self.x_test = x_test
		self.data_type = data_type
		self.output_dir = output_dir
		self.vectorizer = vectorizer


	def local_explain(self,indices_list):
		lime_explainer = LimeTextExplainer()
		svc_tfidf_pipeline = make_pipeline(self.vectorizer,self.model)
		for index in indices_list:
			exp = lime_explainer.explain_instance(self.x_train[index], svc_tfidf_pipeline.decision_function)
			fig = exp.as_pyplot_figure()
			fig.savefig(self.output_dir+'local_'+self.model.__class__.__name__+str(index)+'.jpg')
			
	