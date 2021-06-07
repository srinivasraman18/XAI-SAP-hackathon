import shap
import numpy as np
import matplotlib.pyplot as plt
from shap import LinearExplainer, KernelExplainer, TreeExplainer, DeepExplainer, GradientExplainer   


class ShapExplainer:

	def __init__(self,model,x_train,x_test,data_type,output_dir,feature_names=None,vectorizer=None,tokenizer=None):
		self.model = model
		self.x_train = x_train 
		self.x_test = x_test
		self.data_type = data_type
		self.feature_names = feature_names
		self.output_dir = output_dir
		self.tokenizer = tokenizer


	def __deep_global_explain(self):
		self.explainer = DeepExplainer(self.model, self.x_train)
		self.shap_values = self.explainer.shap_values(self.x_test)
		self.word_index = dict([(value, key) for key, value in self.tokenizer.word_index.items()])
		shap.summary_plot(self.shap_values, self.x_test, self.word_index, show=False)
		plt.savefig(self.output_dir+'global_'+self.model.__class__.__name__+'.png')


	def __deep_local_explain(self,indices_list):
		x_test_words = np.stack([np.array(list(map(lambda x: self.word_index.get(x), self.x_test[i]))) for i in range(len(self.x_test))])
		for index in indices_list:
			shap.force_plot(self.explainer.expected_value[index], self.shap_values[index][0], x_test_words[index],matplotlib=True,show=False)
			plt.savefig(self.output_dir+'local_'+self.model.__class__.__name__+str(index)+'.png')


	def global_explain(self):
		if self.tokenizer is not None:
			self.__deep_global_explain()
			return
		self.explainer = self.map_explainer()(self.model, self.x_train, feature_names=self.feature_names)
		self.shap_values = self.explainer(self.x_test)
		shap.summary_plot(self.shap_values,show=False)
		plt.savefig(self.output_dir+'global_'+self.model.__class__.__name__+'.png')


	def local_explain(self,indices_list):
		if self.tokenizer is not None:
			self.__deep_local_explain(indices_list)
			return
		for index in indices_list:
			shap.plots.force(self.shap_values[index],matplotlib=True,show=False)
			plt.savefig(self.output_dir+'local_'+self.model.__class__.__name__+str(index)+'.png')


	def map_explainer(self):

		return{

        'LinearSVC':KernelExplainer,
        'KNeighborsClassifier':KernelExplainer,
        'RandomForestClassifier':TreeExplainer,
        'XGBClassifier':TreeExplainer,
        'LGBMClassifier':TreeExplainer,
        'CatBoostClassifier':TreeExplainer,
        'LogisticRegression':LinearExplainer,

   			}.get(self.model.__class__.__name__,None)

