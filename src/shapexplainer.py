import shap
import matplotlib.pyplot as plt
from shap import LinearExplainer, KernelExplainer, TreeExplainer, DeepExplainer   


class ShapExplainer:

	def __init__(self,model,x_train,x_test,data_type,feature_names,output_dir):
		self.model = model
		self.x_train = x_train 
		self.x_test = x_test
		self.data_type = data_type
		self.feature_names = feature_names
		self.output_dir = output_dir


	def global_explain(self):
		print(self.model)
		print(self.model.__class__.__name__)
		self.explainer = self.map_explainer()(self.model, self.x_train, feature_names=self.feature_names)
		self.shap_values = self.explainer(self.x_test)
		shap.summary_plot(self.shap_values,show=False)
		plt.savefig(self.output_dir+'global_'+self.model.__class__.__name__+'.png')


	def local_explain(self,indices_list):
		print(indices_list)
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

