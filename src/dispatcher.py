from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from shapexplainer import ShapExplainer

def get_explainer(algorithm):

    return{

        'shap':ShapExplainer
    
    }.get(algorithm,None)