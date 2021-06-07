from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from shapexplainer import ShapExplainer
from limeexplainer import LimeExplainer

def get_explainer(algorithm):

    return{

        'shap':ShapExplainer,
        'lime': LimeExplainer
    
    }.get(algorithm,None)