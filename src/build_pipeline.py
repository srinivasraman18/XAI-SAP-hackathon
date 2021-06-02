from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from dispatcher import get_classifier


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class CustomLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)


class PipelineBuilder:

    def __init__(self,**kwargs):
        self.text_columns = kwargs.get('text_columns',[])
        self.numerical_columns = kwargs.get('numerical_columns',[])
        self.categorical_columns = kwargs.get('categorical_columns',[])
        self.text_extraction = kwargs.get('text_extraction','tf')
        self.categorical_params = kwargs.get('categorical_params',{})
        self.numerical_params = kwargs.get('numerical_params',{})
        self.text_params = kwargs.get('text_params',{})
        self.classifier = kwargs.get('classifier',None)
        self.classifier_params = kwargs.get('classifier_params',{})



    def create_pipeline(self):

        transformers_list = []
        text_transformer = None
        if self.text_extraction == 'tf':
            text_transformer = CountVectorizer
        else:
            text_extraction = TfidfVectorizer
        for i,column in enumerate(self.text_columns):
            transformers_list.append(('text', Pipeline([
                    ('selector', ItemSelector(key=column)),
                    ('text_'+str(i),text_transformer(**self.text_params)),
                ])))

        for i,column in enumerate(self.categorical_columns):
            transformers_list.append(('category', Pipeline([
                    ('selector', ItemSelector(key=column)),
                   ('encoder_'+str(i), CustomLabelBinarizer()),

                ])))

                
        self.pipeline = Pipeline([

            ('union', FeatureUnion(
                transformer_list=transformers_list,
            )),


            (self.classifier, get_classifier(self.classifier)(**self.classifier_params)),
        ])


        return self.pipeline




        








