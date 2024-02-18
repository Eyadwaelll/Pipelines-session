#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline , make_pipeline
from scipy.stats import randint
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, LabelEncoder, StandardScaler,RobustScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

all_cols = ['Brand',
            'Model',
            'Storage',
            'RAM',
            'Screen Size (inches)',
            'Camera (MP)',
            'Battery Capacity (mAh)']


class Datatypefix(BaseEstimator, TransformerMixin):
    """
    Transformer that selects specified columns and fixes the dtype using tools from the regex library.

    Parameters:
    ----------
    cols_to_process : list or array-like, default=None
        List of column names to select from the input data frame. (default = None)
    """
    def __init__(self, cols_to_process=None):
        self.cols_to_process = cols_to_process

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.cols_to_process:
            for col in self.cols_to_process:
                if col in X_copy.columns:
                    X_copy[col] = pd.to_numeric(X_copy[col].apply(lambda x: int(re.sub('[^\d]', '', str(x)))), errors='coerce')
                    if col == 'Screen Size (inches)':
                        X_copy[col] = X_copy[col].apply(lambda x: np.mean if '+' in str(x) else x)
                        X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')
        return X_copy

    def fit_transform(self, X, y=None):
        return self.transform(X)
    
class ColumnSelector(TransformerMixin, BaseEstimator):
    """
    Transformer that selects only the specified columns from a data frame.

    Parameters:
    ----------
    columns : list or array-like, default=None
        List of column names to select from the input data frame. If None, all columns are selected.
    """
    def __init__(self, columns=all_cols):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X = X[self.columns]
        return X
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
    

    
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that drops specified columns from a DataFrame.

    Parameters
    ----------
    cols : list
        A list of column names to be dropped.
    return
    ------
        dataframe with dropped columns
    """
    def __init__(self, cols=None):
        self.cols = cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.cols is None:
            return X
        else:
            return X.drop(self.cols,axis=1)
        

    
class BoxcoxTransform(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply a Boxcox transform to specified columns in a Pandas DataFrame.

    Parameters
    ----------
    cols : list
        The list of column names to apply the Boxcox transform to.
    domain_shift : float
        The value to be added to the columns before applying the Boxcox transform.
    """
    def __init__(self, cols, domain_shift=0):
        self.cols = cols
        self.domain_shift = domain_shift

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.cols:
            X_copy[col] = stats.boxcox(X_copy[col] + self.domain_shift)[0]
        return X_copy

    def fit_transform(self, X, y=None):
        return self.transform(X)
    
class LabelEncodeColumns(BaseEstimator, TransformerMixin):
    """
    A transformer class to encode categorical columns using LabelEncoder.

    Parameters
    ----------
    cols : list of str
        The names of the columns to be encoded.

    return
    ------
        encoded feature
    """
    def __init__(self, cols):
        self.cols = cols
        self.encoders_ = {}

    def fit(self, X, y=None):
        for col in self.cols:
            encoder = LabelEncoder()
            encoder.fit(X[col])
            self.encoders_[col] = encoder
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, encoder in self.encoders_.items():
            X_copy[col] = encoder.transform(X_copy[col])
        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    

class StandardScaleTransform(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply standard scaling to specified columns in a Pandas DataFrame.

    Parameters
    ----------
    cols : list of str
        The names of the columns to apply standard scaling to.
    """
    def __init__(self, cols):
        self.cols = cols
        self.scaler_ = None

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X.loc[:, self.cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[:, self.cols] = self.scaler_.transform(X_copy.loc[:, self.cols])
        return X_copy

    def fit_transform(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X.loc[:, self.cols])
        return self.transform(X)
    
class RobustScaleTransform(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply Robust scaling to specified columns in a Pandas DataFrame.

    Parameters
    ----------
    cols : list of str
        The names of the columns to apply Robust scaling to.
    """
    def __init__(self, cols):
        self.cols = cols
        self.scaler_ = None

    def fit(self, X, y=None):
        self.scaler_ = RobustScaler().fit(X.loc[:, self.cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[:, self.cols] = self.scaler_.transform(X_copy.loc[:, self.cols])
        return X_copy

    def fit_transform(self, X, y=None):
        self.scaler_ = RobustScaler().fit(X.loc[:, self.cols])
        return self.transform(X)
    
class CameraMegapixelsExtractor(BaseEstimator, TransformerMixin):
    """
    A transformer class to extract new features from the camera column.

    Parameters
    ----------
    camera_col : str, default='Camera (MP)'
        The name of the camera column.
    """ 
    def __init__(self, camera_col='Camera (MP)'):
        self.camera_col = camera_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()  
        X_copy['Number of Cameras'] = X_copy[self.camera_col].str.count('\\+') + 1
        megapixels = X_copy[self.camera_col].apply(lambda camera_col: re.findall(r'\d+\.*\d*', str(camera_col)))
        X_copy['Best Camera (MP)'] = megapixels.apply(lambda megapixels: max([float(mp) for mp in megapixels]))
        return X_copy

    def fit_transform(self, X, y=None):
        return self.transform(X)

    
"""
Pipeline 1:
Datatypefix :['Storage','RAM','Screen Size (inches)','Price ($)']
CameraMegapixelsExtractor :
ColumnSelector : all
LabelEncodeColumns : ['Brand','Model']
BoxcoxTransform : ['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']
DropColumnsTransformer : none
StandardScaleTransform :['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']

Pipeline 2:
Datatypefix :['Storage','RAM','Screen Size (inches)','Price ($)']
CameraMegapixelsExtractor :
ColumnSelector : all
LabelEncodeColumns : ['Brand','Model']
BoxcoxTransform : ['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']
DropColumnsTransformer : none
RobustScaleTransform :['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']

Pipeline 3:
Datatypefix :['Storage','RAM','Screen Size (inches)','Price ($)']
CameraMegapixelsExtractor :
ColumnSelector : all
LabelEncodeColumns : 'Brand'
BoxcoxTransform : ['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']
DropColumnsTransformer : 'Model'
StandardScaleTransform :['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']
    
Pipeline 4:
Datatypefix :['Storage','RAM','Screen Size (inches)','Price ($)']
CameraMegapixelsExtractor :
ColumnSelector : all
LabelEncodeColumns : 'Brand'
BoxcoxTransform : ['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']
DropColumnsTransformer : 'Model'
RobustScaleTransform :['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']

"""

class FullPipeline1:

    
    def __init__(self):
        
        self.trans_cols = ['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']
        self.fix_cols = ['Storage','RAM','Screen Size (inches)']
        self.drop_cols = ['Camera (MP)']
        self.encode_cols = ['Brand','Model']
        self.scale_cols = ['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']
        
        self.full_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=all_cols)),
            ('drop_cols', DropColumnsTransformer(cols=self.drop_cols)),
            ('data_fix', Datatypefix(cols_to_process = self.fix_cols)),
            ('camera_extraction', CameraMegapixelsExtractor(camera_col='Camera (MP)')),
            ('power_transformation', BoxcoxTransform(cols= self.trans_cols, domain_shift=1)),
            ('label_encode', LabelEncodeColumns(cols=self.encode_cols)),
            ('scale', StandardScaleTransform(self.scale_cols))
        ])
    
        self.y_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=['Price ($)'])),
            ('data_fix', Datatypefix(cols_to_process = ['Price ($)'])),
        ])
    
    def fit_transform(self, X_train, y_train):
        X_train = self.full_pipeline.fit_transform(X_train)
        y_train = self.y_pipeline.fit_transform(y_train)
        return X_train, y_train
    
    def transform(self, X_test, y_test):
        X_test = self.full_pipeline.transform(X_test)
        y_test = self.y_pipeline.transform(y_test)
        return X_test, y_test


class FullPipeline2:

    
    def __init__(self):
        
        self.trans_cols = ['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']
        self.fix_cols = ['Storage','RAM','Screen Size (inches)']
        self.drop_cols = ['Camera (MP)']
        self.encode_cols = ['Brand','Model']
        self.scale_cols = ['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']
        
        self.full_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=all_cols)),
            ('drop_cols', DropColumnsTransformer(cols=self.drop_cols)),
            ('data_fix', Datatypefix(cols_to_process = self.fix_cols)),
            ('camera_extraction', CameraMegapixelsExtractor(camera_col='Camera (MP)')),
            ('power_transformation', BoxcoxTransform(cols= self.trans_cols, domain_shift=1)),
            ('label_encode', LabelEncodeColumns(cols=self.encode_cols)),
            ('scale', RobustScaleTransform(self.scale_cols))
        ])
    
        self.y_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=['Price ($)'])),
            ('data_fix', Datatypefix(cols_to_process = ['Price ($)'])),
        ])
    
    def fit_transform(self, X_train, y_train):
        X_train = self.full_pipeline.fit_transform(X_train)
        y_train = self.y_pipeline.fit_transform(y_train)
        return X_train, y_train
    
    def transform(self, X_test, y_test):
        X_test = self.full_pipeline.transform(X_test)
        y_test = self.y_pipeline.transform(y_test)
        return X_test, y_test
    

class FullPipeline3:

    
    def __init__(self):
        
        self.trans_cols = ['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']
        self.fix_cols = ['Storage','RAM','Screen Size (inches)']
        self.drop_cols = ['Model','Camera (MP)']
        self.encode_cols = ['Brand']
        self.scale_cols = ['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']
        
        self.full_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=all_cols)),
            ('drop_cols', DropColumnsTransformer(cols=self.drop_cols)),
            ('data_fix', Datatypefix(cols_to_process = self.fix_cols)),
            ('camera_extraction', CameraMegapixelsExtractor(camera_col='Camera (MP)')),
            ('power_transformation', BoxcoxTransform(cols= self.trans_cols, domain_shift=1)),
            ('label_encode', LabelEncodeColumns(cols=self.encode_cols)),
            ('scale', StandardScaleTransform(self.scale_cols))
        ])
    
        self.y_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=['Price ($)'])),
            ('data_fix', Datatypefix(cols_to_process = ['Price ($)'])),
        ])
    
    def fit_transform(self, X_train, y_train):
        X_train = self.full_pipeline.fit_transform(X_train)
        y_train = self.y_pipeline.fit_transform(y_train)
        return X_train, y_train
    
    def transform(self, X_test, y_test):
        X_test = self.full_pipeline.transform(X_test)
        y_test = self.y_pipeline.transform(y_test)
        return X_test, y_test
    

class FullPipeline4:

    
    def __init__(self):
        
        self.trans_cols = ['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']
        self.fix_cols = ['Storage','RAM','Screen Size (inches)']
        self.drop_cols = ['Model','Camera (MP)']
        self.encode_cols = ['Brand']
        self.scale_cols = ['Storage','RAM','Screen Size (inches)','Battery Capacity (mAh)','Best Camera (MP)','Number of Cameras']
        
        self.full_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=all_cols)),
            ('drop_cols', DropColumnsTransformer(cols=self.drop_cols)),
            ('data_fix', Datatypefix(cols_to_process = self.fix_cols)),
            ('camera_extraction', CameraMegapixelsExtractor(camera_col='Camera (MP)')),
            ('power_transformation', BoxcoxTransform(cols= self.trans_cols, domain_shift=1)),
            ('label_encode', LabelEncodeColumns(cols=self.encode_cols)),
            ('scale', RobustScaleTransform(self.scale_cols))
        ])
    
        self.y_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=['Price ($)'])),
            ('data_fix', Datatypefix(cols_to_process = ['Price ($)'])),
        ])
    
    def fit_transform(self, X_train, y_train):
        X_train = self.full_pipeline.fit_transform(X_train)
        y_train = self.y_pipeline.fit_transform(y_train)
        return X_train, y_train
    
    def transform(self, X_test, y_test):
        X_test = self.full_pipeline.transform(X_test)
        y_test = self.y_pipeline.transform(y_test)
        return X_test, y_test
    
    
class ModelEvaluate:
    """
    A class that takes a list of regression models and evaluates their performance
    using cross-validation.

    Parameters
    ----------
    models : list
        A list of regression models to evaluate.

    Methods
    -------
    fit(X_train, y_train)
        Fits the regression models on the training data using cross-validation and
        stores the evaluation results in the 'results' attribute.

    get_results()
        Returns the evaluation results as a pandas DataFrame.
    """
    def __init__(self, models):
        self.models = models

    def fit(self, X_train, y_train):
        self.results = []
        for model in self.models:
            start = time.time()
            scores = cross_validate(model, X_train, y_train, cv=5,
                                    scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
                                    return_train_score=False)
            end = time.time()
            results_dict = {}
            results_dict['model'] = model.__class__.__name__
            results_dict['mean_mae'] = -np.mean(scores['test_neg_mean_absolute_error'])
            results_dict['mean_rmse'] = np.sqrt(-np.mean(scores['test_neg_mean_squared_error']))
            results_dict['mean_r2'] = np.mean(scores['test_r2'])
            results_dict['time'] = end - start
            self.results.append(results_dict)

    def get_results(self):
        return pd.DataFrame(self.results)    



class RegressionPlot:
    """A class for creating a set of plots to visualize the performance of a regression model.

    Parameters
    ----------
    y_test : pandas.DataFrame
        The actual target values for the test set.
    y_pred : array-like
        The predicted target values for the test set.
    color : str, optional
        The color to use for the plot markers and histograms.

    Methods
    -------
    plot()
        Creates a set of three plots to visualize the performance of the regression model.

    """

    def __init__(self, y_test, y_pred, color='b'):
        self.y_test = y_test
        self.y_pred = y_pred
        self.color = color
    
    def plot(self):
        """Creates a set of three plots to visualize the performance of the regression model.

        The three plots are: a scatter plot with regression line, a histogram of errors, and a residual plot.
        """

        # Create subplots
        fig, axs = plt.subplots(ncols=3, figsize=(15,5))

        # Plot scatter plot with regression line
        axs[0].scatter(self.y_test[self.y_test.columns[0]], self.y_pred, color=self.color)
        axs[0].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=3, color='k')
        axs[0].set_xlabel('Actual Values')
        axs[0].set_ylabel('Predicted Values')
        axs[0].set_title('Scatter Plot with Regression Line')

        # Plot histogram of errors
        errors = self.y_test[self.y_test.columns[0]] - self.y_pred
        axs[1].hist(errors, bins=50, color=self.color)
        axs[1].axvline(x=errors.median(), color='k', linestyle='--', lw=3)
        axs[1].set_xlabel('Errors')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Histogram of Errors')

        # Plot residual plot
        axs[2].scatter(self.y_pred, errors, color=self.color)
        axs[2].axhline(y=0, color='k', linestyle='-', lw=3)
        axs[2].set_xlabel('Predicted Values')
        axs[2].set_ylabel('Errors')
        axs[2].set_title('Residual Plot')

        # Show the plots
        plt.tight_layout()
        plt.show()


# In[ ]:




