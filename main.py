# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib as plt
import numpy as np
import os
import sklearn
#put file into a dataframe
path='CSV_Path'
df=pd.read_csv(path)
housing=df
#print(df.describe())
#set np seed to get the same random indexes in multiple runs
#np.random.seed(42)
#Creating a test set
#def split_train_test(data,ratio):
#    random_indices=np.random.permutation(len(data))
#    test_set_size=int(len(data)*ratio)
#    test_set=random_indices[:test_set_size]
#    train_set=random_indices[test_set_size:]
#    return data.iloc[train_set],data.iloc[test_set]

#train_set,test_set=split_train_test(housing,0.2)

#Create a test set(same thing) using sklearn
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print("train set length:",len(train_set)," test set length:",len(test_set))
housing=train_set.copy()
#print(housing.describe())
#housing.plot(kind="scatter",x="longitude",y='latitude',alpha=0.4)
#get all correlations
corr_matrix=housing.corr()
#all correlations from house value
#print(corr_matrix["median_house_value"])
# Divide by 1.5 to limit the number of income categories
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
#select most correlated values
attributes=["median_house_value","median_income","total_rooms","housing_median_age"]
from pandas.tools.plotting import scatter_matrix
colors=['Blue','Red']

#plot a scatter matrix to understand relatinships between attributes
#scatter_matrix(housing[attributes],figsize=(12,8),color=colors)


housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
#print(housing.info())


housing=train_set.drop("median_house_value",axis=1,inplace=False)
housing_labels=train_set["median_house_value"].copy()


#sklearn's function impute can be used to take care of missing values
from sklearn.preprocessing import Imputer
imputer=Imputer(strategy="median")
#dropping ocean_proximity since imputer only works on numbers
housing_num=housing.drop("ocean_proximity",axis=1)
#applying imputer and median is stored in statistics_ instance variable
imputer.fit(housing_num)
#apply imputer and the 
X=imputer.transform(housing_num)
housing_tr=pd.DataFrame(X,columns=housing_num.columns)#converting np array to a dataframe

housing_cat=housing["ocean_proximity"]
#convert non-number attributes
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out
#encoder=LabelBinarizer()
#housing_cat_hot=encoder.fit_transform(housing_cat)



from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix,bedrooms_ix,population_ix,household_ix=3,4,5,6

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room=add_bedrooms_per_room
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        rooms_per_household=X[:,rooms_ix]/X[:,household_ix]
        population_per_household=X[:,population_ix]/X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room=X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]
attr_adder=CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs=attr_adder.transform(housing.values)

from sklearn.base import BaseEstimator,TransformerMixin

class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names=attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_attribs=list(housing_num)
cat_attribs=["ocean proximity"]
num_pipeline=Pipeline([
        ('selector',DataFrameSelector(num_attribs)),
        ('imputer',Imputer(strategy='median') ),
        ('attribs_adder',CombinedAttributesAdder() ),
        ('std_scaler',StandardScaler())
        ])
housing_num_tr=num_pipeline.fit_transform(housing_num)

cat_pipeline=Pipeline([
        ('selector',DataFrameSelector(num_attribs)),
        ('label_binarizer',CategoricalEncoder())
        ])
#combining both pipelines
    
from sklearn.pipeline import FeatureUnion
full_pipeline=FeatureUnion(transformer_list=[
        ('num_pipeline',num_pipeline),
        ('cat_pipeline',cat_pipeline),
        ])
    
housing_prepared=full_pipeline.fit_transform(housing)
#applying linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)

some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
some_data_prepared=full_pipeline.transform(some_data)
print("predictions:",lin_reg.predict(some_data_prepared))
print("Labels:",list(some_labels))

from sklearn.metrics import mean_squared_error
housing_predictions=lin_reg.predict(housing_prepared)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
print("error:",lin_rmse)

#training another model

#from sklearn.tree import DecisionTreeRegressor
#tree_reg=DecisionTreeRegressor()
#tree_reg.fit(housing_prepared,housing_labels)
#getting predictions
#housing_predictions=tree_reg.predict(housing_prepared)
#tree_mse=mean_squared_error(housing_labels,housing_predictions)
#tree_rmse=np.sqrt(tree_mse)
#print("Tree rmse:",tree_rmse)

#training another model

from sklearn.ensemble import RandomForestRegressor
forest_reg=RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_labels)
forest_mse=mean_squared_error(housing_labels,housing_predictions)
forest_rmse=np.sqrt(forest_mse)
print("RFR Error:",forest_rmse)

#fine tuning the model i.e.finding the best combinaton of hyperparameter values
#can also use Randomized search which is better when the search space is large
#from sklearn.model_selection import GridSearchCV
#param_grid=[
 #       {'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
  #      {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},
   #     ]
#forest_reg=RandomForestRegressor()
#grid_search=GridSearchCV(forest_reg,param_grid,cv=5,scoring="neg_mean_squared_error")
#grid_search.fit(housing_prepared,housing_labels)
#print("Best Paramaters",grid_search.best_estimator_)

#Finally, Evaluate on the test set
#final_model=grid_search.best_estimator_
#X_test=strat_test_set.drop("median_house_value",axis=1)
#y_test=strat_test_set["median_house_value"].copy()

