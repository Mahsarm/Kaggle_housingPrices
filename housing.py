import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LassoLarsCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error



class HousingPrices(object):

	
	def __init__(self):

		self.train_set = pd.read_csv('train.csv')
		self.test_set = pd.read_csv('test.csv')
		self.df = pd.concat([self.train_set , self.test_set], axis=0, ignore_index=True)
		self.df = self.df.fillna(self.df.mean())

		self.numeric_feats = self.df.dtypes[self.df.dtypes != "object"].index
		self.cat_feats = self.df.dtypes[self.df.dtypes == "object"].index



	def rescaling_numfeats(self):

		min_max =  MinMaxScaler()
		min_max.fit(self.df[self.numeric_feats]) 
		self.normalized_numfeats = min_max.transform(self.df[self.numeric_feats]) 

		return self.normalized_numfeats



	def encoding_catfeats(self):

		label_enc = LabelEncoder()
		

		self.df[self.cat_feats] = self.df[self.cat_feats].fillna("NA")
		
		for column in self.df[self.cat_feats]:
		    label_enc.fit(self.df[column]) 
		    self.df[column]=label_enc.transform(self.df[column])
    	
		onehot_enc = OneHotEncoder(sparse=False)
		onehot_enc.fit(self.df[self.cat_feats]) 
		encoded_cat_feats = onehot_enc.transform(self.df[self.cat_feats])

		self.data = np.concatenate((encoded_cat_feats , self.rescaling_numfeats()), axis=1)
		return self.data

		

	def data_slices(self):

		data_set = self.encoding_catfeats()
		number_of_samples = len(self.train_set)

		target = data_set[0:number_of_samples,-1]
		samples = data_set[0:number_of_samples ,:-1]
		train_length = int(number_of_samples*0.7)
		self.train_target = target[0:train_length]
		self.train_samples = samples[:train_length,:]	
		self.test_target = target[train_length:]
		self.test_samples = samples[train_length:,:]
		return self.train_samples, self.train_target, self.test_samples, self.test_target



	def rmse_cv(self, model, x_c, y_c):

		self.rmse_cross_val = np.sqrt(-cross_val_score(model, x_c, y_c, scoring="neg_mean_squared_error", cv = 5))
		return self.rmse_cross_val



	def Best_alpha(self):
		X_c, Y_c = slef.data_slices()[0:2]
		alphas = [0.0001,0.0005,0.001,0.005,0.01,0.05, 0.1,0.5,1,10,100,1000]
		cv_ridge = [self.rmse_cv(Ridge(alpha = alpha), X_c, Y_c).mean() for alpha in alphas]
		self.ridge_alpha = pd.Series(cv_ridge, index = alphas).argmin()
		return self.ridge_alpha 



	def rmse(self, y_true, y_predicted):

		return np.sqrt(mean_squared_error(y_true, y_predicted))



	def ridge_regression(self):

		x_train, y_train, x_test, y_test = self.data_slices()

		clf = Ridge(alpha=self.Best_alpha())
		clf.fit(x_train,y_train)

		y_pred = clf.predict(x_test)
		
		print("Ridge score on training set: ", self.rmse(y_test, y_pred))


	
housing = HousingPrices()

housing.ridge_regression()



	
		
		