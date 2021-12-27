'''linear_regression.py
Subclass of Analysis that performs linear regression on data
TAMSIN ROGERS
CS251 Data Analysis Visualization
Spring 2021
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import analysis


class LinearRegression(analysis.Analysis):
	'''
	Perform and store linear regression and related analyses
	'''

	'''initializes linear regression variables'''
	def __init__(self, data):
		'''

		Parameters:
		-----------
		data: Data object. Contains all data samples and variables in a dataset.
		'''
		super().__init__(data)

		# ind_vars: Python list of strings.
		#	1+ Independent variables (predictors) entered in the regression.
		self.ind_vars = None
		# dep_var: string. Dependent variable predicted by the regression.
		self.dep_var = None

		# A: ndarray. shape=(num_data_samps, num_ind_vars)
		#	Matrix for independent (predictor) variables in linear regression
		self.A = None

		# y: ndarray. shape=(num_data_samps, 1)
		#	Vector for dependent variable predictions from linear regression
		self.y = None

		# R2: float. R^2 statistic
		self.R2 = None

		# Mean SEE. float. Measure of quality of fit
		self.m_sse = None

		# slope: ndarray. shape=(num_ind_vars, 1)
		#	Regression slope(s)
		self.slope = None
		# intercept: float. Regression intercept
		self.intercept = None
		# residuals: ndarray. shape=(num_data_samps, 1)
		#	Residuals from regression fit
		self.residuals = None

		# p: int. Polynomial degree of regression model (Week 2)
		self.p = 1

	'''linear regression on independent variables and dependent variable'''
	def linear_regression(self, ind_vars, dep_var):
		
		self.ind_vars = ind_vars
		self.dep_var = dep_var
		self.A = self.data.select_data(self.ind_vars)
		
		Ahat = np.hstack((np.ones((self.data.get_num_samples(), 1)), self.A))
		self.y = self.data.select_data([self.dep_var]).T.reshape(self.data.get_num_samples(), 1)
		c,_,_,_ = scipy.linalg.lstsq(Ahat, self.y)
		smd = np.sum(np.square(self.y - np.mean(self.y)))
		
		self.slope = c[1:]
		self.intercept = c[0][0]
		
		y_pred = self.predict()
		self.R2 = self.r_squared(y_pred)
	
		self.residuals = self.compute_residuals(y_pred)
		self.m_sse = self.mean_sse()

	'''predict the result of mA + b  '''
	def predict(self, X=None):
		'''Use fitted linear regression model to predict the values of data matrix self.A.
		Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
		A is the data matrix.

		Parameters:
		-----------
		X: ndarray. shape=(num_data_samps, num_ind_vars).
			If None, use self.A for the "x values" when making predictions.
			If not None, use X as independent var data as "x values" used in making predictions.

		Returns
		-----------
		y_pred: ndarray. shape=(num_data_samps, 1)
			Predicted y (dependent variable) values

		NOTE: You can write this method without any loops!
		'''
		
		if self.p > 1 and X is not None:
			A = self.make_polynomial_matrix(X, self.p)
		elif self.p > 1 and X is None:
			A = self.make_polynomial_matrix(self.A, self.p)
		elif X is None:
			A = self.A
		else:
			A = X
		y_pred = self.intercept + A @ self.slope
		return y_pred
		

	def r_squared(self, y_pred):
		'''Computes the R^2 quality of fit statistic

		Parameters:
		-----------
		y_pred: ndarray. shape=(num_data_samps,).
			Dependent variable values predicted by the linear regression model

		Returns:
		-----------
		R2: float.
			The R^2 statistic
		'''
		resid = np.sum(np.square(self.y - y_pred))
		mresid = np.sum(np.square(self.y - np.mean(self.y)))
		self.R2 = 1 - (resid/mresid)
		return self.R2

	def compute_residuals(self, y_pred):
		'''Determines the residual values from the linear regression model

		Parameters:
		-----------
		y_pred: ndarray. shape=(num_data_samps, 1).
			Data column for model predicted dependent variable values.

		Returns
		-----------
		residuals: ndarray. shape=(num_data_samps, 1)
			Difference between the y values and the ones predicted by the regression model at the
			data samples
		'''
		new = self.make_polynomial_matrix(self.A, self.p)
		Ahat = np.hstack((np.ones((self.data.get_num_samples(), 1)), new))
		c = np.vstack((self.intercept, self.slope))
		r = self.y - Ahat @ c
		return r

	def mean_sse(self):
		'''Computes the mean sum-of-squares error in the predicted y compared the actual y values.
		See notebook for equation.

		Returns:
		-----------
		float. Mean sum-of-squares error

		Hint: Make use of self.compute_residuals
		'''
		y_pred = self.predict()						# get the predicted y values
		residuals = self.compute_residuals(y_pred)	# compute residuals for those y values
		n = residuals.shape[0]						# num of residuals
		resid = np.sum(np.square(self.y - y_pred))
		self.m_sse = resid/n
		
		return self.m_sse

	def scatter(self, ind_var, dep_var, title):
		'''Creates a scatter plot with a regression line to visualize the model fit.
		Assumes linear regression has been already run.

		Parameters:
		-----------
		ind_var: string. Independent variable name
		dep_var: string. Dependent variable name
		title: string. Title for the plot

		TODO:
		- Use your scatter() in Analysis to handle the plotting of points. Note that it returns
		the (x, y) coordinates of the points.
		- Sample evenly spaced x values for the regression line between the min and max x data values
		- Use your regression slope, intercept, and x sample points to solve for the y values on the
		regression line.
		- Plot the line on top of the scatterplot.
		- Make sure that your plot has a title (with R^2 value in it)
		'''

		title += (f' {ind_var} vs. {dep_var} with R2 = {self.R2}')
		x,y = analysis.Analysis.scatter(self, ind_var, dep_var, title)
		x_line = np.linspace(x.min(), x.max())
		line_x = np.linspace(x.min(), x.max())
		
		if self.p > 1:
			x_line = self.make_polynomial_matrix(x_line, self.p)
			y_line = self.intercept + x_line @ self.slope
			plt.plot(x_line[:,-1], y_line[:,-1], 'r')
		else:
			y_line = (self.intercept + self.slope * x_line).reshape(x_line.shape[0], 1)
			plt.plot(x_line, y_line, 'r')
			
	def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
		'''Makes a pair plot with regression lines in each panel.
		There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
		on x and y axes.

		Parameters:
		-----------
		data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
		fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
			This is useful to change if your pair plot looks enormous or tiny in your notebook!
		hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
			pairplot.

		TODO:
		- Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
		Note that this method returns the figure and axes array that you will need to superimpose
		the regression lines on each subplot panel.
		- In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
		that you used for self.scatter. Note that here you will need to fit a new regression for
		every ind and dep variable pair.
		- Make sure that each plot has a title (with R^2 value in it)
		'''
		fig, ax = analysis.Analysis.pair_plot(self, data_vars)
		
		indList = []										# a list of indices of headers 
		varLength = len(data_vars)

		for i in range(varLength):								# repeat variable # times
			for var in data_vars:							 	# each variable in the list of strings
				indList.append(self.data.header2col[var])		# use the dictionary

		for i in range(varLength):								# repeat variable # times
			y = self.data.data[:, indList[i]]	
			
			for j in range(varLength):
				x = self.data.data[:, indList[j]]
				
				if i==j:
					if hists_on_diag is True:
						numVars = len(data_vars)
						ax[i, j].remove()
						ax[i, j] = fig.add_subplot(numVars, numVars, i*numVars+j+1)
						if j < numVars-1:
							ax[i, j].set_xticks([])
						else:
							ax[i, j].set_xlabel(data_vars[i])
						if i > 0:
							ax[i, j].set_yticks([])
						else:
							ax[i, j].set_ylabel(data_vars[i])

						ax[i][i].hist(x)						# diagonal = (i,i)
				
				if (hists_on_diag is False) or ((hists_on_diag is True) and i!=j):
					ax[i,j].scatter(x,y)						# make the scatter plot
				
					self.linear_regression([data_vars[j]], data_vars[i]) 
				
					line_x = np.linspace(x.min(), x.max())
					line_y = (self.intercept + (self.slope * line_x))
				
				self.y_pred = self.predict()
				self.R2 = self.r_squared(self.y_pred)
				
				if hists_on_diag is True and (i!=j):			# no regression line on histogram diagonal
					ax[i][j].title.set_text(f' R2 = {self.R2:.4f}')
					ax[i][j].plot(line_x, line_y.reshape((line_y.size,)),label="R2")
	
			plt.setp(ax[-1,i], xlabel = data_vars[i])
			plt.setp(ax[i,0], ylabel = data_vars[i])

	def make_polynomial_matrix(self, A, p):
		'''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
		for a polynomial regression model of degree `p`.

		(Week 2)

		Parameters:
		-----------
		A: ndarray. shape=(num_data_samps, 1)
			Independent variable data column vector x
		p: int. Degree of polynomial regression model.

		Returns:
		-----------
		ndarray. shape=(num_data_samps, p)
			Independent variable data transformed for polynomial model.
			Example: if p=10, then the model should have terms in your regression model for
			x^1, x^2, ..., x^9, x^10.

		NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
		should take care of that.
		'''
		
		help = np.empty((A.shape[0], p))
		for i in range(1,p+1):
			help[:,i-1] = A.reshape(A.shape[0])**i
		return help

	def poly_regression(self, ind_var, dep_var, p):
		'''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
		(Week 2)
		NOTE: For single linear regression only (one independent variable only)

		Parameters:
		-----------
		ind_var: str. Independent variable entered in the single regression.
			Variable names must match those used in the `self.data` object.
		dep_var: str. Dependent variable entered into the regression.
			Variable name must match one of those used in the `self.data` object.
		p: int. Degree of polynomial regression model.
			 Example: if p=10, then the model should have terms in your regression model for
			 x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).

		TODO:
		- This method should mirror the structure of self.linear_regression (compute all the same things)
		- Differences are:
			- You create a matrix based on the independent variable data matrix (self.A) with columns
			appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
			- You set the instance variable for the polynomial regression degree (self.p)
		'''
		self.ind_vars = [ind_var]
		self.dep_var = dep_var
		self.p = p
		self.A = self.data.select_data(self.ind_vars)
		
		new = self.make_polynomial_matrix(self.A, self.p)
		Ahat = np.hstack((np.ones((self.data.get_num_samples(),1)),new.reshape(self.data.get_num_samples(), self.p)))
		
		self.y = self.data.select_data([self.dep_var]).T.reshape(self.data.get_num_samples(), 1)
		c,_,_,_ = scipy.linalg.lstsq(Ahat, self.y)
		
		self.slope = c[1:]
		self.intercept = c[0][0]
		
		y_pred = self.predict()
		self.R2 = self.r_squared(y_pred)
	
		self.residuals = self.compute_residuals(y_pred)
		self.m_sse = self.mean_sse()
		
	def get_fitted_slope(self):
		'''Returns the fitted regression slope.
		(Week 2)

		Returns:
		-----------
		ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
		'''
		return self.slope

	def get_fitted_intercept(self):
		'''Returns the fitted regression intercept.
		(Week 2)

		Returns:
		-----------
		float. The fitted regression intercept(s).
		'''
		return self.intercept

	def initialize(self, ind_vars, dep_var, slope, intercept, p):
		'''Sets fields based on parameter values.
		(Week 2)

		Parameters:
		-----------
		ind_var: str. Independent variable entered in the single regression.
			Variable names must match those used in the `self.data` object.
		dep_var: str. Dependent variable entered into the regression.
			Variable name must match one of those used in the `self.data` object.
		slope: ndarray. shape=(num_ind_vars, 1)
			Slope coefficients for the linear regression fits for each independent var
		intercept: float.
			Intercept for the linear regression fit
		p: int. Degree of polynomial regression model.

		TODO:
		- Use parameters and call methods to set all instance variables defined in constructor.
		'''
		
		self.ind_vars = ind_vars
		self.dep_var = dep_var
		self.slope = slope
		self.intercept = intercept
		self.p = p
		y_pred = self.predict()
		self.R2 = self.r_squared(y_pred)
		self.residuals = self.compute_residuals(y_pred)
		self.m_sse = self.mean_sse()
		self.A = self.data.select_data(self.ind_vars)
		
		
		