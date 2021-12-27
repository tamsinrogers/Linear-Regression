'''analysis.py
Run statistical analyses and plot Numpy ndarray data
TAMSIN ROGERS
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors
import random as rand


class Analysis:
	def __init__(self, data):
		'''

		Parameters:
		-----------
		data: Data object. Contains all data samples and variables in a dataset.
		'''
		self.data = data

		# Make plot font sizes legible
		plt.rcParams.update({'font.size': 18})

	'''re-assignes the data instance using the parameter'''
	def set_data(self, data):
	
		self.data = data

	'''computes the min of each variable in headers'''
	def min(self, headers, rows=[]):
		
		minList = []
		
		dat = self.data.select_data(headers, rows)
		minList = np.amin(dat, axis=0)
			
		minList = np.array(minList)
		return minList
			
	'''computes the max of each variable in headers'''
	def max(self, headers, rows=[]):
		
		maxList = []
		
		dat = self.data.select_data(headers, rows)
		maxList = np.amax(dat, axis=0)
			
		maxList = np.array(maxList)
		return maxList

	'''computes the [min, max] range for each variable in headers'''
	def range(self, headers, rows=[]):
	
		themin = self.min(headers, rows)
		themax = self.max(headers, rows)
		
		range = [themin, themax]
		
		return range

	'''computes the mean for each variable in headers'''
	def mean(self, headers, rows=[]):

		dat = self.data.select_data(headers, rows)
		n = dat.shape[0]
		sum = np.sum(dat, axis = 0)
		mean = sum/n
		return mean

	'''computes the variance for each variable in headers'''
	def var(self, headers, rows=[]):
		
		dat = self.data.select_data(headers, rows)
		n = dat.shape[0]
		mean = self.mean(headers, rows)
		help = np.ones(dat.shape)
		meanar = (mean * help)
		selected = dat - mean
		squared = np.square(selected)
		sum = np.sum(squared, axis = 0)
		variance = np.divide(sum, (n-1))
		
		return variance

	'''computes standard deviation for each variable in headers'''
	def std(self, headers, rows=[]):

		s = np.sqrt(self.var(headers, rows))
		
		return s

	'''matplotlib show function'''
	def show(self):
	
		plt.show()

	'''creates a scatter plot with x=ind_var and y=dep_var'''
	def scatter(self, ind_var, dep_var, title):
		
		xind = self.data.headers.index(ind_var)
		yind = self.data.headers.index(dep_var)
		
		headers = [ind_var, dep_var]
		rows = []
		
		x = []
		y = []
		arr = []
		
		dat = self.data.select_data(headers, rows)
		
		for i in range(self.data.get_num_samples()): # repeat for each row
			x.append(dat[:,0])
			y.append(dat[:,1])

		arr.append(x)
		arr.append(y)
		
		plt.title(title, fontsize=15)
		plt.xlabel(ind_var)
		plt.ylabel(dep_var)
		
		colors = cartocolors.qualitative.Safe_10.mpl_colors
		plt.scatter(x, y, color=colors[rand.randint(0,9)])
		
		return np.array(arr)
	
	'''creates a line plot with x=ind_var and y=dep_var'''
	def plot(self, ind_var, dep_var, title):
		
		xind = self.data.headers.index(ind_var)
		yind = self.data.headers.index(dep_var)
		
		headers = [ind_var, dep_var]
		rows = []
		
		x = []
		y = []
		arr = []
		
		dat = self.data.select_data(headers, rows)
		
		for i in range(self.data.get_num_samples()): # repeat for each row
			x.append(dat[:,0])
			y.append(dat[:,1])

		arr.append(x)
		arr.append(y)
		
		plt.title(title, fontsize=15)
		colors = cartocolors.qualitative.Safe_10.mpl_colors
		plt.plot(x, y, color=colors[rand.randint(0,9)])
		
		return np.array(arr)
		

	'''create a pair plot, which is a combinational variable grid of scatter plots, using variable in data_vars'''
	def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
		
		fig, ax = plt.subplots(len(data_vars), len(data_vars), sharex=True, sharey=True, figsize=fig_sz)
		
		indList = []										# a list of indices of headers 
		varLength = len(data_vars)
		#colors = cartocolors.qualitative.Safe_10.mpl_colors
		rows = []
		
		
		for i in range(varLength):							# repeat variable # times
			for var in data_vars:							# each variable in the list of strings
				indList.append(self.data.header2col[var])	# use the dictionary
		
		for i in range(varLength):							# repeat variable # times
			y = self.data.data[:, indList[i]]				# get all the data from the right columns
						
			for j in range(varLength):						# repeat variable # times
				x = self.data.data[:, indList[j]]		
				ax[i][j].scatter(x,y)						# make the scatter plot
				
			# set the axis labels 
			#plt.setp(ax[-1,:], xlabel = data_vars[i])
			#plt.setp(ax[:,0], ylabel = data_vars[i])
			plt.setp(ax[-1,i], xlabel = data_vars[i])
			plt.setp(ax[i,0], ylabel = data_vars[i])
	
		
		#plt.subplots_adjust(left=None, bottom=None, right=1.5, top=1.5, wspace=.5, hspace=.3)
		#plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)	# no ticks
		
		#plt.xticks([],[])
		#plt.yticks([],[])
		
		return fig, ax
		
		
		