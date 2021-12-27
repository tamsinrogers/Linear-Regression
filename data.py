'''data.py
Reads CSV files, stores data, access/filter data by variable name
Tamsin Rogers
CS 251 Data Analysis and Visualization
Spring 2021
'''

import numpy as np
import csv

class Data:

	'''data object constructor'''
	def __init__(self, filepath=None, headers=None, data=None, header2col=None):
		
		self.filepath = filepath
		self.headers = headers
		self.data = data
		self.header2col = header2col
		
		if self.filepath != None:		# if a filepath is included in the parameters
			self.read(filepath)			# read the file

	'''read in the csv file, convert it to a 2d array and then to ndarray'''
	def read(self, filepath):
		
		with open(filepath, newline='') as csvfile:
			read = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
		
			line = 0
			colNum = []
			self.header2col = {}		# set up empty dictionary
			self.data = []				# set up empty data array
			list = []					
		
			for row in read:			
				if line == 0:			# first row is the header row
					list = row			# skip this row, it doesn't have any data
					line += 1			# go to the next row
				elif line == 1:			# data types
					index = self.rowIndices(row)
					self.headers = self.headerIndices(index, list)
					line += 1
				else:
					nRow = []
					for i in index:
						nRow.append(float(row[i]))
					self.data.append(nRow)
			self.makeDict()
			self.data = np.array(self.data)
		return
		
	# HELPER FUNCTIONS
		
	
	'''returns a list of indices of numeric variables'''
	def rowIndices(self, row):
		indices = []
		for i in range(len(row)):		# repeat for #of rows in the data
			if row[i] == 'numeric':		# the column (variable) contains numeric data
				indices.append(i)		# add the index to the list
		assert(len(indices) >= 1), "The second row does not contain the appropriate data types" # second row must be the data type
		return(indices)
	
	'''returns a list of headers'''
	def headerIndices(self, indices, arr):
		headers = []
		for i in indices:			# go through all of the columns that have numeric data
			x = arr[i].rstrip()		# get rid of spaces after the headers
			headers.append(x)		# add the names of the headers that correspond to numeric data
		return headers
	
	'''dictionary: maps the appropriate header to the given column index'''
	def makeDict(self):
		for i in range(len(self.headers)):			# go through all of the columns that have numeric data
			self.header2col[self.headers[i]] = i	# map each header to its index
		return
	
	# END OF HELPER FUNCTIONS
	
	'''returns a string showing each cell of the data (only the first 5 rows)'''
	def __str__(self):

		string = str(self.filepath) + str(self.data.shape)+'\n'	# initialize the string to be the filepath & shape of the data
		string = string + str(self.headers) + '\n'				# next line is the headers
		string = string + "-------------------------------" + '\n'
		for i in range(5):										# get the first 5 rows of data
			string = string + (str(self.data[i,:]))				# each row, given the index
			string = string + '\n'
			if i >= (np.ma.size(self.data, axis = 0)) -1:		# make sure we're not going out of range
			
				break
		return string

	'''returns the list of headers as string (first row of the csv)'''
	def get_headers(self):

		return self.headers
		

	'''returns the dictionary (header, index)'''
	def get_mappings(self):
		
		return self.header2col

	'''returns the number of variables in the data sample'''
	def get_num_dims(self):
		
		return len(self.headers)
	   
	'''returns the number of samples in the dataset'''
	def get_num_samples(self):

		return self.data.shape[0]

	'''returns the data at the given index'''
	def get_sample(self, rowInd):
		
		return self.data[rowInd]

	'''returns a list of indices that correspond to the headers in the given list'''
	def get_header_indices(self, headers):

		indices = []
		
		for i in headers:
			print(self.header2col[i])
			indices.append(self.header2col[i])
		
		return indices

	'''returns a copy of the entire dataset'''
	def get_all_data(self):

		return np.copy(self.data)

	'''returns the first five data samples'''
	def head(self):
		
		samples = []
		sampleRange = self.get_num_samples()
		
		for i in range(5):
			if i > (sampleRange - 1):
				break
			samples.append(self.data[i,:])	

		return np.array(samples)


	'''returns the last 5 data samples'''
	def tail(self):
	
		samples = []
		sampleRange = self.get_num_samples()
		if sampleRange >= 5:
			for i in range(5, 0, -1):				# we want to count from 5 down to 1
				if i >= (sampleRange - 1):
					break
				samples.append(self.data[len(self.data)-i,:])	

		else:
			for i in range(sampleRange, 0, -1):		# we want to count from sampleRange down to 1
				if i <= 0:
					print(i)
					break
				samples.append(self.data[len(self.data)-i,:])	

		return np.array(samples)

	'''update data to store only the data included in the given range'''
	def limit_samples(self, start_row, end_row):

		self.data = self.data[start_row : end_row,]


	def select_data(self, headers, rows=[]):
		'''Return data samples corresponding to the variable names in `headers`.
		If `rows` is empty, return all samples, otherwise return samples at the indices specified
		by the `rows` list.

		(Week 2)

		For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return
		column #2 of self.data. If rows is not [] (say =[0, 2, 5]), then we do the same thing,
		but only return rows 0, 2, and 5 of column #2.

		Parameters:
		-----------
			headers: Python list of str. Header names to take from self.data
			rows: Python list of int. Indices of subset of data samples to select.
				Empty list [] means take all rows

		Returns:
		-----------
		ndarray. shape=(num_data_samps, len(headers)) if rows=[]
				 shape=(len(rows), len(headers)) otherwise
			Subset of data from the variables `headers` that have row indices `rows`.

		Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
		'''
		
		samples = []
		totalRows = self.get_num_samples()
		headIndex = self.get_header_indices(headers)
		
		if len(rows) > 0:
			for i in rows:
				print(i)
				temp = []
				for j in headIndex:
					temp.append(self.data[i,j])
				samples.append(temp)
		else:
			for i in range(totalRows):
				temp = []
				for j in headIndex:
					temp.append(self.data[i,j])
				samples.append(temp)
		return np.array(samples)
				
		
		
		
		
		
		