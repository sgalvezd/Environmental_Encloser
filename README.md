The archive "Class_Functions.py" has 2 classes defined as:
Space_Analysis and Time_Analysis
Space_Analysis, is a class for reading csv files that contains the values of several properties (temperature, velocity, pressure, etc.) in each point of the mesh used.
This class contains functions to clean the data and to rewrite the names of varaibles.
When the archive is run in "Result_Analysis.ipynb" it creates a dictionary (dataset) where the key is the name of the csv (under a format) and the value is a list with an array
of the specific dataframe of that archive and its varaible dictionary. The last corresponds to a dictionary where the keys are the variable names and the values are a list with the index
and the corresponding unit for that variable.
The idea is to call an specific archive in the data set and apply the methods for plotting the varaible, histogram, ranges of values, visualization by planes, maximum value zones, etc.

The class Time_Analysis, is a class for reading plane text archives that contain the time variation of certain physical properties.
This class has a method for plotting the variation in time of max, min and mean value of temperature, pressure, dew point and mass_fraction_h2o
This class has a method for plotting the mean mass_fraction_h2o for all cases, the idea is to compare the case with less final water content.
