## import libraries and packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from math import pi
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.interpolate import griddata
from matplotlib.path import Path
import re
import os
import glob
    
class Space_Analysis:
    """Class for upload, clean and plot dataset from different .csv archives, which resulted
    from numerical simualtions"""
    #######################################################
    ## internal constants and values
    #######################################################
    skip_mesh_rows = 0 ## number of rows to skip in csv archive
    ## unit dictionary
    ## format->{variable:[unit]}
    unit_dict = {'x':'[m]','y':'[m]','z':'[m]','pressure':'[Pa]','density':'[kg/m^3]','velocity':'[m/s]','x velocity':'[m/s]',
             'y velocity':'[m/s]','z velocity':'[m/s]','vorticity':'[1/s]','temperature':'[K]','turb kinetic energy':'[m^2/s^2]',
             'specific diss rate':'[1/s]','viscosity turb':'[m^2/s]','h2o':'[-]','o2':'[-]','n2':'[-]','cell volume':'[m^3]',
             'relative humidity':'[%]','heat flux':'[W]','cellnumber':'[-]','turb diss rate':'[m^2/s^3]','nodenumber':'[-]'}
    ## surface plot dictionary
    ## default planes for plotting
    planes_dictionary = {"x0":0,"y0":0,"y2":0.3,"x1":-0.6,"y1":-0.3,"x2":0.6,"z0":0,"z2":-4.4,"z3":-3.4,
                         "z4":-2.4,"z5":-1.4,"z6":1.4,"z7":2.4,"z8":3.4,"z9":4.4}
    
    ## dataset
    dataset = dict()

    ## dataframe and dictionary
    archive = ""
    dataframe = list()
    variable_dict = dict()
    
    ## plot parameters
    mesh_plot = True
    save_plot = False
    x_size = 15 #x-dim
    y_size = 15 #y-dim

    ## color maps
    colormaps = list(plt.colormaps())
    #######################################################
    ## set dataframe and variable dictionary
    #######################################################
    def set_dataframe(self,archive):
        """uploads dataframe from a single archive of the dataset"""
        self.dataframe = self.dataset[archive][1]
        self.variable_dict = self.dataset[archive][2]
        self.archive = archive
        print("To show list of colormap run 'Space_Analysis().colormaps'")
        print("functions: plot_histograms,plot_histogram,plot_velocityfield,plot_max_regions,plot,plot_layers")
        print(self.variable_dict.keys())
        return
    #######################################################
    ## get all csv docs
    #######################################################
    def get_csv_files(self):
        """Creates a list of all archives in a Folder that fullfills the pattern of
        the archives that hold the database of a single simulation mesh values"""
        search_pattern = os.path.join(r"C:\Users\sebas\Desktop\SAPHIR", '**', '*.csv')
        all_paths = glob.glob(search_pattern, recursive=True) ##list of all paths for each archive
        csv_path = [] ## path of csv
        csv_name = [] ## name of csv
        #######################################################
        for path in all_paths: ## call all archives
            if re.match(r"Mesh_D\d\d_T\d.csv",path.split("\\")[-1]):
                csv_path.append(path)
                name_aux = path.split("\\")[-1].split(".")[0].split("_") ##new format name
                csv_name.append(name_aux[1]+"_"+name_aux[-1]) ##
        return csv_path, csv_name
    #######################################################
    ## create dataset
    #######################################################
    def create_dataset(self):
        """Creates a dataset with dict format. dict->{index,dataframe,dict_of_variables}"""
        dataset = dict()
        archive_path, archive_names = self.get_csv_files() ##gets paths and names
        index = 0
        for path in archive_path:
            df,var_dict = self.read_data(path) # read data of dataframe
            df = self.clean_zeros(df,var_dict) # clean density zeros
            dataset[archive_names[index]] = [index,df,var_dict] ## adss value to dataset
            index = index+1
        print("Names of read archives")
        print(dataset.keys())
        return dataset
    #######################################################
    ## clean dataframe for 0 density and relative humidity
    #######################################################
    def clean_zeros(self,dataframe,var_dict):
        """filters values of a dataframe where density, relative humidity and pressure are 0"""
        density = dataframe[:,var_dict["density"][0]] ##density vector
        rh = dataframe[:,var_dict["relative humidity"][0]] ##relative humidity vector
        indices = list()
        for i in range(len(density)):
            if density[i]!=0 and rh[i]!=0: ##filter
                indices.append(i)
        if len(indices)>0: ## prints number of deleted rows
            print(r"Cleaning dataframe density=0 and relative humidity=0")
            print(f"{len(density)-len(indices)} from {len(density)} substracted. Remaining:{len(indices)}")
        ## reshape dataframe
        new_dataframe = np.zeros((len(indices),len(dataframe[0,:]))) ##new dimensions
        for j in range(len(dataframe[0,:])):
            new_dataframe[:,j] = dataframe[indices,j] ##add filtered values 
        return new_dataframe
    #######################################################
    ## read document of mesh
    #######################################################
    def read_data(self,archive_PATH=""):
        """reads csv archive (single simulation mesh) cleans data, creates dataframe and
        variable dictionary"""
        if len(archive_PATH)==0:
            raise ValueError("No PATH in function")
        ### read document ######################################################################################
        df = pd.read_csv(archive_PATH,skiprows=self.skip_mesh_rows)
        header = df.columns.tolist()
        variable_dict = {}
        ## clean data from nan
        df = self.data_clean(np.array(df))
        ### add column of dew point
        rows,cols = df.shape
        dataframe = np.zeros((rows,cols+1))
        for slot in header:
            clean_name = self.clean_name(slot)
            ### review how to get names and units
            if clean_name not in variable_dict:
                try:
                    variable_dict[clean_name]=[len(variable_dict),self.unit_dict[clean_name]]
                except:
                    raise ValueError(f"variable {clean_name} without unit assignation")
            dataframe[:,(len(variable_dict)-1)] = df[:,(len(variable_dict)-1)] 
        variable_dict["dew point"]=[len(variable_dict),"[°C]"]
    ### add column of dew point 
        rows,cols = df.shape
        new_dataframe = np.zeros((rows,cols+1))
        for col in range(cols):
            new_dataframe[:,col]=df[:,col]
    ## search temperature and mass fraction ######################################################################################
        temperature = new_dataframe[:,variable_dict["temperature"][0]]
        pressure = new_dataframe[:,variable_dict["pressure"][0]]
        mass_fraction = new_dataframe[:,variable_dict["h2o"][0]]
        new_dataframe[:,-1] = self.dew_point(temperature,pressure,mass_fraction)
    ## re-shaping  ######################################################################################
        if new_dataframe.shape[1]!=len(variable_dict):
            dataframe = np.zeros((rows,len(variable_dict)))
            list_keys = list(variable_dict.keys())
            index = 0
            for key in list_keys[:-1]:
                dataframe[:,index] = new_dataframe[:,variable_dict[key][0]]
                index = index+1
            dataframe[:,-1] = new_dataframe[:,-1]
        else:
            dataframe = new_dataframe
    ### adjust to center ######################################################################################
        for i in ["x","y","z"]:
            max_val = max(dataframe[:,variable_dict[i][0]])
            min_val = min(dataframe[:,variable_dict[i][0]])
            mid_val = 0.5*(max_val+min_val)
            dataframe[:,variable_dict[i][0]] = dataframe[:,variable_dict[i][0]]-mid_val
        print("%s: rows = %d, cols = %d"%(archive_PATH.split("\\")[-1].split(".")[0].replace("Mesh_",""),dataframe.shape[0],dataframe.shape[1]))
        return dataframe, variable_dict
    #############################################################################################################################################
    ## plot histograms ##########################################################################################################################
    #############################################################################################################################################
    def plot_histograms(self,bins=100,density=True,xlog=False,ylog=False):
        """Function that plots histograms of all variables"""
        print("kwargs: bins[int],density[boolean],xlog[boolean],ylog[boolean]")
        variables = list(self.variable_dict)
        n = len(variables)
        ## calculate number of rows (3 columns plot)
        if n//3<n/3:
            plot_rows = n//3+1
        else:
            plot_rows = n//3
        ## set the plot matrix
        fig, axes = plt.subplots(plot_rows,3, figsize=(15, 20))
        for i in range(1,n-1):##eliminating cellnumber
            var = self.dataframe[:,i]
            row = (i-4)//3+1 ## changes rows from 0 to 2
            col = (i-4)%3 ## changes col every 3 rows
            axes[row,col].hist(var,bins=bins,density=density) ##plot histogram
            ## add title accroding to variable
            axes[row,col].set_title(variables[i]+" "+self.variable_dict[variables[i]][1],fontsize=12)
            ## option for log scale
            if ylog:
                axes[row,col].set_yscale("log")
            if xlog:
                axes[row,col].set_xscale("log")
        plt.grid(True)
        fig.suptitle(f'Histograms of flow & mesh properties, CASE {self.archive}', fontsize=18,y=1.01)
        plt.tight_layout()
        plt.show()
    #############################################################################################################################################
    ## plot histogram for 1 variable ############################################################################################################
    #############################################################################################################################################
    def plot_histogram(self,variable,bins=100,density=True,xlog=False,ylog=False):
        """plot single variable histogram"""
        print("kwargs:variable[str],bins[int],density[boolean],xlog[boolean],ylog[boolean]")
        fig = plt.figure(figsize=(10, 8))
        if len(variable)==0:
            raise ValueError("Choose variable from:",self.variable_dict.keys())
        plt.hist(self.dataframe[:,self.variable_dict[variable][0]],bins=bins,density=density)
        if ylog:
            plt.yscale("log")
        if xlog:
            plt.xscale("log")
        plt.grid(True)
        plt.title(variable+" "+self.variable_dict[variable][1]+f", CASE {self.archive}",fontsize=18)
        plt.tight_layout()
        plt.show()
    #############################################################################################################################################
    ## clean document of mesh ###################################################################################################################
    #############################################################################################################################################
    @staticmethod
    def data_clean(dataframe):
        """Cleans string data and nan data turning it to 0 or float"""
        rows,cols = dataframe.shape
        new_dataframe = np.zeros((rows,cols))
        for i in range(rows):
            for j in range(cols):
                if isinstance(dataframe[i,j],str):##if it is string
                    try:
                        new_dataframe[i,j] = float(dataframe[i,j])
                    except:
                        new_dataframe[i,j] = 0
                elif isinstance(dataframe[i,j],float): ##if it is float
                    new_dataframe[i,j] = dataframe[i,j]
                else:
                    raise ValueError(f"Unknown variable in {i},{j}: {dataframe[i,j]}")
        return new_dataframe
    #############################################################################################################################################
    ## dew point function #######################################################################################################################
    #############################################################################################################################################
    @staticmethod
    def dew_point(temperature,pressure,mass_fraction,RH=[]):
        """calculates dew point using T,P,w or RH and T"""
        ## range of formula -40°C to 50°C
        if len(temperature) != len(mass_fraction):
            raise ValueError("length of temperature and mass_fraction differ")

        if len(temperature[temperature<233])+len(temperature[temperature>324])>0:
            print("Warning: Values of temperature out of range [-40°C,50°C]")
        ## constants
        temperature_Celsius = temperature-273.15
        pressure_absolute = pressure+101325
        if len(RH)==0:
            a = 611.21 #Pa
            b = 18.678
            c = 257.14 #°C
            Psat = a*np.exp(b*temperature_Celsius/(c+temperature_Celsius))
            RH = mass_fraction*pressure_absolute/(Psat*0.622) ### changed
        else: ##
            RH = RH/100
        for i in range(len(RH)): ##maximum RH=1
            RH[i] = min(RH[i],1)
        b = 17.625
        c = 243.04 #°C
        gamma_factor = np.zeros(len(mass_fraction))
        for i in range(len(temperature)):
            if RH[i]!=0:
                gamma_factor[i] = np.log(RH[i])+b*temperature_Celsius[i]/(c+temperature_Celsius[i])
            else: ##factor for -100 C dew point
                gamma_factor[i] = 100*b/(100-c)
        Tdew = c*gamma_factor/(b-gamma_factor)
        return Tdew
    #############################################################################################################################################
    ## clean name and unit ######################################################################################################################
    #############################################################################################################################################
    @staticmethod
    def clean_name(name):
        """reformats name of variables"""
        new_name = name.replace(' ',"").replace('-',' ').replace('coordinate','').replace('magnitude','').replace('mag','')
        if new_name[-1]==' ':
            new_name = new_name[:-1]
        return new_name
    #############################################################################################################################################
    ## condition handling #######################################################################################################################
    #############################################################################################################################################
    @staticmethod
    def filter_by_condition(condition,data_vector,variable,unit):
        """Function for filter a varaible by putting a range like A1<variable<A2,
        variable>A1 or variable<A2, it is used in plot(arg_1,..,condition,...,arg_n) function"""
        ## two conditions filter
        if len(condition.split("&"))==2:
            cond1,cond2 = condition.split("&")
            ## choose order of limits 
            if "<" in cond1 and ">" in cond2:
                limit_up = float(cond1.split("<")[-1])
                limit_low = float(cond2.split(">")[-1])
            elif "<" in cond2 and ">" in cond1:
                limit_up = float(cond2.split("<")[-1])
                limit_low = float(cond1.split(">")[-1])
            else:
                raise ValueError("Format of condition is '<upper&>lower'/'>lower&<upper'")
            ## problem report
            if limit_up<limit_low:
                raise ValueError("Contradiction in condition upper limit smaller than lower limit")
            ## filter dataframe
            indices = []
            ## get indices of points that fullfill the condition
            for index in range(len(data_vector)):
                if data_vector[index]<limit_up and data_vector[index]>limit_low:
                    indices.append(index)
            title = "Plot of "+str(limit_low)+unit+"<"+variable+"<"+str(limit_up)+unit
            return indices, title
        ## one condition filter #############################################
        elif len(condition.split("&"))==1:
            ## choose between greater or lower than
            if "<" in condition:
                limit = float(condition.split("<")[-1])
                ## filter dataframe
                indices = []
                ## get indices of points that fullfill the condition
                for index in range(len(data_vector)):
                    if data_vector[index]<limit:
                        indices.append(index)
                title = "Plot of "+variable+"<"+str(limit)+unit
                return indices, title
                
            elif ">" in condition:
                limit = float(condition.split(">")[-1])
                ## filter dataframe
                indices = []
                ## get indices of points that fullfill the condition
                for index in range(len(data_vector)):
                    if data_vector[index]>limit:
                        indices.append(index)
                title = "Plot of "+variable+">"+str(limit)+unit
                return indices, title
            else:
                raise ValueError("'<' and '<' absent in condition")
        else:
            raise ValueError("Formats are '>number'/'<number' or '<upper&>lower'/'>lower&<upper'")
    #############################################################################################################################################
    ## regions max values #######################################################################################################################
    #############################################################################################################################################
    ##find the index of the max value
    @staticmethod
    def get_max_index(array,index_list):
        """finds the index of the maximum value of a variable"""
        max_index = index_list[0]
        for index in index_list[1:]:
            if array[index]>array[max_index]: ##condition of greater value found
                max_index = index
        return max_index
    ##get the indices within and outside of length (in 3 directions)
    @staticmethod
    def get_region(x,y,z,length,index_list,max_index):
        """Creates a list of indices which are spacially around a maximum value"""
        indices = []
        not_indices = []
        ## center point
        x0 = x[max_index]
        y0 = y[max_index]
        z0 = z[max_index]
        ##collect the points around center point at "length" distance
        for index in index_list:
            condition_in_x = x[index]<x0+length and x[index]>x0-length
            condition_in_y = y[index]<y0+length and y[index]>y0-length
            condition_in_z = z[index]<z0+length and z[index]>z0-length
            condition = condition_in_x and condition_in_y and condition_in_z
            if condition:
                indices.append(index)
            else:
                not_indices.append(index)
        if len(indices)+len(not_indices)!=len(index_list):
            raise ValueError("Mising indices, len(indices)+len(not indices)!=len(index list)")
        return indices,not_indices
    ## get the list of indices of max regions #length in [m]
    def regions_max_values(self,x,y,z,v,length=0.2,n_regions=3):
        """generates a list of indices for a number of regions where a max value is found"""
        zones = [] ## list of list of points around max value
        not_indices = np.arange(0,len(v),1) ## initialization of indices
        for region in range(n_regions):
            if len(not_indices)==0:
                print(f"only {region} regions will be plotted")
                return zones
            max_index = self.get_max_index(v,not_indices)
            ## indices: points around max value
            ## not_indices: points out of max val region
            indices,not_indices = self.get_region(x,y,z,length,not_indices,max_index)
            zones.append(indices)
        return zones
    #############################################################################################################################################
    ## Filter values by surface #################################################################################################################
    #############################################################################################################################################
    @staticmethod
    def get_surface(x,y,z,plane,value,e=1e-2):
        """creates a surface slice of the volumen. Where it takes a set of points around a
        distance e of the plane"""
        indices = []
        ## choose coordinate
        if "z" in plane.lower():
            position = z
        elif "y" in plane.lower():
            position = y
        elif "x" in plane.lower():
            position = x
        else:
            raise ValueError("Plane must be X,x,Y,y or Z,z")
        index = 0
        for val in position:
            ## filter by values near the plane's value
            if val<value+e and val>value-e:
                indices.append(index)
            index = index+1
        return indices
    #############################################################################################################################################
    ## Plot Mesh ################################################################################################################################
    #############################################################################################################################################
    @staticmethod
    def plot_mesh(ax):
        """plot the physical space where the simulation was performed"""
        ### mesh plot ##########################################################################################################################
        ## face z-
        f1x = [-0.85,0.85,0.85,0.36,-0.36,-0.85,-0.85]
        f1y = [-0.9,-0.9,0.4,0.9,0.9,0.4,-0.9]
        f1z = [-5,-5,-5,-5,-5,-5,-5]
        ax.plot3D(f1x,f1z,f1y,'-k',lw=1)
        ## face z+
        f2z = [5,5,5,5,5,5,5]
        ax.plot3D(f1x,f2z,f1y,'-k',lw=1)
        ## cylinder
        th = np.linspace(0,2*pi,100)
        cx = 0.3475*np.cos(th)
        cy = 0.3475*np.sin(th)
        ax.plot3D(cx,[-4.5]*100,cy,'-k',lw=1)
        ax.plot3D(cx,[4.5]*100,cy,'-k',lw=1)
        ax.plot3D([0,0],[-4.5,4.5],[-0.342,-0.342],'-k',lw=1)
        ax.plot3D([0,0],[-4.5,4.5],[0.352,0.352],'-k',lw=1)
        ## walls
        ax.plot3D([-0.85,-0.85],[5,-5],[-0.9,-0.9],'-k',lw=1)
        ax.plot3D([0.85,0.85],[5,-5],[-0.9,-0.9],'-k',lw=1)
        ax.plot3D([0.85,0.85],[5,-5],[0.4,0.4],'-k',lw=1)
        ax.plot3D([-0.85,-0.85],[5,-5],[0.4,0.4],'-k',lw=1)
        ax.plot3D([-0.36,-0.36],[5,-5],[0.9,0.9],'-k',lw=1)
        ax.plot3D([0.36,0.36],[5,-5],[0.9,0.9],'-k',lw=1)
    #############################################################################################################################################
    ## Plot quiver ##############################################################################################################################
    #############################################################################################################################################
    def plot_velocityfield(self,nsteps=100,length=0.2,nlevels=10,normalize=True,linewidth=0.3,colormap='jet'):
        print("kwargs:nsteps[int],length[float],nlevels[int],normalize[boolean],linewidth[float],colormap[str]")
        fig = plt.figure(figsize=(self.x_size, self.y_size))
        ax = fig.add_subplot(111, projection='3d')
        ## data upload
        x = self.dataframe[::nsteps,self.variable_dict["x"][0]]
        y = self.dataframe[::nsteps,self.variable_dict["y"][0]]
        z = self.dataframe[::nsteps,self.variable_dict["z"][0]]
        u = self.dataframe[::nsteps,self.variable_dict["x velocity"][0]]
        v = self.dataframe[::nsteps,self.variable_dict["y velocity"][0]]
        w = self.dataframe[::nsteps,self.variable_dict["z velocity"][0]]
        vel = self.dataframe[::nsteps,self.variable_dict["velocity"][0]]
        ### mesh plot ##########################################################################################################################
        self.plot_mesh(ax=ax)
        ## create color ###################################################################################################################
        cmap = plt.get_cmap(colormap)
        color = cmap(vel)
        ## quiver plot
        quiver = ax.quiver(x, z, y, u, w, v, colors=color, length=length, normalize=normalize,
                           linewidth=linewidth,edgecolors='face')
        ## color bar
        boundaries = np.linspace(min(vel), max(vel), nlevels)
        mappable = cm.ScalarMappable(cmap=colormap)
        mappable.set_array(vel)
        fig.colorbar(mappable, shrink=0.8, ax=ax, label="Velocity [m/s]", boundaries=boundaries, location="bottom", pad=0)
        ### kwargs for plot
        ax.set_xlabel('X [m]',fontsize=18,color='red',rotation=45,labelpad=10)
        ax.set_ylabel('Z [m]',fontsize=18,color='red',rotation=15,labelpad=30)
        ax.set_zlabel('Y [m]',fontsize=18,color='red',labelpad=5)
        ax.set_xticks(np.linspace(-0.85,0.85,5))
        ax.set_yticks(np.linspace(-5,5,9))
        ax.set_zticks(np.linspace(-0.9,0.9,7))
        ax.set_xlim(-0.85,0.85)
        ax.set_ylim(-5,5)
        ax.set_zlim(-0.9,0.9)
        ax.xaxis.set_tick_params(labelsize=9,rotation=45)
        ax.yaxis.set_tick_params(labelsize=9)
        ax.zaxis.set_tick_params(labelsize=9)
        ax.dist = 8
        ax.set_title("Streamlines Plot"+f", CASE {self.archive}",fontsize=25)
        ax.set_box_aspect([1.7,10,1.8])
        #if self.save_plot:
        #    plt.savefig('images/'+name_variable+'.pdf',bbox_inches='tight')
        plt.show()
    #############################################################################################################################################
    ## Plot max regions #########################################################################################################################
    #############################################################################################################################################
    def plot_max_regions(self,variable,n_regions=3,length=0.2,nlevels=10,normalize=True,linewidth=0.3,
                         colormap='jet',save=save_plot,scatter_size=0.2):
        """Function that plots n_region number of regions where a maximum value of a varaible is found"""
        print("kwargs: variable[str],n_regions[int],length[float],nlevels[int]")
        print("normalize[boolean],linewidth[float],colormap[str],save[boolean],scatter_size[float]")
        fig = plt.figure(figsize=(self.x_size, self.y_size))
        ax = fig.add_subplot(111, projection='3d')
        ## data upload
        x = self.dataframe[:,self.variable_dict["x"][0]]
        y = self.dataframe[:,self.variable_dict["y"][0]]
        z = self.dataframe[:,self.variable_dict["z"][0]]
        var = self.dataframe[:,self.variable_dict[variable][0]]
        unit = self.variable_dict[variable][1]
        ## get regions
        zones = self.regions_max_values(x,y,z,var,length=length,n_regions=n_regions)
        ### mesh plot ##########################################################################################################################
        self.plot_mesh(ax=ax)
        ## scatter plot
        for region in range(len(zones)):
            scatterplot = ax.scatter3D(x[zones[region]],z[zones[region]],y[zones[region]],
                                       c=var[zones[region]],edgecolors='face',cmap=colormap,
                                       s=scatter_size,alpha=0.8,vmin=min(var),vmax=max(var))
        boundaries = np.linspace(min(var),max(var),nlevels)
        cb = fig.colorbar(scatterplot,shrink=0.8,ax=ax,boundaries=boundaries,
                  label=variable+" "+unit,location="bottom",pad=0)
        ### kwargs for plot ######################################################################################################################
        ax.set_xlabel('X [m]',fontsize=18,color='red',rotation=45,labelpad=10)
        ax.set_ylabel('Z [m]',fontsize=18,color='red',rotation=15,labelpad=30)
        ax.set_zlabel('Y [m]',fontsize=18,color='red',labelpad=5)
        ax.set_xticks(np.linspace(-0.85,0.85,5))
        ax.set_yticks(np.linspace(-5,5,9))
        ax.set_zticks(np.linspace(-0.9,0.9,7))
        ax.set_xlim(-0.85,0.85)
        ax.set_ylim(-5,5)
        ax.set_zlim(-0.9,0.9)
        ax.xaxis.set_tick_params(labelsize=9,rotation=45)
        ax.yaxis.set_tick_params(labelsize=9)
        ax.zaxis.set_tick_params(labelsize=9)
        ax.dist = 8
        ax.set_title("Regions of Maximum Value for "+variable+f"\n CASE {self.archive}",fontsize=25)
        ax.set_box_aspect([1.7,10,1.8])
        #if save:
        #    plt.savefig('images/'+name_variable+'.pdf',bbox_inches='tight')
        plt.show()
    #############################################################################################################################################
    ## Plot Function: mesh, variable, filtered variable #########################################################################################
    #############################################################################################################################################
    def plot(self,variable="",condition="",colormap="viridis",nlevels=10,scatter_size=0.2):
        """Plots a variable with or without a filter (range, greater or less than a value)"""
        print("kwargs: variable[str],condition[str],mesh[boolean],save[boolean],colormap[str],nlevels[int],scatter_size[float]")
        fig = plt.figure(figsize=(self.x_size, self.y_size))
        ax = fig.add_subplot(111, projection='3d')
        ### mesh plot ##########################################################################################################################
        if self.mesh_plot:
            self.plot_mesh(ax=ax)
            plot_title = "Physical Domain"
        ### single variable plot ###############################################################################################################
        if len(variable)>0 and len(condition)==0:
            ## extract data to plot
            x = self.dataframe[:,self.variable_dict["x"][0]]
            y = self.dataframe[:,self.variable_dict["y"][0]]
            z = self.dataframe[:,self.variable_dict["z"][0]]
            var = self.dataframe[:,self.variable_dict[variable][0]]
            unit = self.variable_dict[variable][1]
            ## scatter plot
            scatterplot = ax.scatter3D(x,z,y,c=var,edgecolors='face',cmap=colormap,
                                       s=scatter_size,alpha=0.8,vmin=min(var),vmax=max(var))
            boundaries = np.linspace(min(var),max(var),nlevels)
            cb = fig.colorbar(scatterplot,shrink=0.8,ax=ax,boundaries=boundaries,
                  label=variable+" "+unit,location="bottom",pad=0)
            for tick in cb.ax.get_xticklabels():
                tick.set_rotation(30)
                tick.set_fontsize(15)
            plot_title = "Plot of "+variable+" "+unit
        ### filtered variable plot #############################################################################################################
        if len(variable)>0 and len(condition)>0:
            ## extract data to plot
            var = self.dataframe[:,self.variable_dict[variable][0]]
            unit = self.variable_dict[variable][1]
            indices,plot_title = self.filter_by_condition(condition,var,variable,unit)
            ## extract data to plot
            x = self.dataframe[indices,self.variable_dict["x"][0]]
            y = self.dataframe[indices,self.variable_dict["y"][0]]
            z = self.dataframe[indices,self.variable_dict["z"][0]]
            new_var = self.dataframe[indices,self.variable_dict[variable][0]]
            unit = self.variable_dict[variable][1]
            if len(indices)==0:
                raise ValueError(f"Empty array max value={max(var)}, min value={min(var)}")
            ## scatter plot
            scatterplot = ax.scatter3D(x,z,y,c=new_var,edgecolors='face',cmap=colormap,
                                       s=scatter_size,alpha=0.8,
                                       vmin=min(new_var),vmax=max(new_var))
            boundaries = np.linspace(min(new_var),max(new_var),nlevels)
            cb = fig.colorbar(scatterplot,shrink=0.8,ax=ax,boundaries=boundaries,
                  label=variable+" "+unit,location="bottom",pad=0)
            for tick in cb.ax.get_xticklabels():
                tick.set_rotation(30)
                tick.set_fontsize(15)
        
        ### kwargs for plot ####################################################################################################################
        ax.set_xlabel('X [m]',fontsize=18,color='red',rotation=45,labelpad=10)
        ax.set_ylabel('Z [m]',fontsize=18,color='red',rotation=15,labelpad=30)
        ax.set_zlabel('Y [m]',fontsize=18,color='red',labelpad=5)
        ax.set_xticks(np.linspace(-0.85,0.85,5))
        ax.set_yticks(np.linspace(-5,5,9))
        ax.set_zticks(np.linspace(-0.9,0.9,7))
        ax.set_xlim(-0.85,0.85)
        ax.set_ylim(-5,5)
        ax.set_zlim(-0.9,0.9)
        ax.xaxis.set_tick_params(labelsize=9,rotation=45)
        ax.yaxis.set_tick_params(labelsize=9)
        ax.zaxis.set_tick_params(labelsize=9)
        ax.dist = 8
        ax.set_title(plot_title+f", CASE {self.archive}",fontsize=25)
        ax.set_box_aspect([1.7,10,1.8])
        #if save:
        #    plt.savefig('images/'+name_variable+'.pdf',bbox_inches='tight')
        plt.show()
    #############################################################################################################################################
    ## Plot Layer ###############################################################################################################################
    #############################################################################################################################################
    def plot_layer(self,variable,plane="z",value=0,ylog=False,xlog=False,colormap='viridis',delta=1e-2,N=100,
                    aspect_ratioYZ=0.4,aspect_ratioZX=0.4,nbound=6,shrink_factor=1,xysize=[10,8]):
        """Plots a surface (planes X,Y or Z) of a variable"""
        print("kwargs: vaiable[str],plane[str],value[float],ylog[boolean],xlog[boolean],colormap[str],delta[float],N[int]")
        print("aspect_ratio_YZ[float],aspect_ratio_ZX[float],nbound[int],shrink_factor[float],xysize[list[float]]")
        ## upload variables
        x = self.dataframe[:,self.variable_dict["x"][0]]
        y = self.dataframe[:,self.variable_dict["y"][0]]
        z = self.dataframe[:,self.variable_dict["z"][0]]
        v = self.dataframe[:,self.variable_dict[variable][0]]
        unit = self.variable_dict[variable][1]
        ## get surface values
        indices = self.get_surface(x,y,z,plane,value,e=delta)
        ##plot  ##############################################################################################################################
        if len(xysize)==2:
            fig = plt.subplots(1,1, figsize=(xysize[0], xysize[1]))
        else:
            fig = plt.subplots(1,1, figsize=(self.x_size, self.y_size))
        ax = plt.gca()
        ## z-plane ############################################################################################################################
        if "z" in plane.lower():
            ##data interpolation
            xi = np.linspace(min(x[indices]), max(x[indices]), N)
            yi = np.linspace(min(y[indices]), max(y[indices]), N)
            xi, yi = np.meshgrid(xi, yi)
            # Interpolate f values onto the grid
            vi = griddata((x[indices], y[indices]), v[indices], (xi, yi), method='cubic')
            ##plot 
            pcolormesh = plt.pcolormesh(xi, yi, vi, shading='auto', cmap=colormap,edgecolors='face',
                                                 vmin=min(v[indices]),vmax=max(v[indices]))
            cbar = plt.colorbar(pcolormesh,shrink=shrink_factor,
                         boundaries=np.linspace(min(v[indices]),max(v[indices]),nbound))
            cbar.set_label(variable + unit, size=18)
            ##cylinder
            if abs(value)<=4.5:
                r = 0.3475
                th = np.linspace(0,2*pi,100)
                x0 = 0
                y0 = 0.05
                plt.fill(r*np.cos(th)+x0,r*np.sin(th)+y0,color="white")
            plt.xlabel("X")
            plt.ylabel("Y")
            aspect_ratio = (max(y[indices])-min(y[indices]))/(max(x[indices])-min(x[indices]))
            ax.set_aspect(aspect_ratio)
        ## y-plane ############################################################################################################################
        elif "y" in plane.lower():
            ##data interpolation
            xi = np.linspace(min(x[indices]), max(x[indices]), N)
            zi = np.linspace(min(z[indices]), max(z[indices]), N)
            xi, zi = np.meshgrid(xi, zi)
            # Interpolate f values onto the grid
            vi = griddata((x[indices], z[indices]), v[indices], (xi, zi), method='cubic')
            ##plot 
            pcolormesh = plt.pcolormesh(xi, zi, vi, shading='auto', cmap=colormap,edgecolors='face',
                                                 vmin=min(v[indices]),vmax=max(v[indices]))
            cbar = plt.colorbar(pcolormesh,shrink=shrink_factor,
                         boundaries=np.linspace(min(v[indices]),max(v[indices]),nbound))
            cbar.set_label(variable + unit, size=18)
            ##cylinder
            if abs(value-0.05)<0.3475:
                xb = np.sqrt(0.3475**2-(value-0.05)**2)
                plt.fill([-xb,xb,xb,-xb],[-4.5,-4.5,4.5,4.5],color='white')
            plt.xlabel("X")
            plt.ylabel("Z")
            ax.set_aspect(aspect_ratioZX)
        ## x-plane ############################################################################################################################
        elif "x" in plane.lower():
            ##data interpolation
            zi = np.linspace(min(z[indices]), max(z[indices]), N)
            yi = np.linspace(min(y[indices]), max(y[indices]), N)
            zi, yi = np.meshgrid(zi, yi)
            # Interpolate f values onto the grid
            vi = griddata((y[indices],z[indices]), v[indices], (yi, zi), method='cubic')
            ##plot 
            pcolormesh = plt.pcolormesh(yi, zi, vi, shading='auto', cmap=colormap,edgecolors='face',
                                                  vmin=min(v[indices]),vmax=max(v[indices]))
            cbar = plt.colorbar(pcolormesh,shrink=shrink_factor,
                         boundaries=np.linspace(min(v[indices]),max(v[indices]),nbound))
            cbar.set_label(variable + unit, size=18)
            ##cylinder
            if abs(value)<0.3475:
                yb = np.sqrt(0.3475**2-value**2)
                plt.fill([0.05-yb,0.05+yb,0.05+yb,0.05-yb],[-4.5,-4.5,4.5,4.5],color='white')
            plt.xlabel("Y")
            plt.ylabel("Z")
            ax.set_aspect(aspect_ratioYZ)
        else:
            raise ValueError("Plane does not contain x,y,z")
        if ylog:
            plt.yscale("log")
        if xlog:
            plt.xscale("log")
    
        plt.tight_layout()
        plt.title(f"Surface Plot {plane}={value}, "+variable+unit+f", CASE {self.archive}", fontsize=16)
        #fig.suptitle('Surface Plots for '+variable+unit+f", CASE {self.archive}", fontsize=18,y=1.025)
        plt.show()
    #############################################################################################################################################
    ## Plot Layers ##############################################################################################################################
    #############################################################################################################################################
    def plot_layers(self,variable="",plane_dict={},ylog=False,xlog=False,colormap='viridis',delta=1e-2,N=100,
                    aspect_ratioYZ=0.4,aspect_ratioZX=0.4,nbound=6,shrink_factor=1):
        """plots a set of surfaces (X, Y or Z) of a variable, uses interpolation"""
        print("kwargs: vaiable[str],plane_dict[dict[plane]=value],ylog[boolean],xlog[boolean],colormap[str],delta[float],N[int]")
        print("aspect_ratio_YZ[float],aspect_ratio_ZX[float],nbound[int],shrink_factor[float]")
        ### create default dictionary with values of planes
        if len(plane_dict)==0:
            plane_dict = self.planes_dictionary
        ## recognize variable ##################################################################################################################
        if len(variable)==0:
            print("Choose Variable")
            variable = str(input())
            try:
                index = self.variable_dict[variable][0]
                unit = self.variable_dict[variable][1]
            except:
                print(self.variable_dict.keys())
                raise ValueError("Variable is not in the list")
        else:
            try:
                index = self.variable_dict[variable][0]
                unit = self.variable_dict[variable][1]
            except:
                print(self.variable_dict.keys())
                raise ValueError("Variable is not in the list")
        ## upload variables
        x = self.dataframe[:,self.variable_dict["x"][0]]
        y = self.dataframe[:,self.variable_dict["y"][0]]
        z = self.dataframe[:,self.variable_dict["z"][0]]
        v = self.dataframe[:,index]
        plane_indices = []
        ## get surface values
        for plane in list(plane_dict.keys()):
            plane_indices.append(self.get_surface(x,y,z,plane,plane_dict[plane],e=delta))
        ##plot  ##############################################################################################################################
        ## prepare plot in 3 columns #########################################################################################################
        n = len(plane_indices)
        planes = list(plane_dict.keys())
        if n//3<n/3:
            plot_rows = n//3+1
        else:
            plot_rows = n//3
        if plot_rows<2:
            plot_rows=2
        ##subplots
        fig, axes = plt.subplots(plot_rows,3, figsize=(self.x_size, self.y_size))
        index = 0
        ## get values from all dataframes to be used
        max_value = np.max(v)
        min_value = np.min(v)
        for plane in planes:
            ## sets the position of each plot
            row = (index-3)//3+1
            col = (index-3)%3
            indices = plane_indices[index]
            ## z-plane
            if "z" in plane.lower():
                ##data interpolation
                xi = np.linspace(min(x[indices]), max(x[indices]), N)
                yi = np.linspace(min(y[indices]), max(y[indices]), N)
                xi, yi = np.meshgrid(xi, yi)
                ##interpolate f values onto the grid
                vi = griddata((x[indices], y[indices]), v[indices], (xi, yi), method='cubic')
                ##plot 
                pcolormesh = axes[row,col].pcolormesh(xi, yi, vi, shading='auto', cmap=colormap,edgecolors='face',
                                                      vmin=min_value,vmax=max_value)
                                                     #vmin=min(v[indices]),vmax=max(v[indices]))
                axes[row,col].set_title("Plane Z = %1.2f"%plane_dict[plane],fontsize=12)
                #plt.colorbar(pcolormesh,ax=axes[row,col],label=variable+unit,shrink=shrink_factor,
                #             boundaries=np.linspace(min(v[indices]),max(v[indices]),nbound))

                ##cylinder
                if abs(plane_dict[plane])<=4.5:
                    r = 0.3475
                    th = np.linspace(0,2*pi,100)
                    x0 = 0
                    y0 = 0.05
                    axes[row,col].fill(r*np.cos(th)+x0,r*np.sin(th)+y0,color="white")
                    
                axes[row,col].set_xlabel("X")
                axes[row,col].set_ylabel("Y")
                aspect_ratio = (max(y[indices])-min(y[indices]))/(max(x[indices])-min(x[indices]))
                axes[row, col].set_aspect(aspect_ratio)
            ## y-plane ############################################################################################################################
            elif "y" in plane.lower():
                ##data interpolation
                xi = np.linspace(min(x[indices]), max(x[indices]), N)
                zi = np.linspace(min(z[indices]), max(z[indices]), N)
                xi, zi = np.meshgrid(xi, zi)
                ##interpolate f values onto the grid
                vi = griddata((x[indices], z[indices]), v[indices], (xi, zi), method='cubic')
                ##plot 
                pcolormesh = axes[row,col].pcolormesh(xi, zi, vi, shading='auto', cmap=colormap,edgecolors='face',
                                                      vmin=min_value,vmax=max_value)
                                                     #vmin=min(v[indices]),vmax=max(v[indices]))
                axes[row,col].set_title("Plane Y = %1.2f"%plane_dict[plane],fontsize=12)
                #plt.colorbar(pcolormesh,ax=axes[row,col],label=variable+unit,shrink=shrink_factor,
                #             boundaries=np.linspace(min(v[indices]),max(v[indices]),nbound))
                ##cylinder
                if abs(plane_dict[plane]-0.05)<0.3475:
                    value = plane_dict[plane]
                    xb = np.sqrt(0.3475**2-(value-0.05)**2)
                    axes[row,col].fill([-xb,xb,xb,-xb],[-4.5,-4.5,4.5,4.5],color='white')
                axes[row,col].set_xlabel("X")
                axes[row,col].set_ylabel("Z")
                axes[row, col].set_aspect(aspect_ratioZX)
            ## x-plane
            elif "x" in plane.lower():
                ##data interpolation
                zi = np.linspace(min(z[indices]), max(z[indices]), N)
                yi = np.linspace(min(y[indices]), max(y[indices]), N)
                zi, yi = np.meshgrid(zi, yi)
                #interpolate f values onto the grid
                vi = griddata((y[indices],z[indices]), v[indices], (yi, zi), method='cubic')
                ##plot 
                pcolormesh = axes[row,col].pcolormesh(yi, zi, vi, shading='auto', cmap=colormap,edgecolors='face',
                                                      vmin=min_value,vmax=max_value)
                                                      #vmin=min(v[indices]),vmax=max(v[indices]))
                axes[row,col].set_title("Plane X = %1.2f"%plane_dict[plane],fontsize=12)
                #plt.colorbar(pcolormesh,ax=axes[row,col],label=variable+unit,shrink=shrink_factor,
                #             boundaries=np.linspace(min(v[indices]),max(v[indices]),nbound))
                ##cylinder
                if abs(plane_dict[plane])<0.3475:
                    value = plane_dict[plane]
                    yb = np.sqrt(0.3475**2-value**2)
                    axes[row,col].fill([0.05-yb,0.05+yb,0.05+yb,0.05-yb],[-4.5,-4.5,4.5,4.5],color='white')
                axes[row,col].set_xlabel("Y")
                axes[row,col].set_ylabel("Z")
                axes[row, col].set_aspect(aspect_ratioYZ)
                
            else:
                raise ValueError("Plane does not contain x,y,z")
            if ylog:
                axes[row,col].set_yscale("log")
            if xlog:
                axes[row,col].set_xscale("log")
            index = index+1
        
        plt.tight_layout()
        cbar = fig.colorbar(pcolormesh, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.02, pad=0.04)
        cbar.set_label(f"{variable} {unit}")
        fig.suptitle('Surface Plots for '+variable+unit+f", CASE {self.archive}", fontsize=18,y=1.025)
        plt.show()

#############################################################################################
#############################################################################################
#############################################################################################

class Time_Analysis:
    """Class for upload text files that contain the time variation of certain variables
    resulting from different simulations"""
    ## folder location
    PATH = r"C:\Users\sebas\Desktop\SAPHIR"
    Folder = ""
    ## jump rows where there is no relevant information
    skiprows = 2
    ## dataset
    dataset = dict()
    ## charge dataframe and headers
    archive = ""
    dataframe = []
    headers = []
    ## plot parameters
    x_size = 10
    y_size = 8
    #############################################################################################################################################
    ## set dataframe ############################################################################################################################
    #############################################################################################################################################
    def set_dataframe(self,archive):
        """uploads information from a single datafram of the dataset"""
        self.dataframe = self.dataset[archive][1]
        self.headers = self.dataset[archive][2]
        self.archive = archive
        print(self.headers)
        return
    
    #############################################################################################################################################
    ## list out files ###########################################################################################################################
    #############################################################################################################################################
    def get_out_files(self):
        """Creates a list of all out files (text files) which fullfills the pattern conditions
        of the archives corresponding the simulations' results"""
        search_pattern = os.path.join(self.PATH, '**', f'*.out')
        all_paths = glob.glob(search_pattern, recursive=True)
        out_path = []
        out_name = []
        for path in all_paths:
            if "Time_" in path.split("\\")[-1]: #filters specific archives
                out_path.append(path) ##adds path
                name_aux = path.split("\\")[-1].split(".")[0].split("_") 
                out_name.append(name_aux[1]+"_"+name_aux[-1]) #creates new name for archive
        return out_path, out_name
    #############################################################################################################################################
    ## create dataset ###########################################################################################################################
    #############################################################################################################################################
    def create_dataset(self):
        """Creates dataset from .out files"""
        dataset = dict()
        out_path, out_names = self.get_out_files() ##names and paths of archivess
        index = 0
        for path in out_path:
            df, head = self.read_data(path) ## dataframe and headers
            dataset[out_names[index]] = [index,df,head] ##add values to dataset
            index = index+1
        print("List read out files")
        print(dataset.keys())
        return dataset
    #############################################################################################################################################
    ## read out file ############################################################################################################################
    #############################################################################################################################################
    def read_data(self,archive_PATH=""):
        """creates dataframe from archive, and adds dew point variable"""
        if len(archive_PATH)==0:
            raise ValueError("No path found")
        ## read file
        dataframe, headers = self.read_out_file(archive_PATH,skiprows=self.skiprows)
        ## create dew point variables
        max_dew_point = np.zeros(dataframe.shape[0])
        min_dew_point = np.zeros(dataframe.shape[0])
        mean_dew_point= np.zeros(dataframe.shape[0])
        ## add headers for new variables
        headers.append("min_dew")
        headers.append("max_dew")
        headers.append("mean_dew")
        ## reshape dataframe
        new_dataframe = np.zeros((dataframe.shape[0],dataframe.shape[1]+3))
        new_dataframe[:,:-3] = dataframe
        ##create dictionary
        variables_dict = {}
        for slot in headers:
            variables_dict[slot] = len(variables_dict)
        ## use dewpoint function
        variables = ['p','t','h2o']
        positions = ['min','max','mean']
        index = -3
        ## creates dew point variable
        for position in positions:
            pressure = dataframe[:,self.get_variable_index(headers=headers,variables_dict=variables_dict,variable=variables[0],position=position)]
            temperature = dataframe[:,self.get_variable_index(headers=headers,variables_dict=variables_dict,variable=variables[1],position=position)]
            h2o = dataframe[:,self.get_variable_index(headers=headers,variables_dict=variables_dict,variable=variables[2],position=position)]
            #add dew point
            new_dataframe[:,index] = Space_Analysis.dew_point(temperature,pressure,h2o)
            index = index+1
        
        #print(variables_dict.keys())
        return new_dataframe,headers
    #############################################################################################################################################
    ## read .out file ###########################################################################################################################
    #############################################################################################################################################
    ## get the headers of the archive
    @staticmethod
    def get_headers(archive,skiprows):
        """reads headers from archive and cleans the strings"""
        with open(archive,'r') as file:
            for skip in range(skiprows):
                file.readline()
            header = file.readline() #get header
        headers = list() ##list of headers
        parts = header.split('"') ## delimiter
        for index in range(1,len(parts)-1):
            if not parts[index]==" " or parts[index]=="": #clean false headers
                headers.append(parts[index])
        return headers
    ## read out file
    def read_out_file(self,archive,skiprows=skiprows,delimiter=" "):
        headers = self.get_headers(archive,skiprows=skiprows) #get headers
        dataframe = np.array(pd.read_csv(archive, delimiter=delimiter, skiprows=skiprows+1)) #read data frame
        if dataframe.shape[1]!=len(headers): #check if dimensions are equal
            raise ValueError("Columns of dataframe are not equal to number of headers")
        return dataframe,headers
    #############################################################################################################################################
    ##get timeflow variable #####################################################################################################################
    #############################################################################################################################################
    @staticmethod
    def get_timeflow(dataframe,headers):
        """finds the time variable vector"""
        count = 0 ## count if there is more than 1 flow-time or none
        index = 0 ##count loops
        for head in headers:
            if head=="flow-time": ##find time-flow
                t = dataframe[:,index]
                count = count+1
            index = index+1
        if count>1: ##check 
            print(headers)
            raise ValueError("More than 1 flow-time variable found")
        elif count==0: ##check
            print(headers)
            raise ValueError("No flow-time variable found")
        return t
    #############################################################################################################################################
    ##get variable and state: min mean or max ###################################################################################################
    #############################################################################################################################################
    @staticmethod
    def get_variable(dataframe,header,variable,state):
        """identifies max, min or mean for a variable and checks if there is only 1 
        max min or mean column for that variable"""
        count = 0 #counts if there is more 1 case in the search or none at all
        index = 0
        for head in header:
            if (variable in head) and (state in head): ##find max,min,mean variable
                var = dataframe[:,index]
                count = count+1
            index = index+1
        if count>1:#check
            raise ValueError("More than 1 variable found")
        elif count==0:#check
            raise ValueError("No variable found")
        return var
    #############################################################################################################################################
    ## gets the index of variable max,min,mean ##################################################################################################
    #############################################################################################################################################
    @staticmethod
    def get_variable_index(headers,variables_dict,variable='',position=''):
        """Gets indices of variable"""
        ### choose variable and position
        if variable=='':
            print("Choose 0:pressure, 1:temperature, 2:h2o mass fraction")
            choose = int(input())
            if choose==0:
                variable='p'
            elif choose==1:
                variable='t'
            elif choose==2:
                variable='h2o'
            else:
                raise ValueError("Chosen option not valid")
        if position=='':
            print("Choose 0:minimum, 1:maximum, 2:mean")
            choose = int(input())
            if choose==0:
                position='min'
            elif choose==1:
                position='max'
            elif choose==2:
                position='mean'
            else:
                raise ValueError("Chosen option not valid")
        ## look for selected option
        index = 0
        for slot in headers:
            if variable in slot and position in slot: #find variable and max,min, mean
                return index
            index = index+1
        raise ValueError(f"Index of {variable}-{position} not found")
    #############################################################################################################################################
    ## plot variable in time ####################################################################################################################
    #############################################################################################################################################
    def plot(self,xlogscale=[False,False,False,False],ylogscale=[False,False,False,False],
             line_width=0.1,marker_size=2,legend_loc=1):
        """plot of mean,max min for each variable in time"""
        #import data
        header=self.headers
        dataframe=self.dataframe
        ## choose x axis time
        x_axis = "flow-time"
        exception = "Time Step"
        ## dont ake time step
        index = 0
        ## find time flow variable
        t = self.get_timeflow(dataframe,header)
        index = 0
        ## figure 
        fig, axes = plt.subplots(2,2, figsize=(self.x_size, self.y_size))
        #changing unit
        constant = 1
        ## look for all cases        
        for case in header:
            if not case==exception and not case==x_axis:
                if "p" in case:
                    row = 0
                    col = 0
                    axes[row,col].set_ylabel("Pressure [MPa]")
                    if ylogscale[0]:
                        axes[row,col].set_yscale("log")
                    if xlogscale[0]:
                        axes[row,col].set_xscale("log")
                    constant=1e6
                elif "t" in case:
                    row = 0
                    col = 1
                    axes[row,col].set_ylabel("Temperature [K]")
                    if ylogscale[1]:
                        axes[row,col].set_yscale("log")
                    if xlogscale[1]:
                        axes[row,col].set_xscale("log")
                    constant=1
                elif "h2o" in case:
                    row = 1
                    col = 0
                    axes[row,col].set_ylabel("H2O Mass Fraction [-]")
                    if ylogscale[2]:
                        axes[row,col].set_yscale("log")
                    if xlogscale[2]:
                        axes[row,col].set_xscale("log")
                    constant=1
                elif "dew" in case:
                    row = 1
                    col = 1
                    axes[row,col].set_ylabel("Dew Point [°C]")
                    if ylogscale[3]:
                        axes[row,col].set_yscale("log")
                    if xlogscale[3]:
                        axes[row,col].set_xscale("log")
                    constant=1
                else:
                    raise ValueError(f"Variable out of case {case}")
                if 'min' in case:
                    marker = "x"
                elif 'max' in case:
                    marker = "^"
                elif 'mean' in case:
                    marker = "o"
                else:
                    raise ValueError("Error in case loop")
                line, = axes[row,col].plot(t,dataframe[:,index]/constant,linestyle="--",
                label=case,marker=marker,linewidth=line_width,ms=marker_size)
                axes[row,col].set_aspect('auto')
                axes[row,col].set_xlabel("Time [s]")
                axes[row,col].set_xlim(min(t),max(t))
                axes[row,col].legend(loc=legend_loc)
                axes[row,col].grid(True)
            index = index+1
        ## plot kwargs
        fig.suptitle('Time Evolution of Variables'+f", CASE {self.archive}", fontsize=18,y=1)
        plt.tight_layout()
        plt.show()
    #############################################################################################################################################
    ## plot mass fraction of h2o ################################################################################################################
    #############################################################################################################################################
    def plot_h2o(self,exception=[],xlog=False,ylog=False,location=1,line_width=1):
        """plot the mass fraction variation in time and adds a tau variable for choosing best option"""
        files = list(self.dataset.keys())
        ## filter exception files
        for exp in exception:
            files.remove(exp)
        fig = plt.subplots(1,1,figsize=(self.x_size, self.y_size))
        ## plot h2o mean for each file
        for file in files:
            header = self.dataset[file][2]
            dataframe = self.dataset[file][1]
            t = self.get_timeflow(dataframe,header)
            h2o = self.get_variable(dataframe,header,"h2o","mean")
            plt.plot(t,h2o,label=file+r", $\tau=$%1.2f%%"%(100*(1-h2o[-1]/8.57e-3)),linewidth=line_width,linestyle="-.")
        #plot kwags
        plt.xlabel("Time [s]")
        plt.ylabel("Mass Fraction of H2O [-]")
        plt.title("Mass Fraction of H2O Time Evolution")
        if xlog:
            plt.xscale("log")
        if ylog:
            plt.yscale("log")
        plt.grid(True)
        plt.legend(loc=location,fontsize=12)
        plt.show()