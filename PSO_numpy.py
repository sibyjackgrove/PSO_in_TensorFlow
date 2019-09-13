
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def dummy_dataset():
    """A dummy dataset."""
    
    df1=pd.read_csv("x_y_data.csv")   #Importing data
    col1 = df1['X']
    col2 = df1['Y']
    x1 = col1.as_matrix()
    y1 = col2.as_matrix()
    plt.scatter(x1,y1,color='red')   #Plotting data
    
    return x1,y1

class PSONumpy():
    """PSO implemented in Numpy."""
    

    #PSO parameters
    c1 = 0.1  #PSO hyperparametre
    c2 =0.1
    P = 30   #Number of particles
    N = 500 #Number of Iterations


    def __init__(self,Xdata,Ydata,name=None):
        """Create a PSO object."""
        
        self.name = name
        
        self.Xdata = Xdata
        self.Ydata = Ydata
    
    def initialize_swarm(self,P=10):  #Define partilces
        """Initialize the particles in a PSO swar."""
        
        self.Wpc =  np.random.normal(size=P)
        self.bpc =  np.random.normal(size=P)
        self.fitness_pc = np.zeros(P)
        self.update_swarm_fitness()        
       
        self.Wpbest =  self.Wpc
        self.bpbest =  self.bpc 
        self.fitness_pbest = np.copy(self.fitness_pc)

        self.update_globalbest()

        self.V_W = np.zeros(shape=P)
        self.V_b = np.zeros(shape=P)    

#Wpc,bpc,Wpbest,bpbest,Wglobal,bglobal,V_W,V_b,fitness_pc,fitness_pbest,fitness_global =swarm(P)
#print(Wglobal)

#x = tf.placeholder(tf.float32)
#W = tf.placeholder(tf.float32)
#b = tf.placeholder(tf.float32)
#i = tf.placeholder(tf.int32)
#y_ = tf.placeholder(tf.float32)

    
    def update_pbest(self):
        """Update the particle best."""
        
        truth = np.less(self.fitness_pc,self.fitness_pbest)
        indices = np.argwhere(truth)
        
        self.Wpbest[indices] =  self.Wpc[indices]
        self.bpbest[indices] =  self.bpc[indices]
        self.fitness_pbest[indices] = self.fitness_pc[indices]
        #print(self.fitness_pc,self.fitness_pbest)        
        
    def update_globalbest(self):
        """Update the particle best."""
        
        self.fitness_global = np.amin(self.fitness_pbest)
        self.best_index = np.argmin(self.fitness_pbest)
        
        self.Wglobal =  self.Wpbest[self.best_index]
        self.bglobal =  self.bpbest[self.best_index]
    
    def update_swarm_position(self):  #Define partilces
        """Update particle velocities and positions in PSO swarm."""

        #Update particles
        
        self.V_W = self.V_W + (self.Wpbest - self.Wpc)*self.c1*np.random.random() +(self.Wglobal - self.Wpc)*self.c2*np.random.random()
        self.V_b = self.V_b + (self.bpbest - self.bpc)*self.c1*np.random.random() +(self.bglobal - self.bpc)*self.c2*np.random.random()
        
        self.Wpc = self.Wpc + self.V_W 
        self.bpc = self.bpc + self.V_b
   
    def update_swarm_fitness(self):
        """Update fitness of the swarm."""
        
        for i,(Wpi,bpi) in enumerate(zip(self.Wpc,self.bpc)):
            
            self.fitness_pc[i] = self.fitness_function(Wpi,bpi)            
    
    def PSO_algorithm(self,epochs=10):
        """PSO algorithm."""
        
        
        start_time = time.time()
        for i in range(epochs):
            self.update_swarm_position()
            self.update_swarm_fitness()
            
            self.update_pbest()
            self.update_globalbest()
           
            print('Global best:Index:{},Fitness:{}'.format(self.best_index,self.fitness_global))
            
            if i%5== 0:
                print("Iteration:",i)
                print('Particle fitness:{}'.format(self.fitness_pc))
                print("Particle best fitness:{}".format(self.fitness_pbest))             
        
        self.plot_fit(self.Wglobal,self.bglobal)
        print("Time taken:{}".format(time.time() - start_time))        
        
        
    def fitness_function(self,Wpi,bpi):
        """The objective function."""        
        
        Ypredicted = Wpi*self.Xdata + bpi
        
        return np.mean(np.square(Ypredicted - self.Ydata))        
    
    def plot_fit(self,Wp,bp):
        """Plot fit."""
        
        Ypredicted = Wp*self.Xdata + bp
        
        plt.scatter(self.Xdata,self.Ydata,color='red')  #Plot original line
        plt.scatter(self.Xdata,Ypredicted,color='blue')   #Plot fitted line        

