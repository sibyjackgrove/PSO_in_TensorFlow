
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df1=pd.read_csv("x_y_data.csv")   #Importing data
col1 = df1['X']
col2 = df1['Y']
x1 = col1.as_matrix()
y1_ = col2.as_matrix()
plt.scatter(x1,y1_,color='red')   #Plotting data


# In[3]:


#PSO parameters
c1 = 0.1  #PSO hyperparametre
c2 =0.1
P = 30   #Number of particles
N = 500 #Number of Iterations


# In[4]:


def swarm(P=10):  #Define partilces
    Wpc = tf.Variable(tf.random_normal([P])) #Current value
    bpc = tf.Variable(tf.random_normal([P]))
    fitness_pc = tf.Variable(tf.zeros([P]))
    
    Wpbest= tf.Variable(tf.random_normal([P]))
    bpbest= tf.Variable(tf.random_normal([P]))
    fitness_pbest = tf.Variable(tf.zeros([P]))
    
    Wglobal= tf.Variable(tf.random_normal([1]))
    bglobal= tf.Variable(tf.random_normal([1]))
    fitness_global = tf.Variable(tf.zeros([1]))
    
    V_W = tf.Variable(tf.zeros([P]))   #Particle velocity
    V_b = tf.Variable(tf.zeros([P]))
    return Wpc,bpc,Wpbest,bpbest,Wglobal,bglobal,V_W,V_b,fitness_pc,fitness_pbest,fitness_global

Wpc,bpc,Wpbest,bpbest,Wglobal,bglobal,V_W,V_b,fitness_pc,fitness_pbest,fitness_global =swarm(P)
print(Wglobal)


# In[5]:



x = tf.placeholder(tf.float32)
W = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
i = tf.placeholder(tf.int32)
y_ = tf.placeholder(tf.float32)


# In[6]:


#Update particles
V_W_update = V_W[i].assign(V_W[i]+(Wpbest[i] - Wpc[i])*c1*np.random.random() +(Wglobal[0] - Wpc[i])*c2*np.random.random() )
V_b_update = V_b[i].assign(V_b[i]+(bpbest[i] - bpc[i])*c1*np.random.random() +(bglobal[0] - bpc[i])*c2*np.random.random() )
Wp_update = Wpc[i].assign(V_W[i]+Wpc[i])
bp_update = bpc[i].assign(V_b[i]+bpc[i])

#Cost
y = Wpc[i]*x + bpc[i]
y_best = Wglobal[0]*x + bglobal[0]
error= tf.reduce_mean(tf.square(y- y_))
#Update fitness
update_fitness_pc = fitness_pc[i].assign(error)
update_fitness_pbest = fitness_pbest[i].assign(fitness_pc[i])
update_fitness_global = fitness_global[0].assign(fitness_pc[i])
initialize_fitness_global = fitness_global[0].assign(tf.reduce_min(fitness_pbest))

update_Wpbest = Wpbest[i].assign(Wpc[i])
update_bpbest = bpbest[i].assign(bpc[i])
update_Wglobal = Wglobal[0].assign(Wpc[i])
update_bglobal = bglobal[0].assign(bpc[i])

def f1():return [Wpbest[i].assign(Wpc[i]),bpbest[i].assign(bpc[i]),fitness_pbest[i].assign(fitness_pc[i])]
def f2():return [Wpbest[i],bpbest[i],fitness_pbest[i]]
pbest_update= tf.cond(tf.less(fitness_pc[i], fitness_pbest[i]), f1, f2)

def f3():return [Wglobal[0].assign(Wpc[i]),bglobal[0].assign(bpc[i]),fitness_global[0].assign(fitness_pc[i])]
def f4():return [Wglobal[0],bglobal[0],fitness_global[0]]
global_update= tf.cond(tf.less(fitness_pc[i], fitness_global[0]), f3, f4)


# In[7]:


start_time = time.time()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run([V_W_update,V_b_update],feed_dict={i:0})
    sess.run([Wp_update,bp_update],feed_dict={i:0})
   
    for k in range(P):     #Initialize particles
        sess.run([update_fitness_pc],feed_dict={i:k,x:x1,y_:y1_})
        sess.run([update_fitness_pbest],feed_dict={i:k})
    sess.run([initialize_fitness_global])
    print("Particle fitness:",sess.run(fitness_pc))
    print("Global fitness:",sess.run(fitness_global))
    
    for j in range(N):      #Loop over N iterations
        for k in range(P):  #Loop over P particles
            
            sess.run([V_W_update,V_b_update],feed_dict={i:k})
            sess.run([Wp_update,bp_update],feed_dict={i:k})
            sess.run([update_fitness_pc],feed_dict={i:k,x:x1,y_:y1_})
            
           
            sess.run([pbest_update],feed_dict={i:k}) 
           
            sess.run([global_update],feed_dict={i:k})
                        
            
        if j%50== 0:
            print("Iteration:",j)    
    print("Particle best fitness:",sess.run(fitness_pbest)) 
    print("Global best fitness:",sess.run(fitness_global)) 
    print("Global best particle:",sess.run([Wglobal,bglobal]))  
    
    yfinal =sess.run([y_best],feed_dict={x:x1,y_:y1_}) #storing values
print("Time taken:", (time.time() - start_time))        


# In[8]:


plt.scatter(x1,y1_,color='red')  #Plot original line
plt.scatter(x1,yfinal,color='blue')   #Plot fitted line

