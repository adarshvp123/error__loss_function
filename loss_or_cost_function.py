#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #numpy helps to do vector functions easily


# In[2]:


y_predicted = np.array([1,1,0,0,1])
y_true = np.array([0.30,0.7,1,0,0.5])


# In[3]:


#Implement Mean Absolute Error


# In[4]:


def mae(y_predicted, y_true):
    total_error = 0
    for yp, yt in zip(y_predicted, y_true):
        total_error += abs(yp - yt)
    print("Total error is:",total_error)
    mae = total_error/len(y_predicted)
    print("Mean absolute error is:",mae)
    return mae


# In[5]:


#The code you provided defines a Python function mae(y_predicted, y_true) that calculates the Mean Absolute Error (MAE) between two sets of values, y_predicted (the predicted values) and y_true (the true or actual values). Here's a breakdown of the code:

#total_error: This variable is initialized to 0 and is used to accumulate the absolute errors between the predicted values and true values.

#for yp, yt in zip(y_predicted, y_true):: This loop iterates through pairs of predicted and true values. It uses the zip function to pair corresponding elements from the two lists.

#total_error += abs(yp - yt): Inside the loop, the absolute difference between the predicted value (yp) and the true value (yt) is calculated and added to the total_error. This step calculates the sum of absolute errors between all corresponding values.

#mae = total_error / len(y_predicted): After the loop, the mean absolute error (MAE) is computed by dividing the total_error by the total number of predictions, which is given by the length of y_predicted.

#return mae: The function returns the computed MAE.

#print("Total error is:", total_error) and print("Mean absolute error is:", mae): These lines print the total error and the calculated MAE to the console for debugging and informational purposes.


# In[6]:


mae(y_predicted, y_true) #calculates mean absolute error


# In[11]:


#Total error is: 2.5 means total error sum TE=e1+e2+e3..en


# In[12]:


#Implement same thing using numpy in much easier way


# In[13]:


np.abs(y_predicted-y_true)#abs for  absolute error


# In[14]:


np.mean(np.abs(y_predicted-y_true))# for mean of absolute error


# In[16]:


def mae_np(y_predicted, y_true):
    return np.mean(np.abs(y_predicted-y_true))


# In[17]:


mae_np(y_predicted, y_true)


# In[18]:


#Implement Log Loss or Binary Cross Entropy


# In[19]:


np.log([0])


# In[20]:


epsilon = 1e-15 # to opproximate  0 to 0.0000000001


# In[21]:


np.log([1e-15])


# In[22]:


y_predicted


# In[23]:


y_predicted_new = [max(i,epsilon) for i in y_predicted]
y_predicted_new
#approximated matrix for 0


# In[24]:


1-epsilon # to opproximate 1 to 0.9999 


# In[25]:


y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
y_predicted_new


# In[26]:


y_predicted_new = np.array(y_predicted_new)


# In[27]:


np.log(y_predicted_new)


# In[28]:


-np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))


# In[31]:


#Doing same as above in simpler way
def log_loss(y_true, y_predicted):
    y_predicted_new = [max(i,epsilon) for i in y_predicted]
    y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))


# In[32]:


log_loss(y_true, y_predicted)


# In[ ]:




