# # CASE STUDY: BREAST CANCER CLASSIFICATION

# # STEP #1: PROBLEM STATEMENT

# 
# - Predicting if the cancer diagnosis is benign or malignant based on several observations/features 
# - 30 features are used, examples:
#         - radius (mean of distances from center to points on the perimeter)
#         - texture (standard deviation of gray-scale values)
#         - perimeter
#         - area
#         - smoothness (local variation in radius lengths)
#         - compactness (perimeter^2 / area - 1.0)
#         - concavity (severity of concave portions of the contour)
#         - concave points (number of concave portions of the contour)
#         - symmetry 
#         - fractal dimension ("coastline approximation" - 1)
# 
# - Datasets are linearly separable using all 30 input features
# - Number of Instances: 569
# - Class Distribution: 212 Malignant, 357 Benign
# - Target class:
#          - Malignant
#          - Benign
# 
# 
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
# 
# ![image.png](attachment:image.png)

# # STEP #2: IMPORTING DATA

# In[1]:


# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns
sns.set() # Statistical data visualization
# %matplotlib inline


# In[2]:


# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


# In[3]:


# Explore the data 
cancer


# In[4]:


# View the keys in the dictionary. What columns do we have?
cancer.keys()


# In[5]:


# Show the description section of the dataset
print(cancer['DESCR'])


# In[6]:


# Show the target classes of the dataset
print(cancer['target_names'])


# In[7]:


# Show the target results
print(cancer['target'])


# In[8]:


# Show the names of all the features in the dataset
print(cancer['feature_names'])


# In[9]:


# Show the data results
print(cancer['data'])


# In[10]:


# Show the shape of the data; 569 rows with 30 columns
cancer['data'].shape


# In[11]:


# Create data frame of the results for the target and data
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))


# In[12]:


# Explore the data frame
df_cancer.head()


# In[13]:


# Explore the end of the results
df_cancer.tail()


# In[14]:



x = np.array([1,2,3])
x.shape


# In[15]:


Example = np.c_[np.array([1,2,3]), np.array([4,5,6])]
Example.shape


# # STEP #3: VISUALIZING THE DATA

# In[16]:


# Gives us a snapshot of all the data at once. vars-variables that we want; use the first five to get a good exploration
# of the data
# hue-specifiy the target class in order to view the relationships between variables and how they vary over target
# Blue points are malignant cases; orange points are benign.
# Bigger mean radius for malignant
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )


# In[17]:


# Look at how many are malignant and how many are benign. 
# Malign around 350 and benign around 200
sns.countplot(df_cancer['target'], label = "Count") 


# In[18]:


# Create scatterplot with mean area on x; mean smoothness on y; hue-target as our target; data-the data we are using
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)


# In[19]:


# data = df_cancer_all
#sns.lmplot('mean area', 'mean smoothness', hue ='target', data = df_cancer_all, fit_reg=False)


# In[20]:


# Let's check the correlation between the variables 
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter

# Set the figuresize to a smaller size since there are 30 features
# White areas show very high correlations
plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True) 


# # STEP #4: MODEL TRAINING (FINDING A PROBLEM SOLUTION)

# In[21]:


# Let's drop the target label coloumns; to get only our input values
# axis=1 in order to drop the entire column
X = df_cancer.drop(['target'],axis=1)


# In[22]:


# Explore the inputs
X


# In[23]:


# Define the target column
y = df_cancer['target']
# Explore the target column
y


# In[24]:


from sklearn.model_selection import train_test_split
# Split our data into training and test data (80:20 split); shift tab to explore the function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)


# In[25]:


# View the shape of the X train; 455 samples with 30 features
X_train.shape


# In[26]:


# View the shape of X_test; 114 samples with 30 features
X_test.shape


# In[27]:


# View the shape of the y_train; 455 samples
y_train.shape


# In[28]:


# View the shape of y_test; 114 samples
y_test.shape


# In[29]:


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

# Create a object from SVC
svc_model = SVC()
# Fit the SVC to the Training data
svc_model.fit(X_train, y_train)


# # STEP #5: EVALUATING THE MODEL

# In[30]:


# We want general model (a model not overfitted to the training data but works for future unknown data)

# Get our models predictions on the test data, data that the model has not seen before
y_predict = svc_model.predict(X_test)

# Confusion Matrix-Columns show the true class; rows show the predicted class
# Top left shows a predicted 0 with a true value of 0; Bottom right shows a predicted 1 and a true value of 1
# Bottom left-predicted 0 but a true value of 1; False negative-a big problem; predict no problem even though there was one
# Called a type 2 error
# Top Right-predicted 1 but a true value of 0; False Positive-predict a problem but there is none. Type 1 error

cm = confusion_matrix(y_test, y_predict)
# Model is only predicting malignant tumors
# 48 type 1 errors
cm


# In[31]:


# Heat map of the confusion matrix
# annot=True-shows the actual values for the confusion matrix
sns.heatmap(cm, annot=True)


# In[32]:


print(classification_report(y_test, y_predict))


# # STEP #6: IMPROVING THE MODEL

# In[33]:


# Get the minimum value of the training data
min_train = X_train.min()
min_train


# In[34]:


# Get the range of the training data
# Subtract minimum value from the training data and get the max value
range_train = (X_train - min_train).max()
range_train


# In[35]:


# Normalize the data by subtracting the minimum values from the training data and dividing by the range (largest-minimum)
X_train_scaled = (X_train - min_train)/range_train


# In[36]:


X_train_scaled


# In[37]:


# Scatter plot of the unnormalized training data
sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)


# In[38]:


# The scatterplot of the normalized data
sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)


# In[39]:


# Get the mimimum values
min_test = X_test.min()
# Subtract minimum test values from test data  and get the max value
range_test = (X_test - min_test).max()
# Normalize the test data
X_test_scaled = (X_test - min_test)/range_test


# In[40]:


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
# Fit the model to the scaled training data
svc_model.fit(X_train_scaled, y_train)


# In[41]:


# Predict the values of the scaled test data
y_predict = svc_model.predict(X_test_scaled)
# Create the confusion matrix of the predictions of the test data
# y-test is the true values with our predictions
cm = confusion_matrix(y_test, y_predict)

# Plot the heatmap of the confusion matrix; annot=True show that it shows the true values
sns.heatmap(cm,annot=True,fmt="d")


# In[42]:


# Show the classification report to give a summary of the models performance
print(classification_report(y_test,y_predict))


# # IMPROVING THE MODEL - PART 2

# In[43]:


# Set up the grid of values that the model will try in order to find the optimal parameters
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 


# In[44]:


from sklearn.model_selection import GridSearchCV


# In[45]:


# Use grid search to search through our SVC class with the param_grid values
# verbose-the number of values that it will display
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[46]:


# fit the grid to our scaled training data with training targets
grid.fit(X_train_scaled,y_train)


# In[47]:


# Find the best parameter values
grid.best_params_


# In[48]:


grid.best_estimator_


# In[49]:


# Give the predictions using the optimal values on the scaled training data
grid_predictions = grid.predict(X_test_scaled)


# In[50]:


# Create confusion matrix on y_test with the grid_predictions values
cm = confusion_matrix(y_test, grid_predictions)


# In[51]:


# Create a heatmap of the confusion matrix with the true values
# Only 3 misclassifactions and they are all type 1 errors, so not that mad(predicted malignant but a benign)
sns.heatmap(cm, annot=True)


# In[52]:


# Output the classification report to summarize the data
print(classification_report(y_test,grid_predictions))

