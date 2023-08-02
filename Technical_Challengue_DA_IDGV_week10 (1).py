#!/usr/bin/env python
# coding: utf-8

# ## Technical Challenge DA✅

# Emojis: ✅⚠️⁉️➡️▶️⏸️🟡🔴🥳👀🙌🏻🚀🤯

# In[76]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics


from sklearn import datasets # sklearn comes with some toy datasets to practice
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot
from sklearn.metrics import silhouette_score
from PIL import Image
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

##############################################
from sklearn.linear_model import LogisticRegression




get_ipython().run_line_magic('matplotlib', 'inline')


# ### You should aim for:
# Exploratory data analysis✅
# 
# Get to know the domain✅
# 
# Explore your data✅
# 
# Clean your data✅
# 
# Take a look and find connections between data✅

# In[4]:


df = pd.read_csv("measurements.csv")


# In[5]:


df


# ### Second Data Frame

# In[34]:


df01 = pd.read_excel("measurements2.xlsx")


# In[33]:


df01 


# # ⚠️STAGE 01

# ### Exploratory data analysis / Cleanning data

# In[13]:


df.info()


# In[16]:


column_value_counts = df.count()


# ### ➡️ Converting object values into numerical values

# In[39]:


columns_to_convert = ["distance", "consume","temp_inside"]

for column in columns_to_convert:
    df[column] = pd.to_numeric(df[column], errors='coerce', downcast='integer')


# In[19]:


df.info()


# In[17]:


column_value_counts


# ### ➡️  Removing columns with more thant 70% of Null values

# In[35]:


columns_to_delete = ["specials", "refill liters", "refill gas"]
df = df.drop(columns=columns_to_delete)


# In[21]:


df


# In[ ]:





# In[40]:


column_means = df.mean()

df.fillna(column_means, inplace=True)


# In[41]:


df


# ### ➡️  Counting Values

# In[42]:


df['gas_type'].value_counts()


# In[43]:


df['rain'].value_counts()


# In[44]:


df['sun'].value_counts()


# In[45]:


df['AC'].value_counts()


# In[46]:


df['distance'].value_counts()


# In[48]:


column_value_counts


# In[49]:


df.info()


# # ⚠️STAGE 02

# ## Preparing our data for our Linear Regression Model👀

# ### ➡️Splitting up our data into numerical and categorical

# In[50]:


numerical_df = df.select_dtypes(include=[np.number])
categorical_df = df.select_dtypes(include=['object'])


# In[51]:


numerical_df = numerical_df.reset_index(drop=True) 


# In[52]:


numerical_df


# ### To be able to check the correlation of our variable we are goint to plot the Hitmap

# In[56]:


corr= numerical_df.loc[:,['distance','speed', 'temp_inside',"temp_outside"]].corr()


# In[57]:


matrix = np.triu(corr)
np.fill_diagonal(matrix,False)
sns.heatmap(corr, annot=True, mask=matrix)


# ### ✅*Conclusion*: we do not need to drop any independent variable because these do not have a high correlation between them

# In[58]:


categorical_df


# ### ➡️ Our variable "gas_type" is Nominal ( not Hierarque)
# We can use dummys to do the transformation.

# In[59]:


dummy_nominals = ["gas_type"]
categorical_df = pd.get_dummies(categorical_df, columns=dummy_nominals)


# In[60]:


categorical_df 


# ### ➡️ Concatenating our data

# In[61]:


df_model = pd.concat([numerical_df, categorical_df], axis=1)


# In[62]:


df_model 


# In[132]:


df_model.to_csv("analysis_best_gas.csv",index=False)


# ### ➡️ Splitting the data into independent variables and dependent variable

# In[119]:


X=df_model[['distance', 'speed', 'temp_inside', 'temp_outside', 'AC',
       'rain', 'sun', 'gas_type_E10','gas_type_SP98']]
y=df_model['consume']


# ### ➡️ Splitting the data into training data and testing data

# In[120]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)


# ## ⚠️ **Checklist**
# 
# 1.Tranform our scales using min and max✅
# 
# 2.To apply the linear regression✅
# 
# 3.Getting the predictions✅
# 
# 4.Evaluate the Model✅
# 
# 5.Classification Report✅
# 
# 6.Confussion Matriz✅

# In[121]:


#1


scaler = MinMaxScaler()
scaler.fit(X_train)
X_scaler_train= scaler.transform(X_train)
X_scaler_test= scaler.transform(X_test)

#2

model_lr = LinearRegression()
model_lr.fit(X_scaler_train, y_train)

#3

y_predictive = model_lr.predict(X_scaler_test)


# In[122]:


# 4
mse = mean_squared_error(y_test, y_predictive)
r2 = r2_score(y_test, y_predictive)


# In[123]:


mse


# In[124]:


r2


# ### ➡️  Coefficients Analysis

# In[125]:


coefficients = model_lr.coef_


# In[126]:


coefficients 


# In[127]:


a={"independent variables" : ['distance', 'speed', 'temp_inside', 'temp_outside', 'AC',
       'rain', 'sun', 'gas_type_E10','gas_type_SP98'],
   'coefficients':[1.10635062,  -1.8328791,  -0.97811411,  -1.2708804, 0.42096231,
         0.62794039, -0.06115135,  0.04191564,  -0.04191564]}


# In[128]:


coefficients_analysis = pd.DataFrame(a)


# In[129]:


coefficients_analysis


# In[130]:


df_orden_importances = coefficients_analysis.sort_values(by='coefficients', ascending=False)

# Crear el gráfico de barras utilizando Seaborn
plt.figure(figsize=(10, 8))
sns.barplot(x='coefficients', y='independent variables', data=df_orden_importances)
plt.xlabel('coefficients')
plt.ylabel('Independent Variables')
plt.title('Importance of variables in the model')

# Mostrar el gráfico
plt.show()


# In[ ]:





# In[131]:


distance = int(input("Enter distance: "))
rain = int(input("is the day going to be rainy? (1 for yes, 0 for no): "))
sun = int(input("Is the day going to be sunny?(1 for yes, 0 for no): "))
speed = int(input("what is the speed?: "))
ac = int(input("Are you using air-conditioning? (1 for yes, 0 for no): "))
e10 = int(input("Are you using E10? (1 for yes, 0 for no): "))
sp98 = int(input("Are you using E10? (1 for yes, 0 for no): "))
temp_inside = int(input("What was the temperature inside the car?: "))
temp_outside = int(input("What was the temperature outside the car?: "))



X_example = pd.DataFrame({"distance":distance,"rain":rain,"sun":sun,"speed":speed,
                          "ac":ac,"e10":e10,
                          "sp98":sp98,"temp_inside":temp_inside,
                          "temp_outside":temp_outside,}, index=[0])






X_example= scaler.transform(X_example)

y_example= model_lr.predict(X_example)

print("Consume:", int(y_example))


# In[ ]:





# ### ---------------------------------------------------------------------------------------------------------------------------------------------------

# ### ---------------------------------------------------------------------------------------------------------------------------------------------------

# ## dont pay atention in the following code!

# # ⚠️Using Logistic Regression Model!

# ## Preparing our data for our Logistic Regression Model👀

# In[94]:


categorical_df_lr = df.select_dtypes(include=['object'])


# In[95]:


categorical_df_lr


# In[96]:


df_model_lr = pd.concat([numerical_df, categorical_df_lr], axis=1)


# In[97]:


df_model_lr


# In[ ]:





# In[102]:


X_lr=df_model_lr[['distance', 'speed', 'temp_inside', 'temp_outside', 'AC','rain', 'sun','consume']]

y_lr=df_model_lr['gas_type']


# In[ ]:





# In[103]:


X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X,y, test_size = 0.3, random_state = 42)


# ## ⚠️ Let's check what happend with our linear regression before balancing  our data

# In[106]:


#1
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train_lr)

X_scaler_train_lr= scaler.transform(X_train_lr)

X_scaler_test_lr= scaler.transform(X_test_lr)


# In[107]:



#2

model_unbalenced = LogisticRegression()
model_unbalenced.fit(X_scaler_train_lr, y_train_lr)

#3

y_predictive_lr = model_unbalenced.predict(X_scaler_test_lr)

#4

accuracy = accuracy_score(y_test_lr, y_predictive_lr)

print("Model Accuracy: {:.2f}%".format(accuracy * 100))


# ###  ➡️ Balancing data using: Synthetic Minority Over-sampling Technique (SMOTE)

# In[108]:


from imblearn.over_sampling import SMOTE

# Create the SMOTE oversampling object
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_smote_train, y_smote_train = smote.fit_resample(X_scaler_train_lr, y_train_lr)


# ### Let's check the performance of our model using SMOTE to balance our data

# In[113]:


#2

model_smote = LogisticRegression()
model_smote.fit(X_smote_train, y_smote_train)

#3

y_predictive_using_smote = model_smote.predict(X_scaler_test_lr)

#4

accuracy = accuracy_score(y_test, y_predictive_using_smote)

print("Model Accuracy: {:.2f}%".format(accuracy * 100))


# In[112]:





# In[115]:


residuals= pd.DataFrame({"y-pred":y_predictive_using_smote,"y-test":y_test_lr})

residuals.reset_index(drop=True,inplace=True)
residuals.to_csv("confusion_matrix.csv",index=False)

rep = classification_report(y_test,y_predictive_using_smote)
print(rep)


# In[116]:


cm = confusion_matrix(y_test_lr, y_predictive_using_smote, labels=model_smote.classes_)
color = 'white'
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_smote.classes_)
disp.plot()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




