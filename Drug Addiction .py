#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r"E:\Drug addiction\Data set\Dataset.csv")


# In[3]:


df.head()


# In[85]:


df.info()


# In[86]:


df.describe()


# In[4]:


x=df.drop(['Date'],axis=1)
x=x.dropna()
y = x['addiction_ratio']
x = x.drop(['addiction_ratio'],axis=1)


# In[5]:


x


# In[6]:


from sklearn.preprocessing import LabelEncoder


# In[7]:


le =LabelEncoder()


# In[8]:


x['smoker']= le.fit_transform(x['smoker'])


# In[9]:


x['smoker'].unique()


# In[10]:


x['food_prob']= le.fit_transform(x['food_prob'])


# In[11]:


x['food_prob'].unique()


# In[12]:


x['sleeping']= le.fit_transform(x['sleeping'])


# In[13]:


x['sleeping'].unique()


# In[14]:


x['depression']= le.fit_transform(x['depression'])


# In[15]:


x['depression'].unique()


# In[16]:


x['confuision']= le.fit_transform(x['confuision'])


# In[17]:


x['confuision'].unique()


# In[18]:


x['forget']= le.fit_transform(x['forget'])


# In[19]:


x['forget'].unique()


# In[20]:


x.head(1)


# In[21]:


x['relation']= le.fit_transform(x['relation'])


# In[22]:


x['relation'].unique()


# In[23]:


x['weight_loss']= le.fit_transform(x['weight_loss'])


# In[24]:


x['weight_loss'].unique()


# In[25]:


x['illness_other']= le.fit_transform(x['illness_other'])


# In[26]:


x['illness_other'].unique()


# In[27]:


x['economic']= le.fit_transform(x['economic'])


# In[28]:


x['economic'].unique()


# In[29]:


x.head(1)


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[32]:


X_train.shape


# In[33]:


X_test.shape


# In[34]:


y_train.shape


# In[35]:


y_test.shape


# In[36]:


from sklearn.linear_model import LogisticRegression


# In[37]:


logmodel=LogisticRegression()


# In[38]:


logmodel.fit(X_train,y_train)


# In[39]:


predictions = logmodel.predict(X_test)


# In[40]:


from sklearn.metrics import classification_report


# In[41]:


classification_report(y_test,predictions)


# In[42]:


from sklearn.metrics import confusion_matrix


# In[43]:


confusion_matrix(y_test,predictions)


# In[44]:


from sklearn.metrics import accuracy_score


# In[45]:


accuracy_score(y_test,predictions)


# In[46]:


logmodel.score(X_test,y_test)


# In[47]:


from sklearn.naive_bayes import GaussianNB


# In[48]:


logmodel =GaussianNB()


# In[49]:


logmodel.fit(X_train, y_train)


# In[50]:


predictions = logmodel.predict(X_test)


# In[51]:


from sklearn.metrics import classification_report


# In[52]:


classification_report(y_test,predictions)


# In[53]:


from sklearn.metrics import confusion_matrix


# In[54]:


confusion_matrix(y_test,predictions)


# In[55]:


from sklearn.metrics import accuracy_score


# In[56]:


accuracy_score(y_test,predictions)


# In[57]:


from sklearn import tree


# In[58]:


logmodel = tree.DecisionTreeClassifier()


# In[59]:


logmodel.fit(X_train, y_train)


# In[60]:


predictions = logmodel.predict(X_test)


# In[61]:


from sklearn.metrics import accuracy_score


# In[62]:


accuracy_score(y_test,predictions)


# In[63]:


from sklearn.metrics import classification_report


# In[64]:


classification_report(y_test,predictions)


# In[65]:


from sklearn.metrics import confusion_matrix


# In[66]:


confusion_matrix(y_test,predictions)


# In[67]:


from sklearn.ensemble import AdaBoostClassifier


# In[68]:


from sklearn.datasets import make_classification


# In[69]:


x_train, y_train = make_classification(n_samples=1000, n_features=4,
n_informative=2, n_redundant=0,
random_state=0, shuffle=False)


# In[70]:


clf = AdaBoostClassifier(n_estimators=100, random_state=0)


# In[71]:


clf.fit(x_train, y_train)


# In[72]:


clf.score(x_train, y_train)


# In[73]:


from sklearn.metrics import classification_report


# In[74]:


classification_report(y_test,predictions)


# In[75]:


from sklearn.metrics import confusion_matrix


# In[76]:


confusion_matrix(y_test,predictions)


# In[77]:


from sklearn.neural_network import MLPClassifier


# In[78]:


anna = MLPClassifier(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(5, 2), random_state=1)


# In[79]:


anna.fit(x_train, y_train)


# In[80]:


anna.score(x_train, y_train)


# In[81]:


from sklearn.metrics import classification_report


# In[82]:


classification_report(y_test,predictions)


# In[83]:


from sklearn.metrics import confusion_matrix


# In[84]:


confusion_matrix(y_test,predictions)


# In[93]:


import seaborn as sns
from matplotlib import pyplot as plt 


# In[94]:


sns.heatmap(x, vmin=50, vmax=100)


# In[115]:


plt.xlabel('Age')
plt.ylabel('')
plt.title('Smoker ratio')
plt.scatter(df.smoker,df.age,color='black',linewidth=5, linestyle='dotted')


# In[116]:


plt.xlabel('Age')
plt.ylabel('')
plt.title('Addition ratio')
plt.scatter(df.age,df.addiction_ratio,color='black',linewidth=5, linestyle='dotted')


# In[117]:


x.head(1)


# In[118]:


plt.xlabel('Age')
plt.ylabel('')
plt.title('Depression ratio')
plt.bar(df.age,df.depression,color='red',linewidth=5, linestyle='dotted')


# In[120]:


plt.xlabel('Age')
plt.ylabel('')
plt.title('Econimic Ratio')
plt.bar(df.age,df.economic,color='blue',linewidth=5, linestyle='dotted')


# In[126]:


a = ['ANN','Logistic Regresion','Decision Tree','AdaBoostClassifier','GaussianNB']


# In[127]:


b=['96','97','98','98','99']


# In[132]:


plt.bar(b,a)
plt.ylabel("Algorithms")
plt.xlabel("Accuracy")
plt.title("Algorithms Accuracy Graphs")
plt.show()


# In[134]:


plt.scatter(b,a)
plt.ylabel("Algorithms")
plt.xlabel("Accuracy")
plt.title("Algorithms Accuracy Graphs")
plt.show()


# In[135]:


plt.plot(b,a)
plt.ylabel("Algorithms")
plt.xlabel("Accuracy")
plt.title("Algorithms Accuracy Graphs")
plt.show()


# In[ ]:




