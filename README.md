![image](https://github.com/user-attachments/assets/3a3c4add-e5d1-4be9-854f-c12d3ca956a0)# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

~~~

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()

~~~

![image](https://github.com/user-attachments/assets/e1e4aa84-73b9-4a68-940d-a7afd9a4ea30)

~~~

df.dropna()

~~~

![image](https://github.com/user-attachments/assets/297ad23c-e624-4cf9-b78c-24bb29718cff)

~~~

max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals

~~~

![image](https://github.com/user-attachments/assets/73cc064e-4813-4398-91f7-dd28dec6baa1)

~~~

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)

~~~

![image](https://github.com/user-attachments/assets/861cfc87-8ea2-4788-8150-099a0e16214a)

~~~

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)

~~~


![image](https://github.com/user-attachments/assets/4cd58545-9f00-491d-8645-be9bf3a3e728)

~~~

df2=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2

~~~

![image](https://github.com/user-attachments/assets/a3dd3e43-f61e-4e0a-aec0-282ccd36fe21)

~~~

df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3

~~~

![image](https://github.com/user-attachments/assets/2e389278-1b01-4319-b7b3-810faced3c78)

~~~

df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()

~~~

![image](https://github.com/user-attachments/assets/252842e9-4dc7-454f-932a-1651c1100712)

~~~

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

~~~

![image](https://github.com/user-attachments/assets/9f00e67f-372e-470f-b3ca-f319d0600edb)

~~~

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

~~~

![image](https://github.com/user-attachments/assets/cc83bb5f-38bb-4565-8949-c4e69b6c460e)

~~~

chip2,p, _, _=chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chip2}")
print(f"P-value: {p}")

~~~

![image](https://github.com/user-attachments/assets/2aa2f591-f1f2-475c-bed4-46f5cc54d44d)

~~~

import pandas as pd 
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif 

data = { 
'Feature1': [1, 2, 3, 4, 5], 
'Feature2': ['A', 'B', 'C', 'A', 'B'], 
'Feature3': [0, 1, 1, 0, 1], 
'Target': [0, 1, 1, 0, 1] 
} 
df = pd.DataFrame(data) 

x= df[['Feature1', 'Feature3']] 
y = df['Target'] 
 
selector = SelectKBest(score_func=mutual_info_classif, k=1) 
X_new = selector.fit_transform(x, y)

selected_feature_indices = selector.get_support(indices=True) 


selected_features = X.columns[selected_feature_indices] 
print("Selected Features:") 
print(selected_features)

~~~

![image](https://github.com/user-attachments/assets/80513658-01d1-4342-8802-856e2213fa24)


# RESULT:
      Thus,The given data is read and performed Feature Scaling and Feature Selection process and saved the data to a file.
