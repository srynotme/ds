
# def RemoveOutlier(df, var):
#     Q1=df[var].quantile(0.25)
#     Q3=df[var].quantile(0.75)
#     IQR =Q3-Q1
#     high, low=Q3+1.5*IQR, Q1-1.5*IQR

#     df=df[((df[var] >= low) & (df[var] <=high))]
#     return df
# def DisplayOutlier(df, msg):
#     fig,axes=plt.subplots(1,2)
#     fig.suptitle(msg)
#     sns.boxplot(data=df, x="Age", ax=axes[0])
#     sns.boxplot(data = df, x="EstimatedSalary", ax=axes[1])
#     fig.tight_layout()
#     plt.show()

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("/content/Social_Network_Ads.csv")
print(df)

print(df.info())
print(df.size)
print(df.shape)
print(df.columns)
print(df.head())
print(df.tail())
print(df.sample())
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())
print(df.isna().sum())


#display correlation heatmap
sns.heatmap(df.corr(), annot=True)
plt.show()


l = ['Age', 'EstimatedSalary', 'Purchased']
for i in l:
    sns.boxplot(df[i])
    plt.suptitle(i)
    plt.show()

# #Finding and removing outliers
# print("Finding and removing outliers: ")
# DisplayOutlier(df, "Before removing Outliers:")
# print("Identifying overall outliers in Column Name variables")
# df = RemoveOutlier(df, "Age")
# df = RemoveOutlier(df,"EstimatedSalary")
# DisplayOutlier(df,"After removing Outliers")


#split data into input output
x = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import classification_report
print("classification_report: \n",classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion metrix \n",cm)

sns.heatmap(cm, annot=True, linewidths=.3)
plt.show()
