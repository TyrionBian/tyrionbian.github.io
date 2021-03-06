---
layout:            post
title:             "More About Model Features"
date:              2019-10-16
tag:               Machine learning
category:          Machine Learning
author:            tianliang
---
## More About Model Features

Machine learning models map a set of data inputs (called features) to predictors or target variables.
The goal of this process is for the model to learn the patterns or mappings between these inputs and the target variable, 
so that given new data where the target is unknown, the model can accurately predict the target variable.

For any given data set, we hope to develop a model that can make predictions with the highest accuracy. 
In machine learning, there are many levers that affect the performance of the model. 

Generally, these contents include:

- Algorithm selection.
- The parameters used in the algorithm.
- The quantity and quality of the data set.
- Features for training models.

Usually in a dataset, a given feature set in its original form cannot provide enough or optimized information to train a performance model. 
In some cases, it may be useful to delete unnecessary or conflicting features, which is called feature selection.

In other cases, if we transform one or more features into different representations to provide better information to the model,
we can improve model performance, which is called feature engineering.

### Feature Selection

In many cases, the most predictable model cannot be derived using all available features in the data set.
Depending on the type of model used, the size of the data set and various other factors (including too much functionality) may reduce the performance of the model.

Feature selection has three main goals:

- Improving the model's ability to predict the accuracy of new data.
- Reduce computing costs.
- Produce a more interpretable model.

There are many reasons that can cause you to delete certain features without using others. 
This includes the relationship between the elements, whether the statistical relationship with the target variable exists or is important enough, 
or the value of the information contained in the element.

### Manual feature selection

There are various reasons why you might want to remove features from the training phase. 
These include:
- A feature highly related to another feature in the data set. If this is the case, then both features essentially provide the same information. Some algorithms are sensitive to related features.
Provides little or no information. An example is a features where most examples have the same value.
- Elements that have almost no statistical relationship with the target variable.
- Features can be selected by performing data analysis before or after training the model. The following are some common techniques for manually performing feature selection.

Feature selection can be performed manually by analyzing the data set before and after training, or by automatic statistical methods.

#### Correlation plot

One manual technique for performing feature selection is to create visualizations that plot the relevant metrics for each feature in the dataset. [Seaborn](https://seaborn.pydata.org/) is a good python library. 
The following code generates a correlation diagram of features in the breast cancer data set available from the scikit-learn API.

```python
# library imports
import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
# load the breast_cancer data set from the scikit-learn api
breast_cancer = load_breast_cancer()
data = pd.DataFrame(data=breast_cancer['data'], columns = breast_cancer['feature_names'])
data['target'] = breast_cancer['target']
data.head()
# use the pands .corr() function to compute pairwise correlations for the dataframe
corr = data.corr()
# visualise the data with seaborn
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.set_style(style = 'white')
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(10, 250, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, 
        square=True,
        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
```

In the final visualization, we can identify some closely related features. Therefore, we may want to delete some of these features and some features with very low correlation with the target variable. We may also want to delete these features.

<figure>
<img src="{{ "/images/1_OOI-G2W1jxh8SGcHWwx_ZQ.png" | absolute_url }}" />
<figcaption>Correlation plot</figcaption>
</figure>

#### Feature importances

Once we have trained the model, we can perform further statistical analysis to understand the impact of features on the model output and determine the most useful features based on this.

There are many tools and techniques that can be used to determine feature importance. Some technologies are unique to specific algorithms, while others can be applied to various models and are called agnostic models.

To illustrate feature importance, I will use the built-in feature importance method for the random forest classifier in scikit-learn. The following code fits the classifier and creates a graph showing the importance of the features.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Spliiting data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.20, random_state=0)
# fitting the model
model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
# plotting feature importances
features = data.drop('target', axis=1).columns
importances = model.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10,15))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
```

