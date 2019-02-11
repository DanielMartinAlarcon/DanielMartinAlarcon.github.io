---
title: Goundwater Arsenic Revisited
subtitle: Predictive analytics on messy real-world data
image: /img/7_groundwater-arsenic/poison.png
---


# Assignment
I'm actually going to work with a new dataset that I've been meaning to parse.  It is mostly numerical, but it does contain one important categorical variable.  The objective is to find the variables that best predict the arsenic content in water samples from the city of Durango, Mexico.  I'm going to format this one for publication.

# Predictors of arsenic content in goundwater
\[Details about where the data came from\]


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import category_encoders as ce
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

pd.set_option('display.max_columns', None)  # Unlimited columns
```

## Data cleanup


```python
df = pd.read_csv('water_samples.csv', index_col=0)
```


```python
# Initial state of the dataframe
print(df.shape)
df.head()
```

    (146, 22)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Municipio</th>
      <th>Localidad</th>
      <th>Coordenadas</th>
      <th>Unnamed: 3</th>
      <th>Muestra</th>
      <th>FECHA DE MUESTREO</th>
      <th>pH</th>
      <th>Conductividad (μs/cm)</th>
      <th>As (μg/L)</th>
      <th>Flúor (mg/L)</th>
      <th>Na+ (mg/L)</th>
      <th>K+    (mg/L)</th>
      <th>Fe+ (mg/L)</th>
      <th>Ca+ (mg/L)</th>
      <th>Mg+ (mg/L)</th>
      <th>NO3- (mg/L)</th>
      <th>Cl- (mg/L)</th>
      <th>CO3-2 (mg/L)</th>
      <th>HCO3- (mg/L)</th>
      <th>Alcalinidad total                (mg CaCO3/L)</th>
      <th>SO4</th>
      <th>Tipo de Agua</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Longitud</td>
      <td>Latitud</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Durango</td>
      <td>El Nayar</td>
      <td>-104.69545</td>
      <td>23.96292</td>
      <td>60.0</td>
      <td>2017-08-08</td>
      <td>8.140</td>
      <td>337.0</td>
      <td>61.5</td>
      <td>3.15</td>
      <td>40.9220</td>
      <td>1.4075</td>
      <td>0.007</td>
      <td>15.0015</td>
      <td>0.2675</td>
      <td>3.25</td>
      <td>4.140</td>
      <td>0.0</td>
      <td>97.0</td>
      <td>97.0</td>
      <td>70.615</td>
      <td>BICARBONATADA SODICA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Durango</td>
      <td>Sebastián Lerdo de Tejada</td>
      <td>-104.64026</td>
      <td>23.95718</td>
      <td>61.0</td>
      <td>2017-08-08</td>
      <td>8.110</td>
      <td>406.0</td>
      <td>38.5</td>
      <td>2.60</td>
      <td>45.8850</td>
      <td>0.6100</td>
      <td>NaN</td>
      <td>20.1530</td>
      <td>0.0645</td>
      <td>2.10</td>
      <td>2.040</td>
      <td>0.0</td>
      <td>122.0</td>
      <td>122.0</td>
      <td>79.445</td>
      <td>BICARBONATADA SODICA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Durango</td>
      <td>Felipe Ángeles</td>
      <td>-104.55661</td>
      <td>23.93505</td>
      <td>62.0</td>
      <td>2017-08-08</td>
      <td>8.375</td>
      <td>384.1</td>
      <td>26.5</td>
      <td>1.40</td>
      <td>38.5360</td>
      <td>6.3665</td>
      <td>NaN</td>
      <td>21.8090</td>
      <td>1.1890</td>
      <td>1.35</td>
      <td>2.325</td>
      <td>0.0</td>
      <td>140.0</td>
      <td>140.0</td>
      <td>53.730</td>
      <td>BICARBONATADA SODICA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Durango</td>
      <td>Villa Montemorelos</td>
      <td>-104.48167</td>
      <td>23.99177</td>
      <td>63.0</td>
      <td>2017-08-08</td>
      <td>8.500</td>
      <td>557.5</td>
      <td>23.5</td>
      <td>1.20</td>
      <td>31.6805</td>
      <td>6.7350</td>
      <td>NaN</td>
      <td>32.8330</td>
      <td>6.5825</td>
      <td>4.40</td>
      <td>4.680</td>
      <td>0.0</td>
      <td>206.5</td>
      <td>206.5</td>
      <td>60.245</td>
      <td>BICARBONATADA CALCICA Y/O MAGNESICA</td>
    </tr>
  </tbody>
</table>
</div>




```python
# I rename all columns with simpler English names
df2 = df.rename(
    {'Municipio':'municipality',
     'Localidad':'town',
     'Coordenadas':'longitude',
     'Unnamed: 3':'latitude',
     'Muestra':'id',
     'FECHA DE MUESTREO ':'sampling_date',
     'pH':'pH',
     'Conductividad (μs/cm)':'conductivity',
     'As (μg/L)':'As',
     'Flúor (mg/L)':'F',
     'Na+ (mg/L)':'Na',
     'K+    (mg/L)':'K',
     'Fe+ (mg/L)':'Fe',
     'Ca+ (mg/L)':'Ca',
     'Mg+ (mg/L)':'Mg',
     'NO3- (mg/L)':'nitrate',
     'Cl- (mg/L)':'Cl',
     ' CO3-2 (mg/L)':'carbonate',
     'HCO3- (mg/L)':'bicarbonate',
     'Alcalinidad total                (mg CaCO3/L)':'total_alcalinity',
     'SO4':'sulfate',
     'Tipo de Agua':'water_type'}, axis='columns')

# The first row is garbage
df2 = df2.drop(index=0)

# The id column shouldn't have any predictive power
df2 = df2.drop(columns='id')

# The column for iron (Fe) is the only one with null values, and has 128/146. 
# Probably not worth fixing, so I'll drop it.
df2 = df2.drop(columns='Fe')

# I assume that the sampling date is not important.
df2 = df2.drop(columns='sampling_date')

# I'm only interested in variables that are likely to generalize to 
# other datasets. Though there may be some extra predictive power in the
# locations of these samples, that seems specific to this dataset.
# I will drop geographic variables.

df2 = df2.drop(columns=['longitude','latitude','municipality','town'])

# I'll also reorder the columns a bit
df2 = df2[['As', # Dependent variable first
 'pH',
 'conductivity',
 'F',
 'Na',
 'K',
 'Ca',
 'Mg',
 'nitrate',
 'Cl',
 'carbonate',
 'bicarbonate',
 'total_alcalinity',
 'sulfate',
 'water_type' # Categorical feature last
]]
```


```python
# Confirming correct data types
df2.dtypes
```




    As                  float64
    pH                  float64
    conductivity        float64
    F                   float64
    Na                  float64
    K                   float64
    Ca                  float64
    Mg                  float64
    nitrate             float64
    Cl                  float64
    carbonate           float64
    bicarbonate         float64
    total_alcalinity    float64
    sulfate             float64
    water_type           object
    dtype: object




```python
# Confirming absence of nulls
df2.isnull().sum()
```




    As                  0
    pH                  0
    conductivity        0
    F                   0
    Na                  0
    K                   0
    Ca                  0
    Mg                  0
    nitrate             0
    Cl                  0
    carbonate           0
    bicarbonate         0
    total_alcalinity    0
    sulfate             0
    water_type          0
    dtype: int64




```python
# Brief look at the clean dataframe
print(df2.shape)
df2.head()
```

    (145, 15)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>As</th>
      <th>pH</th>
      <th>conductivity</th>
      <th>F</th>
      <th>Na</th>
      <th>K</th>
      <th>Ca</th>
      <th>Mg</th>
      <th>nitrate</th>
      <th>Cl</th>
      <th>carbonate</th>
      <th>bicarbonate</th>
      <th>total_alcalinity</th>
      <th>sulfate</th>
      <th>water_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>61.5</td>
      <td>8.140</td>
      <td>337.0</td>
      <td>3.15</td>
      <td>40.9220</td>
      <td>1.4075</td>
      <td>15.0015</td>
      <td>0.2675</td>
      <td>3.25</td>
      <td>4.140</td>
      <td>0.0</td>
      <td>97.0</td>
      <td>97.0</td>
      <td>70.615</td>
      <td>BICARBONATADA SODICA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.5</td>
      <td>8.110</td>
      <td>406.0</td>
      <td>2.60</td>
      <td>45.8850</td>
      <td>0.6100</td>
      <td>20.1530</td>
      <td>0.0645</td>
      <td>2.10</td>
      <td>2.040</td>
      <td>0.0</td>
      <td>122.0</td>
      <td>122.0</td>
      <td>79.445</td>
      <td>BICARBONATADA SODICA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26.5</td>
      <td>8.375</td>
      <td>384.1</td>
      <td>1.40</td>
      <td>38.5360</td>
      <td>6.3665</td>
      <td>21.8090</td>
      <td>1.1890</td>
      <td>1.35</td>
      <td>2.325</td>
      <td>0.0</td>
      <td>140.0</td>
      <td>140.0</td>
      <td>53.730</td>
      <td>BICARBONATADA SODICA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23.5</td>
      <td>8.500</td>
      <td>557.5</td>
      <td>1.20</td>
      <td>31.6805</td>
      <td>6.7350</td>
      <td>32.8330</td>
      <td>6.5825</td>
      <td>4.40</td>
      <td>4.680</td>
      <td>0.0</td>
      <td>206.5</td>
      <td>206.5</td>
      <td>60.245</td>
      <td>BICARBONATADA CALCICA Y/O MAGNESICA</td>
    </tr>
    <tr>
      <th>5</th>
      <td>97.5</td>
      <td>8.330</td>
      <td>326.1</td>
      <td>5.95</td>
      <td>45.6745</td>
      <td>1.6525</td>
      <td>8.5190</td>
      <td>0.2100</td>
      <td>0.52</td>
      <td>5.390</td>
      <td>0.0</td>
      <td>83.0</td>
      <td>83.0</td>
      <td>59.880</td>
      <td>BICARBONATADA SODICA</td>
    </tr>
  </tbody>
</table>
</div>



## Train/test sets



```python
X = df2.drop(columns='As')
y = df2['As']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42)

# Verify all shapes
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((130, 14), (15, 14), (130,), (15,))



I'll store `X_test` and `y_test` for later, and carry out all further training and validation with the other datasets only.

## Data Exploration
Let's take a quick look at what Arsenic levels are present in our training data, and whether we need to worry much about outliers.


```python
fig, ax = plt.subplots(figsize=(10,5))
ax.hist(y_train, bins=np.linspace(0,300,100))
ax.set_xlabel('Arsenic concentration (µg/L)', fontsize=14)
ax.set_ylabel('Number of samples with this value', fontsize=14)
plt.axvline(x=35, color='r', linestyle='--', label='Mean');
plt.legend();
plt.title('Histogram: Arsenic in the training data', fontsize=16);
```


![png](output_14_0.png)



```python
sns.set(style="ticks", color_codes=True)
x_vars = X_train.columns
sns.pairplot(data=df2, y_vars=['As'], x_vars=x_vars)
plt.show()
```


![png](output_15_0.png)


Looks like there's a few outliers in both X and y, so I'll use RobustScaler to scale all my data.  As an interesting side-note, it looks as if arsenic concentration has a log-normal distribution.


```python
df3 = df2
df3['ln_As'] = np.log(df3['As'])

fig, ax = plt.subplots(figsize=(10,5))
ax.hist(df3['As'], bins=np.logspace(0,3,100))
ax.set_xlabel('Arsenic concentration (µg/L)', fontsize=14)
ax.set_ylabel('Number of samples with this value', fontsize=14)
ax.set_xscale('log')
plt.axvline(x=35, color='r', linestyle='--', label='Mean');
plt.legend();
plt.title('Arsenic concentration has a log-normal distribution', fontsize=16);
```


![png](output_17_0.png)


# Baseline Model

I'll start with the simplest possible model: assuming that all wells have an arsenic concentration equal to the mean of the training set.  This will provide a baseline score that I can use to judge more complex models.


```python
# Because I want to keep track of the results from several models, 
# I'll create a dataframe to track them.
scorecard = pd.DataFrame(columns = ['Model', 'RMSE','R^2'])
```


```python
# And the mean value is..
np.mean(y_train)
```




    36.05243022607692




```python
y_pred = [np.mean(y_train)] * len(y_train)
RMSE = np.sqrt(mean_squared_error(y_train, y_pred))
R2 = r2_score(y_train, y_pred)
scorecard = scorecard.append({'Model':'Mean Everywhere', 'RMSE':RMSE, 'R^2':R2},
                ignore_index=True)
scorecard
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>RMSE</th>
      <th>R^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean Everywhere</td>
      <td>40.286292</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



# Linear Regression
Start simple.


```python
# One-hot encode the one categorical variable 
# (also in test dataset, lest we forget)
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
```


```python
# Create a pipeline to scale and regress
pipe = make_pipeline(
    RobustScaler(),  
    LinearRegression())

# Fit on the train set
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_train)

# Fit on the train set, with grid search cross-validation
gs = GridSearchCV(pipe, param_grid={}, cv=10, 
                  scoring='neg_mean_squared_error')

gs.fit(X_train, y_train);
validation_score = np.sqrt(-gs.best_score_)
print()
print('Mean RMSE in cross-validation: ', validation_score)
print()
```

    
    Mean RMSE in cross-validation:  40.18115199336001
    



```python
# Score results
y_pred = gs.predict(X_train)
R2 = r2_score(y_train, y_pred)
scorecard = scorecard.append({'Model':'Linear Regression', 'RMSE':validation_score, 'R^2':R2},
                ignore_index=True)
scorecard
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>RMSE</th>
      <th>R^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean Everywhere</td>
      <td>40.286292</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Linear Regression</td>
      <td>40.181152</td>
      <td>0.413865</td>
    </tr>
  </tbody>
</table>
</div>



# Ridge Regression


```python
%%time

pipe = make_pipeline(
    RobustScaler(),  
    SelectKBest(f_regression), 
    Ridge())

# I'll find the optimal number of features, and also try out several
# values of alpha
param_grid = {
    'selectkbest__k': range(1, len(X_train.columns)), 
    'ridge__alpha': [0, 0.1, 1.0, 10., 100.]
}

# Fit on the train set, with grid search cross-validation
gs = GridSearchCV(pipe, param_grid=param_grid, cv=10, 
                  scoring='neg_mean_squared_error')

gs.fit(X_train, y_train)
```


```python
validation_score = np.sqrt(-gs.best_score_)
print()
print('Mean RMSE in cross-validation: ', validation_score)
```

    
    Mean RMSE in cross-validation:  35.26583153352785



```python
# Which features were selected by SelectKBest?
selected_mask = gs.best_estimator_.named_steps['selectkbest'].get_support()
selected_names = X_train.columns[selected_mask]
unselected_names = X_train.columns[~selected_mask]
coefs = gs.best_estimator_.named_steps['ridge'].coef_

print('Best parameters from cross-validation:')
for key, value in gs.best_params_.items():
    print(f'{key}: {value}')
print()
    
print('Selected Features and regression coefficients:')
for name, coef in zip(selected_names, coefs):

    print(f'{name:5}{coef:.2f}')

print()
print('Features not selected:')
for name in unselected_names:
    print(f'> {name}')
```

    Best parameters from cross-validation:
    ridge__alpha: 10.0
    selectkbest__k: 3
    
    Selected Features and regression coefficients:
    pH   11.62
    F    14.26
    K    20.44
    
    Features not selected:
    > conductivity
    > Na
    > Ca
    > Mg
    > nitrate
    > Cl
    > carbonate
    > bicarbonate
    > total_alcalinity
    > sulfate
    > water_type_BICARBONATADA CALCICA Y/O MAGNESICA
    > water_type_BICARBONATADA SODICA
    > water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS


Looks like pH, F, and K are the most important features for predicting arsenic concentration.  There's some valuable insight right there.


```python
# Score results
y_pred = gs.predict(X_train)
R2 = r2_score(y_train, y_pred)
scorecard = scorecard.append({'Model':'Ridge Regression', 'RMSE':validation_score, 'R^2':R2},
                ignore_index=True)
scorecard
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>RMSE</th>
      <th>R^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean Everywhere</td>
      <td>40.286292</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Linear Regression</td>
      <td>40.181152</td>
      <td>0.413865</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ridge Regression</td>
      <td>35.265832</td>
      <td>0.325887</td>
    </tr>
  </tbody>
</table>
</div>



# Adding polynomial features


```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_train_polynomial = poly.fit_transform(X_train)
print(X_train.shape, X_train_polynomial.shape)
```

    (130, 17) (130, 171)



```python
%%time

pipe = make_pipeline(
    RobustScaler(),  
    SelectKBest(f_regression), 
    Ridge())

param_grid = {
    'selectkbest__k': range(1, 10), 
    'ridge__alpha': [0, 0.1, 1.0, 10.]
}

# Fit on the train set, with grid search cross-validation
gs = GridSearchCV(pipe, param_grid=param_grid, cv=10, 
                  scoring='neg_mean_squared_error')

gs.fit(X_train_polynomial, y_train)
```


```python
validation_score = np.sqrt(-gs.best_score_)
print()
print('Mean RMSE in cross-validation: ', validation_score)
```

    
    Mean RMSE in cross-validation:  34.83497412458334



```python
# Which features were selected by SelectKBest?
all_names = np.array(poly.get_feature_names(X_train.columns))
selected_mask = gs.best_estimator_.named_steps['selectkbest'].get_support()
selected_names = all_names[selected_mask]
unselected_names = all_names[~selected_mask]
coefs = gs.best_estimator_.named_steps['ridge'].coef_

print('Best parameters from cross-validation:')
for key, value in gs.best_params_.items():
    print(f'{key}: {value}')
print()
    
print('Selected Features and regression coefficients:')
for name, coef in zip(selected_names, coefs):
    print(f'{name:20}{coef:.2f}')

print()
print('Features not selected:')
for name in unselected_names:
    print(f'> {name}')
```

    Best parameters from cross-validation:
    ridge__alpha: 1.0
    selectkbest__k: 7
    
    Selected Features and regression coefficients:
    K                   -44.68
    pH K                16.85
    conductivity F      8.36
    F K                 4.10
    K^2                 29.12
    K bicarbonate       0.20
    K total_alcalinity  -4.90
    
    Features not selected:
    > 1
    > pH
    > conductivity
    > F
    > Na
    > Ca
    > Mg
    > nitrate
    > Cl
    > carbonate
    > bicarbonate
    > total_alcalinity
    > sulfate
    > water_type_BICARBONATADA CALCICA Y/O MAGNESICA
    > water_type_BICARBONATADA SODICA
    > water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > pH^2
    > pH conductivity
    > pH F
    > pH Na
    > pH Ca
    > pH Mg
    > pH nitrate
    > pH Cl
    > pH carbonate
    > pH bicarbonate
    > pH total_alcalinity
    > pH sulfate
    > pH water_type_BICARBONATADA CALCICA Y/O MAGNESICA
    > pH water_type_BICARBONATADA SODICA
    > pH water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > pH water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > conductivity^2
    > conductivity Na
    > conductivity K
    > conductivity Ca
    > conductivity Mg
    > conductivity nitrate
    > conductivity Cl
    > conductivity carbonate
    > conductivity bicarbonate
    > conductivity total_alcalinity
    > conductivity sulfate
    > conductivity water_type_BICARBONATADA CALCICA Y/O MAGNESICA
    > conductivity water_type_BICARBONATADA SODICA
    > conductivity water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > conductivity water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > F^2
    > F Na
    > F Ca
    > F Mg
    > F nitrate
    > F Cl
    > F carbonate
    > F bicarbonate
    > F total_alcalinity
    > F sulfate
    > F water_type_BICARBONATADA CALCICA Y/O MAGNESICA
    > F water_type_BICARBONATADA SODICA
    > F water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > F water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > Na^2
    > Na K
    > Na Ca
    > Na Mg
    > Na nitrate
    > Na Cl
    > Na carbonate
    > Na bicarbonate
    > Na total_alcalinity
    > Na sulfate
    > Na water_type_BICARBONATADA CALCICA Y/O MAGNESICA
    > Na water_type_BICARBONATADA SODICA
    > Na water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > Na water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > K Ca
    > K Mg
    > K nitrate
    > K Cl
    > K carbonate
    > K sulfate
    > K water_type_BICARBONATADA CALCICA Y/O MAGNESICA
    > K water_type_BICARBONATADA SODICA
    > K water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > K water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > Ca^2
    > Ca Mg
    > Ca nitrate
    > Ca Cl
    > Ca carbonate
    > Ca bicarbonate
    > Ca total_alcalinity
    > Ca sulfate
    > Ca water_type_BICARBONATADA CALCICA Y/O MAGNESICA
    > Ca water_type_BICARBONATADA SODICA
    > Ca water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > Ca water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > Mg^2
    > Mg nitrate
    > Mg Cl
    > Mg carbonate
    > Mg bicarbonate
    > Mg total_alcalinity
    > Mg sulfate
    > Mg water_type_BICARBONATADA CALCICA Y/O MAGNESICA
    > Mg water_type_BICARBONATADA SODICA
    > Mg water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > Mg water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > nitrate^2
    > nitrate Cl
    > nitrate carbonate
    > nitrate bicarbonate
    > nitrate total_alcalinity
    > nitrate sulfate
    > nitrate water_type_BICARBONATADA CALCICA Y/O MAGNESICA
    > nitrate water_type_BICARBONATADA SODICA
    > nitrate water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > nitrate water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > Cl^2
    > Cl carbonate
    > Cl bicarbonate
    > Cl total_alcalinity
    > Cl sulfate
    > Cl water_type_BICARBONATADA CALCICA Y/O MAGNESICA
    > Cl water_type_BICARBONATADA SODICA
    > Cl water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > Cl water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > carbonate^2
    > carbonate bicarbonate
    > carbonate total_alcalinity
    > carbonate sulfate
    > carbonate water_type_BICARBONATADA CALCICA Y/O MAGNESICA
    > carbonate water_type_BICARBONATADA SODICA
    > carbonate water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > carbonate water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > bicarbonate^2
    > bicarbonate total_alcalinity
    > bicarbonate sulfate
    > bicarbonate water_type_BICARBONATADA CALCICA Y/O MAGNESICA
    > bicarbonate water_type_BICARBONATADA SODICA
    > bicarbonate water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > bicarbonate water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > total_alcalinity^2
    > total_alcalinity sulfate
    > total_alcalinity water_type_BICARBONATADA CALCICA Y/O MAGNESICA
    > total_alcalinity water_type_BICARBONATADA SODICA
    > total_alcalinity water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > total_alcalinity water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > sulfate^2
    > sulfate water_type_BICARBONATADA CALCICA Y/O MAGNESICA
    > sulfate water_type_BICARBONATADA SODICA
    > sulfate water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > sulfate water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > water_type_BICARBONATADA CALCICA Y/O MAGNESICA^2
    > water_type_BICARBONATADA CALCICA Y/O MAGNESICA water_type_BICARBONATADA SODICA
    > water_type_BICARBONATADA CALCICA Y/O MAGNESICA water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > water_type_BICARBONATADA CALCICA Y/O MAGNESICA water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > water_type_BICARBONATADA SODICA^2
    > water_type_BICARBONATADA SODICA water_type_CLORURADAS Y/O SULFATADAS SODICAS
    > water_type_BICARBONATADA SODICA water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > water_type_CLORURADAS Y/O SULFATADAS SODICAS^2
    > water_type_CLORURADAS Y/O SULFATADAS SODICAS water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS
    > water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS^2


Well, look at that.  We already knew that K (potassium) was important, but now that I've tested polynomial features it seems like K is way more important than we thought.  It's the most significant feature on its own, it shows up as K^2, and the other features seem to matter only inasmuch as they interact with K (though there's also the interaction between F and conductivity).  


```python
# Score results
y_pred = gs.predict(X_train_polynomial)
R2 = r2_score(y_train, y_pred)
scorecard = scorecard.append({'Model':'Ridge Regression, Polynomial Features', 'RMSE':validation_score, 'R^2':R2},
                ignore_index=True)
scorecard
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>RMSE</th>
      <th>R^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean Everywhere</td>
      <td>40.286292</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Linear Regression</td>
      <td>40.181152</td>
      <td>0.413865</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ridge Regression</td>
      <td>35.265832</td>
      <td>0.325887</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ridge Regression, Polynomial Features</td>
      <td>34.834974</td>
      <td>0.431645</td>
    </tr>
  </tbody>
</table>
</div>



# Testing on test data
Alright, it seems like Ridge Regression with Polynomial Features wins.  Let's see how well it performs on the test data.


```python
# X_train and X_test have different columns, because the one-hot encoder
# encountered a different set of options to encode.  Thus, I have to add 
# the missing column before running the polynomial feature generator
X_train.columns
```




    Index(['pH', 'conductivity', 'F', 'Na', 'K', 'Ca', 'Mg', 'nitrate', 'Cl',
           'carbonate', 'bicarbonate', 'total_alcalinity', 'sulfate',
           'water_type_BICARBONATADA CALCICA Y/O MAGNESICA',
           'water_type_BICARBONATADA SODICA',
           'water_type_CLORURADAS Y/O SULFATADAS SODICAS',
           'water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS'],
          dtype='object')




```python
X_test.columns
```




    Index(['pH', 'conductivity', 'F', 'Na', 'K', 'Ca', 'Mg', 'nitrate', 'Cl',
           'carbonate', 'bicarbonate', 'total_alcalinity', 'sulfate',
           'water_type_BICARBONATADA CALCICA Y/O MAGNESICA',
           'water_type_BICARBONATADA SODICA',
           'water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS'],
          dtype='object')




```python
# This column will get selected out anyway, so it's ok.
# I just need it here so that PolynomialFeatures will generate the right
# column names
X_test['water_type_CLORURADAS Y/O SULFATADAS SODICAS']=0
```




    Index(['pH', 'conductivity', 'F', 'Na', 'K', 'Ca', 'Mg', 'nitrate', 'Cl',
           'carbonate', 'bicarbonate', 'total_alcalinity', 'sulfate',
           'water_type_BICARBONATADA CALCICA Y/O MAGNESICA',
           'water_type_BICARBONATADA SODICA',
           'water_type_SULFATADA Y/O CLORURADAS CALCICAS Y/O MAGNESICAS',
           'water_type_CLORURADAS Y/O SULFATADAS SODICAS'],
          dtype='object')




```python
# Confirmed, they both have the same shape
poly = PolynomialFeatures(degree=2)
X_test_polynomial = poly.fit_transform(X_test)
X_test_polynomial.shape, X_train_polynomial.shape
```




    ((15, 171), (130, 171))




```python
# Same gs as used last time.
y_pred = gs.predict(X_test_polynomial)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
R2 = r2_score(y_test, y_pred)
print(f'RMSE: {RMSE}')
print(f'R^2: {R2}')
```

    RMSE: 20.812209976076126
    R^2: -0.48578768514621995


The RMSE is within the range of what we expect. R^2 is negative, but I guess that happens with very small datasets like this one (15 test samples).


```python

```
