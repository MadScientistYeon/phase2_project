## Final Project Submission

Please fill out:
* Student name: Yeonjae Zhang
* Student pace: full time
* Scheduled project review date/time: April 1st, 2022 Friday
* Instructor name: Praveen Gowtham
* Blog post URL: 


# Overview
Home buyer as family person requested the guide line to buy a house.

# Business Problem
The stakeholder previously bought a house that was overvalued and far from his son’s school. His family was unimpressed.

# Data Understanding


```python
# import standard packages
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
```

### King County House Data
King County real estate data for homes sold in and around King County,  Washington.


```python
# explore the data
df = pd.read_csv('data/kc_house_data.csv')
df2 = pd.read_csv('data/middle_school_hd.csv')
```

object dtype: Date, waterfront, view, condition, grade, sqft_basement


```python
# check all the dtype is numbers
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 21 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   id             21597 non-null  int64  
     1   date           21597 non-null  object 
     2   price          21597 non-null  float64
     3   bedrooms       21597 non-null  int64  
     4   bathrooms      21597 non-null  float64
     5   sqft_living    21597 non-null  int64  
     6   sqft_lot       21597 non-null  int64  
     7   floors         21597 non-null  float64
     8   waterfront     19221 non-null  object 
     9   view           21534 non-null  object 
     10  condition      21597 non-null  object 
     11  grade          21597 non-null  object 
     12  sqft_above     21597 non-null  int64  
     13  sqft_basement  21597 non-null  object 
     14  yr_built       21597 non-null  int64  
     15  yr_renovated   17755 non-null  float64
     16  zipcode        21597 non-null  int64  
     17  lat            21597 non-null  float64
     18  long           21597 non-null  float64
     19  sqft_living15  21597 non-null  int64  
     20  sqft_lot15     21597 non-null  int64  
    dtypes: float64(6), int64(9), object(6)
    memory usage: 3.5+ MB



```python
df.head()
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
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NONE</td>
      <td>...</td>
      <td>7 Average</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>NO</td>
      <td>NONE</td>
      <td>...</td>
      <td>7 Average</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>NO</td>
      <td>NONE</td>
      <td>...</td>
      <td>6 Low Average</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>NO</td>
      <td>NONE</td>
      <td>...</td>
      <td>7 Average</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>NO</td>
      <td>NONE</td>
      <td>...</td>
      <td>8 Good</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



### Middle school distance data
Middle school locations in King County. We are able to calculate the distances from the houses in King County real estate data. 


```python
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 23 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   id             21597 non-null  int64  
     1   date           21597 non-null  object 
     2   price          21597 non-null  int64  
     3   bedrooms       21597 non-null  int64  
     4   bathrooms      21597 non-null  float64
     5   sqft_living    21597 non-null  int64  
     6   sqft_lot       21597 non-null  int64  
     7   floors         21597 non-null  float64
     8   waterfront     19221 non-null  object 
     9   view           21534 non-null  object 
     10  condition      21597 non-null  object 
     11  grade          21597 non-null  object 
     12  sqft_above     21597 non-null  int64  
     13  sqft_basement  21597 non-null  object 
     14  yr_built       21597 non-null  int64  
     15  yr_renovated   17755 non-null  float64
     16  zipcode        21597 non-null  int64  
     17  lat            21597 non-null  float64
     18  long           21597 non-null  float64
     19  sqft_living15  21597 non-null  int64  
     20  sqft_lot15     21597 non-null  int64  
     21  HubName        21597 non-null  int64  
     22  HubDist        21597 non-null  float64
    dtypes: float64(6), int64(11), object(6)
    memory usage: 3.8+ MB



```python
df2.describe()
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
      <th>id</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>HubName</th>
      <th>HubDist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.159700e+04</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>17755.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.580474e+09</td>
      <td>5.402966e+05</td>
      <td>3.373200</td>
      <td>2.115826</td>
      <td>2080.321850</td>
      <td>1.509941e+04</td>
      <td>1.494096</td>
      <td>1788.596842</td>
      <td>1970.999676</td>
      <td>83.636778</td>
      <td>98077.951845</td>
      <td>47.560093</td>
      <td>-122.213982</td>
      <td>1986.620318</td>
      <td>12758.283512</td>
      <td>6374.645877</td>
      <td>1.264706</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.876736e+09</td>
      <td>3.673681e+05</td>
      <td>0.926299</td>
      <td>0.768984</td>
      <td>918.106125</td>
      <td>4.141264e+04</td>
      <td>0.539683</td>
      <td>827.759761</td>
      <td>29.375234</td>
      <td>399.946414</td>
      <td>53.513072</td>
      <td>0.138552</td>
      <td>0.140724</td>
      <td>685.230472</td>
      <td>27274.441950</td>
      <td>621.799657</td>
      <td>1.049036</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000102e+06</td>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>370.000000</td>
      <td>1900.000000</td>
      <td>0.000000</td>
      <td>98001.000000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
      <td>5103.000000</td>
      <td>0.020115</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.123049e+09</td>
      <td>3.220000e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>1190.000000</td>
      <td>1951.000000</td>
      <td>0.000000</td>
      <td>98033.000000</td>
      <td>47.471100</td>
      <td>-122.328000</td>
      <td>1490.000000</td>
      <td>5100.000000</td>
      <td>5893.000000</td>
      <td>0.689758</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.904930e+09</td>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1910.000000</td>
      <td>7.618000e+03</td>
      <td>1.500000</td>
      <td>1560.000000</td>
      <td>1975.000000</td>
      <td>0.000000</td>
      <td>98065.000000</td>
      <td>47.571800</td>
      <td>-122.231000</td>
      <td>1840.000000</td>
      <td>7620.000000</td>
      <td>6399.000000</td>
      <td>1.069560</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.308900e+09</td>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.068500e+04</td>
      <td>2.000000</td>
      <td>2210.000000</td>
      <td>1997.000000</td>
      <td>0.000000</td>
      <td>98118.000000</td>
      <td>47.678000</td>
      <td>-122.125000</td>
      <td>2360.000000</td>
      <td>10083.000000</td>
      <td>6827.000000</td>
      <td>1.588646</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.900000e+09</td>
      <td>7.700000e+06</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>9410.000000</td>
      <td>2015.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>6210.000000</td>
      <td>871200.000000</td>
      <td>7592.000000</td>
      <td>24.952758</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.head()
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
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>HubName</th>
      <th>HubDist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NONE</td>
      <td>...</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
      <td>6219</td>
      <td>1.750332</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>False</td>
      <td>NONE</td>
      <td>...</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
      <td>7170</td>
      <td>1.440778</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>False</td>
      <td>NONE</td>
      <td>...</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
      <td>5516</td>
      <td>1.243018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>False</td>
      <td>NONE</td>
      <td>...</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
      <td>6214</td>
      <td>1.473385</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>False</td>
      <td>NONE</td>
      <td>...</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
      <td>5338</td>
      <td>0.687628</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



# Data Preparation
Prepare the data for house price prediction model

### Data Cleaning
I drop NaN, encode categorical variables and find correlated variables with price


```python
# drop NaN
cleaned_df = df.dropna()
```


```python
# sqft_basement is not categorical varibale but have to change dtype to float.
cleaned_df.sqft_basement.value_counts()
```




    0.0       9362
    ?          333
    600.0      155
    500.0      151
    700.0      148
              ... 
    2010.0       1
    1481.0       1
    1913.0       1
    4820.0       1
    248.0        1
    Name: sqft_basement, Length: 283, dtype: int64




```python
# sqft_basement have '?' value that is string. I drop '?' values.
cleaned_df = cleaned_df.loc[cleaned_df.sqft_basement != '?']
cleaned_df.sqft_basement = cleaned_df.sqft_basement.astype('float')
```


```python
# drop NaN
cleaned_df2 = df2.dropna()
```

# Data Analysis

### Correlation features with price


```python
# import library
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn')
```


```python
# plot continuous variables with price
corr_df = abs(cleaned_df.corr().iloc[2:10])['price'].sort_values()
corr_df.plot.bar(figsize=(15,9), fontsize=20)
plt.title('Features Effect on Price', fontsize=20)
plt.xlabel('Features', fontsize=20)
plt.ylabel('Effect on price', fontsize=20)
plt.show()
```


    
![png](image/numeric_feature.png)
    



```python
# plot categorical variables with price except date

fig, axes2 = plt.subplots(nrows = 1, ncols = 4, figsize = (15, 10), sharey = True)

categoricals = ['waterfront', 'view', 'condition', 'grade']

for col, ax in zip(categoricals, axes2.flatten()):
    cleaned_df.groupby(col).mean()['price'].sort_values().plot.bar(ax=ax, fontsize=15)
    ax.set(xlabel=None)
    ax.set_title(col, fontsize=20)
    ax.set_yticks([])
    ax.set_ylabel('Price', fontsize=20)
fig.tight_layout()
```


    
![png](image/categorical_feature.png)
    


# Feature Engineering

### Preprocess Train Data


```python
# preprocessing with scikit-learn
y = cleaned_df['price']
X = cleaned_df.drop('price', axis=1)
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```


```python
print(f"X_train is a DataFrame with {X_train.shape[0]} rows and {X_train.shape[1]} columns")
print(f"y_train is a Series with {y_train.shape[0]} values")
```

    X_train is a DataFrame with 11571 rows and 20 columns
    y_train is a Series with 11571 values



```python
# Select relevant Columns
relevant_columns = ['bathrooms',
                    'bedrooms',
                    'sqft_living',
                    'waterfront',
                    'view',
                    'condition',
                    'grade',
                    'sqft_basement',
                    'lat',
                    'floors',
                    'sqft_above'
                    ]

# Reassign X_train so that it only contains relevant columns
X_train = X_train.loc[:, relevant_columns]
```


```python
X_train.isna().sum()
```




    bathrooms        0
    bedrooms         0
    sqft_living      0
    waterfront       0
    view             0
    condition        0
    grade            0
    sqft_basement    0
    lat              0
    floors           0
    sqft_above       0
    dtype: int64




```python
# Convert Categorical Features into Numbeers
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 11571 entries, 560 to 10173
    Data columns (total 11 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   bathrooms      11571 non-null  float64
     1   bedrooms       11571 non-null  int64  
     2   sqft_living    11571 non-null  int64  
     3   waterfront     11571 non-null  object 
     4   view           11571 non-null  object 
     5   condition      11571 non-null  object 
     6   grade          11571 non-null  object 
     7   sqft_basement  11571 non-null  float64
     8   lat            11571 non-null  float64
     9   floors         11571 non-null  float64
     10  sqft_above     11571 non-null  int64  
    dtypes: float64(4), int64(3), object(4)
    memory usage: 1.1+ MB



```python
# date, waterfront, view, condition, and grade are objects
from sklearn.preprocessing import OrdinalEncoder

```


```python
# waterfront transform
waterfront_train = X_train[['waterfront']]
encoder_waterfront = OrdinalEncoder(categories=[['NO', 'YES']])
encoder_waterfront.fit(waterfront_train)
waterfront_encoded_train = encoder_waterfront.transform(waterfront_train)
waterfront_encoded_train = waterfront_encoded_train.flatten()
X_train['waterfront'] = waterfront_encoded_train
```


```python
# view transform
view_train = X_train[['view']]
encoder_view = OrdinalEncoder(categories=[['NONE', 'FAIR', 'AVERAGE', 'GOOD', 'EXCELLENT']])
encoder_view.fit(view_train)
view_encoded_train = encoder_view.transform(view_train)
view_encoded_train = view_encoded_train.flatten()
X_train['view'] = view_encoded_train
```


```python
# condition transform
condition_train = X_train[['condition']]
encoder_condition = OrdinalEncoder(categories=[['Poor', 'Fair', 'Average', 'Good', 'Very Good']])
encoder_condition.fit(condition_train)
condition_encoded_train = encoder_condition.transform(condition_train)
condition_encoded_train = condition_encoded_train.flatten()
X_train['condition'] = condition_encoded_train
```


```python
# grade transform
grade_train = X_train[['grade']]
encoder_grade = OrdinalEncoder(categories=[['3 Poor', '4 Low', '5 Fair', '6 Low Average', '7 Average', '8 Good', '9 Better', '10 Very Good', '11 Excellent', '12 Luxury', '13 Mansion']])
encoder_grade.fit(grade_train)
grade_encoded_train = encoder_grade.transform(grade_train)
grade_encoded_train = grade_encoded_train.flatten()
X_train['grade'] = grade_encoded_train
```

### Preprocess Test Data


```python
# Drop Irrelevant Columns
X_test = X_test.loc[:, relevant_columns]

# Transform categorical values to numbers
# waterfront transform
waterfront_test = X_test[['waterfront']]
waterfront_encoded_test = encoder_waterfront.transform(waterfront_test).flatten()
X_test['waterfront'] = waterfront_encoded_test
# view transform
view_test = X_test[['view']]
view_encoded_test = encoder_view.transform(view_test).flatten()
X_test['view'] = view_encoded_test
# condition transform
condition_test = X_test[['condition']]
condition_encoded_test = encoder_condition.transform(condition_test).flatten()
X_test['condition'] = condition_encoded_test
# grade transform
grade_test = X_test[['grade']]
grade_encoded_test = encoder_grade.transform(grade_test).flatten()
X_test['grade'] = grade_encoded_test

```

### Find houses near middle school


```python
# visualize the hub distance
cleaned_df2.describe()['HubDist'].loc[['25%','50%','75%']].plot.bar(figsize=(16,10), fontsize=20)
plt.axhline(0.7, color='red', label='0.7 miles', linewidth = 2)
plt.title('Top Three Interquartile Ranges for Home to School Distance', fontsize=20)
plt.ylabel('Distance')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f7920a3d520>




    
![png](image/top3_distance.png)
    



```python
# drop houses' school disctance under 0.7 miles that is shown above
cleaned_df2 = cleaned_df2.loc[cleaned_df2.HubDist <= 0.7]
cleaned_df2.head()
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
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>HubName</th>
      <th>HubDist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>False</td>
      <td>NONE</td>
      <td>...</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
      <td>5338</td>
      <td>0.687628</td>
    </tr>
    <tr>
      <th>11</th>
      <td>9212900260</td>
      <td>5/27/2014</td>
      <td>468000</td>
      <td>2</td>
      <td>1.00</td>
      <td>1160</td>
      <td>6000</td>
      <td>1.0</td>
      <td>False</td>
      <td>NONE</td>
      <td>...</td>
      <td>300.0</td>
      <td>1942</td>
      <td>0.0</td>
      <td>98115</td>
      <td>47.6900</td>
      <td>-122.292</td>
      <td>1330</td>
      <td>6000</td>
      <td>6260</td>
      <td>0.544642</td>
    </tr>
    <tr>
      <th>13</th>
      <td>6054650070</td>
      <td>10/7/2014</td>
      <td>400000</td>
      <td>3</td>
      <td>1.75</td>
      <td>1370</td>
      <td>9680</td>
      <td>1.0</td>
      <td>False</td>
      <td>NONE</td>
      <td>...</td>
      <td>0.0</td>
      <td>1977</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6127</td>
      <td>-122.045</td>
      <td>1370</td>
      <td>10208</td>
      <td>5338</td>
      <td>0.651845</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6865200140</td>
      <td>5/29/2014</td>
      <td>485000</td>
      <td>4</td>
      <td>1.00</td>
      <td>1600</td>
      <td>4300</td>
      <td>1.5</td>
      <td>False</td>
      <td>NONE</td>
      <td>...</td>
      <td>0.0</td>
      <td>1916</td>
      <td>0.0</td>
      <td>98103</td>
      <td>47.6648</td>
      <td>-122.343</td>
      <td>1610</td>
      <td>4300</td>
      <td>5676</td>
      <td>0.559904</td>
    </tr>
    <tr>
      <th>19</th>
      <td>7983200060</td>
      <td>4/24/2015</td>
      <td>230000</td>
      <td>3</td>
      <td>1.00</td>
      <td>1250</td>
      <td>9774</td>
      <td>1.0</td>
      <td>False</td>
      <td>NONE</td>
      <td>...</td>
      <td>0.0</td>
      <td>1969</td>
      <td>0.0</td>
      <td>98003</td>
      <td>47.3343</td>
      <td>-122.306</td>
      <td>1280</td>
      <td>8850</td>
      <td>6399</td>
      <td>0.576901</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



# Modeling and Evaluation

### House Price Prediction Modeling


```python
# Model define
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```


```python
# Evaluation with cross-validation
from sklearn.model_selection import cross_val_score
cross_val_score(model, X_train**2, y_train, cv=5)
```




    array([0.75132775, 0.69063972, 0.64892535, 0.72535465, 0.73110131])




```python
# Evaluation with test set
model.fit(X_train**2, y_train)
model.score(X_test**2, y_test)
```




    0.7150335566493079



### Prediction


```python
# Get total data
y_total = pd.concat([y_test, y_train]).sort_index()
X_total = pd.concat([X_test, X_train]).sort_index()

# Predict price
pred = model.predict(X_total**2).round()
cleaned_df['predict_price'] = pred
```


```python
# Visualize the real data with predicted data
average_df = cleaned_df[['price', 'predict_price']].mean()
average_df.plot.bar(fontsize=20, figsize=(10,9), color=['blue', 'orange'])
plt.title('Real Price VS Predicted Price', fontsize=20)
plt.xticks([0,1], ['Real Price: $541,497', 'Predicted: $542,798'], rotation=0)
plt.show()
```


    
![png](image/evaluation.png)
    


### Apply to business problem


```python
# Find houses that is under the 60% price that AI predicted
results_df = cleaned_df.loc[cleaned_df.price < cleaned_df.predict_price*0.6, ['id', 'price', 'predict_price', 'lat', 'long']]
results_df
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
      <th>id</th>
      <th>price</th>
      <th>predict_price</th>
      <th>lat</th>
      <th>long</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>51</th>
      <td>7231300125</td>
      <td>345000.0</td>
      <td>604994.0</td>
      <td>47.4934</td>
      <td>-122.189</td>
    </tr>
    <tr>
      <th>65</th>
      <td>3253500160</td>
      <td>317625.0</td>
      <td>656787.0</td>
      <td>47.5747</td>
      <td>-122.304</td>
    </tr>
    <tr>
      <th>107</th>
      <td>3530510041</td>
      <td>188500.0</td>
      <td>367002.0</td>
      <td>47.3813</td>
      <td>-122.322</td>
    </tr>
    <tr>
      <th>142</th>
      <td>1432900240</td>
      <td>205000.0</td>
      <td>345225.0</td>
      <td>47.4563</td>
      <td>-122.171</td>
    </tr>
    <tr>
      <th>194</th>
      <td>3996900125</td>
      <td>230000.0</td>
      <td>439855.0</td>
      <td>47.7481</td>
      <td>-122.300</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21211</th>
      <td>2738640470</td>
      <td>623300.0</td>
      <td>1207573.0</td>
      <td>47.7726</td>
      <td>-122.162</td>
    </tr>
    <tr>
      <th>21263</th>
      <td>6056110780</td>
      <td>229800.0</td>
      <td>405694.0</td>
      <td>47.5647</td>
      <td>-122.293</td>
    </tr>
    <tr>
      <th>21347</th>
      <td>3782760080</td>
      <td>410000.0</td>
      <td>689145.0</td>
      <td>47.7345</td>
      <td>-121.967</td>
    </tr>
    <tr>
      <th>21420</th>
      <td>1608000120</td>
      <td>255000.0</td>
      <td>430354.0</td>
      <td>47.3860</td>
      <td>-122.184</td>
    </tr>
    <tr>
      <th>21546</th>
      <td>2122059216</td>
      <td>422000.0</td>
      <td>735882.0</td>
      <td>47.3846</td>
      <td>-122.186</td>
    </tr>
  </tbody>
</table>
<p>507 rows × 5 columns</p>
</div>




```python
# Visualize all houses
import folium

map1 = folium.Map(location=[47.5,-122])
points = (results_df.lat, results_df.long)
lat = points[0]
long = points[1]

for la, lo, real, pred in zip(lat, long, results_df.price, results_df.predict_price):
    iframe = folium.IFrame('price: ${} predict: ${}'.format(real, pred), width=100, height=100)
    popup = folium.Popup(iframe, max_width=100)
    folium.Marker(location=[la,lo],popup=popup).add_to(map1)
    
map1

```
![png](image/map1.png)



```python
# Available profits
results_df[['price', 'predict_price']].mean().plot.bar(figsize=(10,9), fontsize=20, color=['green', 'red'])
plt.xticks([0,1],['BUY: $364,219', 'VALUE: $698,198'], rotation=0);
plt.title('Profitable Houses Average Price', fontsize=20)
```




    Text(0.5, 1.0, 'Profitable Houses Average Price')




    
![png](image/profitable_houses.png)
    



```python
# Visualize houses that is near middle school
results_df = results_df.join(cleaned_df2, how='inner', lsuffix='index')
map2 = folium.Map(location=[47.5,-122])
points = (results_df.lat, results_df.long)
lat = points[0]
long = points[1]

for la, lo, real, pred in zip(lat, long, results_df.price, results_df.predict_price):
    iframe = folium.IFrame('price: ${} predict: ${}'.format(real, pred), width=100, height=100)
    popup = folium.Popup(iframe, max_width=100)
    folium.Marker(location=[la,lo],popup=popup).add_to(map2)
```


```python
map2
```
![png](image/map2.png)



```python
# Available profits
results_df[['price', 'predict_price']].mean().plot.bar(figsize=(10,9), fontsize=20, color=['blue', 'red'])
plt.xticks([0,1],['BUY: $301,645', 'VALUE: $564,435'], rotation=0);
plt.title('Profitable Houses Near Middle School', fontsize=20)
```




    Text(0.5, 1.0, 'Profitable Houses Near Middle School')




    
![png](image/near_middle_school.png)
    


## Conclusion
* 71.5% of the data fit our house price prediction model. 

* The model was able to recommend 507 houses to purchase after finding homes where the actual price was 40% lower than the predicted price. 

* To mitigate the commute time for the middle school child we found how many of the 507 houses fall within 0.7 miles from the closest middle school. We found a final list of100 houses that lie within 0.7 miles from a middle school!


```python

```
