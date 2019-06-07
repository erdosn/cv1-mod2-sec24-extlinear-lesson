
### Questions
* What is bias?
    *  cumulative sum of differences sum(abs((yhat - y)))
* Are bias and variance related to accuracy and precision?

### Objectives
YWBAT
* explain ridge and lasso regression
* explain what hyperparameters do
* explain bias and variance tradeoff 

### Outline


```python
import pandas as pd
import numpy as np


import statsmodels.api as sm
import scipy.stats as scs

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


import matplotlib.pyplot as plt
```


```python
x = np.linspace(0, 2*np.pi, 1000)
intercept = np.random.randint(20, 30)
```


```python
x2 = np.column_stack([x, x**2])
x3 = np.column_stack([x2, x**3])
x4 = np.column_stack([x3, x**4])
xfin = np.column_stack([x4, x**5])
```


```python
error = np.random.normal(1, 0.5, 1000)
```


```python
y = np.sin(x) + error + intercept
```


```python
xfin = sm.add_constant(xfin)
linreg = sm.OLS(y, xfin).fit()
```


```python
xfin
```




    array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00],
           [1.00000000e+00, 6.28947478e-03, 3.95574930e-05, 2.48795855e-07,
            1.56479526e-09, 9.84174030e-12],
           [1.00000000e+00, 1.25789496e-02, 1.58229972e-04, 1.99036684e-06,
            2.50367241e-08, 3.14935689e-10],
           ...,
           [1.00000000e+00, 6.27060636e+00, 3.93205041e+01, 2.46563403e+02,
            1.54610204e+03, 9.69499729e+03],
           [1.00000000e+00, 6.27689583e+00, 3.93994213e+01, 2.47306063e+02,
            1.55231440e+03, 9.74371578e+03],
           [1.00000000e+00, 6.28318531e+00, 3.94784176e+01, 2.48050213e+02,
            1.55854546e+03, 9.79262991e+03]])




```python
linreg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.664</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.663</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   393.5</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 07 Jun 2019</td> <th>  Prob (F-statistic):</th> <td>1.00e-232</td>
</tr>
<tr>
  <th>Time:</th>                 <td>11:51:06</td>     <th>  Log-Likelihood:    </th> <td> -717.35</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1000</td>      <th>  AIC:               </th> <td>   1447.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   994</td>      <th>  BIC:               </th> <td>   1476.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>   22.0766</td> <td>    0.094</td> <td>  236.025</td> <td> 0.000</td> <td>   21.893</td> <td>   22.260</td>
</tr>
<tr>
  <th>x1</th>    <td>    0.5150</td> <td>    0.301</td> <td>    1.709</td> <td> 0.088</td> <td>   -0.076</td> <td>    1.106</td>
</tr>
<tr>
  <th>x2</th>    <td>    0.6544</td> <td>    0.298</td> <td>    2.199</td> <td> 0.028</td> <td>    0.070</td> <td>    1.238</td>
</tr>
<tr>
  <th>x3</th>    <td>   -0.5536</td> <td>    0.120</td> <td>   -4.608</td> <td> 0.000</td> <td>   -0.789</td> <td>   -0.318</td>
</tr>
<tr>
  <th>x4</th>    <td>    0.1156</td> <td>    0.021</td> <td>    5.481</td> <td> 0.000</td> <td>    0.074</td> <td>    0.157</td>
</tr>
<tr>
  <th>x5</th>    <td>   -0.0074</td> <td>    0.001</td> <td>   -5.506</td> <td> 0.000</td> <td>   -0.010</td> <td>   -0.005</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 5.749</td> <th>  Durbin-Watson:     </th> <td>   1.959</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.056</td> <th>  Jarque-Bera (JB):  </th> <td>   5.660</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.182</td> <th>  Prob(JB):          </th> <td>  0.0590</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.061</td> <th>  Cond. No.          </th> <td>8.46e+04</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 8.46e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
yhat = linreg.predict()
```


```python
plt.figure(figsize=(10, 8))
plt.scatter(x, y, c='g', alpha=0.5)
plt.plot(x, yhat, linewidth=2, c='r')
plt.show()
```


![png](lesson-plan_files/lesson-plan_12_0.png)



```python
resid = yhat - y
```


```python
plt.hist(resid, bins=20)
plt.show()
```


![png](lesson-plan_files/lesson-plan_14_0.png)



```python
poly = PolynomialFeatures(2)
```


```python
xfin = poly.fit_transform(x.)
```

# Ridge and Lasso are applied to our cost function
What cost function do we normally use? 

residuals, mean squared, rmse

Ridge_Cost = MSE + $\lambda \sum(\beta_j^2)$

What is lambda times the sum of our beta js for j>=1?

it's as simple as saying *it* is a constant a number

why do we want to add penalty as we add features? 
Because it's going to increase our bias and decrease our variance. 

bias comes from training. 

bias increases by increasing training size or increasing terms



variance is the ability to predict unseen data

loses flexibility when it's biased

### What about lambda?

if lambda = 0 then what happens?

**our cost function remains the same**

if lambda >>1 then what happens?

**our error gets too big and we're no longer letting our model have robustedness**

RidgeCostFunction(betas, lambda)

LassoCostFunction(betas, lambda)

LassoCostFunction = MSE + $\lambda \sum\|\beta_j\|$

examples

high bias, low variance: trainscore = 90%, testscore = 50%

low bias, low variance: trainscore = 40%, testscore = 40%

high bias, high variance: trainscore = 85%, testscore = 85%

### WDWL
* putting the train/test scores clarifies what we're looking for in a model
* better understanding of ridge and lasso and why we use it
* ridge and lasso are methods for balancing the bias/variance tradeoff
* we learned why linear regression models can form curved lines
