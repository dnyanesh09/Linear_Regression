import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data=pd.read_csv("Advertising.csv")
# print(data)
print(data.describe())
x1=data['TV']
y=data['Sales']
plt.scatter(x1,y)
plt.xlabel("Advertising Budget On TV")
plt.ylabel("Sales")

x=sm.add_constant(x1)
result=sm.OLS(y,x).fit()
print(result.summary())
yhat=7.0326+x1*0.0475
fig=plt.plot(x1,yhat,lw=5,c='blue',label='Regression Line')
plt.show()