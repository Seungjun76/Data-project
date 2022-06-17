#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 나라별 독립변수에 따른 행복지수에 미치는 영향 분석 및 행복 지수 예측모델 만들기


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[9]:


"""
독립변수인 행복 지수와 영향을 미칠 것 같은 종속변수들의 CSV파일의 데이터를 
jupyter 노트북으로의 표현
"""
"""
HPI = Happy Planet Index (행복 지수)
QLI = Quality of life Index (삶의 질 지수)
PPI = Purchasing Power Index (구매력 지수)
SI = Safety Index (안전 지수)
HCI = Health Care Index (건강 관리 지수)
CLI = Cost of Living Index (생활비지수)
PPIR = Property Price to Income Ratio (부동산 가격 대 소득 비율)
TCTI = Traffic Commute Time Index (교통 통근 시간 지수)
PI = Pollution Index (오염 지수)
CI = Climate Index (기후 지수)
"""


df= pd.read_csv('happy.csv')
df


# In[583]:


"""데이터의 행과 열의 갯수 확인"""

df.shape        


# In[584]:


"""데이터 결측치 확인"""

df.isna().sum()


# In[585]:


"""상위 10개 목록을 표현"""

df.head(10)


# In[6]:


from statsmodels.formula.api import ols 
from statsmodels.stats.anova import anova_lm


# In[612]:


"""
H0: 삶의 질 지수의 차이에 따른 행복 지수의 차이가 없다.
H1: 삶의 질 지수의 차이에 따른 행복 지수의 차이가 있다.
P_value = 0.159946, 유의수준=0.05, 즉 P>0.05이므로 귀무가설 채택

결론:  삶의 질 지수에 따른 행복 지수의 차이가 있다는 근거가 충분하지 않다.  
"""

model= ols('HPI ~ C(QLI)' , df).fit()  
anova_lm(model)


# In[588]:


plt.scatter(df['HPI'],df['QLI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('QLI')
plt.show()
a1= np.corrcoef(df.HPI,df.QLI)                     #매우 높은 양의 상관관계를 나타냄
print(a1)


# In[611]:


"""
같은 방식으로
H0: 구매력 지수, 건강 관리 지수, 생활비지수 (PPI, HCI, CLI)에 따른 행복 지수의 차이가 없다.
H1: 구매력 지수, 건강 관리 지수, 생활비지수 (PPI, HCI, CLI)에 따른 행복 지수의 차이가 있다.
유의수준=0.05, P-value값 다 0.05보다 높게 측정됨, 즉 P>0.05이므로 귀무가설 채택

결론:구매력 지수, 건강 관리 지수, 생활비지수에 따른 행복 지수의 차이가 있다는 근거가 충분하지 않다.  
"""
plt.scatter(df['HPI'],df['PPI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('PPI')
plt.show()
a2= np.corrcoef(df.HPI,df.PPI)                     #높은 양의 상관관계를 나타냄

print(a2)
plt.scatter(df['HPI'],df['HCI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('HCI')
plt.show()
a4= np.corrcoef(df.HPI,df.HCI)                    #보통인 양의 상관관계를 나타냄

print(a4)
plt.scatter(df['HPI'],df['CLI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('CLI')
plt.show()
a5= np.corrcoef(df.HPI,df.CLI)                    #높은 양의 상관관계를 나타냄
print(a5)


# In[592]:


"""
H0: 안전 지수에 따른 행복 지수의 차이가 없다.
H1: 안전 지수에 따른 행복 지수의 차이가 있다.
P_value = 0.960988, 유의수준=0.05, 즉  P>0.05이므로 귀무가설 채택

결론: 안전 지수에 따른 행복 지수의 차이가 있다는 근거가 충분하지 않다.  
"""

model= ols('HPI ~ C(SI)' , df).fit()              
anova_lm(model)


# In[593]:


plt.scatter(df['HPI'],df['SI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('SI')
plt.show()

a3= np.corrcoef(df.HPI,df.SI)                   #낮은 양의 상관관계를 나타냄
print(a3)


# In[598]:


"""
H0: 부동산 가격 대 소득 비율에 따른 행복 지수의 차이가 없다.
H1: 부동산 가격 대 소득 비율에 따른행복 지수의 차이가 있다.
P_value = 0.295657, 유의수준=0.05, 즉 P>0.05이므로 귀무가설 채택
결론: 부동산 가격 대 소득 비율에 따른 행복 지수의 차이가 있다는 근거가 충분하지 않다. 
"""

model= ols('HPI ~ C(PPIR)' , df).fit()  
anova_lm(model)


# In[599]:


plt.scatter(df['HPI'],df['PPIR'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('PPIR')
plt.show()
a6= np.corrcoef(df.HPI,df.PPIR)                   #보통인 양의 상관관계를 나타냄
print(a6)


# In[600]:


"""
H0: 교통 통근 시간 지수에 따른 행복 지수의 차이가 없다.
H1: 교통 통근 시간 지수에 따른 행복 지수의 차이가 있다.
P_value = 0.188361, 유의수준=0.05, 즉 P>0.05이므로 귀무가설 채택
결론: 교통 통근 시간 지수에 따른 행복 지수의 차이가 있다는 근거가 충분하지 않다. 
"""

model= ols('HPI ~ C(TCTI)' , df).fit()  
anova_lm(model)


# In[575]:


plt.scatter(df['HPI'],df['TCTI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('TCTI')
plt.show()
a7= np.corrcoef(df.HPI,df.TCTI)                   #보통인 양의 상관관계를 나타냄
print(a7)


# In[601]:


"""
H0: 오염 지수에 따른 행복 지수의 차이가 없다.
H1: 오염 지수에 따른 행복 지수의 차이가 있다.
P_value = 0.502966, 유의수준=0.05, 즉 P>0.05이므로 귀무가설 채택
결론: 오염 지수에 따른 행복 지수의 차이가 있다는 근거가 충분하지 않다. 
"""

model= ols('HPI ~ C(PI)' , df).fit()  
anova_lm(model)


# In[602]:


plt.scatter(df['HPI'],df['PI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('PI')
plt.show()
a8= np.corrcoef(df.HPI,df.PI)                  #매우 높은 음의 상관관계를 나타냄
print(a8)


# In[603]:


"""
H0: 기후 지수에 따른 행복 지수의 차이가 없다.
H1: 기후 지수에 따른 행복 지수의 차이가 있다.
P_value = 0.699304, 유의수준=0.05, 즉 P>0.05이므로 귀무가설 채택
결론: 기후 지수에 따른 행복 지수의 차이가 있다는 근거가 충분하지 않다. 
"""

model= ols('HPI ~ C(CI)' , df).fit()  
anova_lm(model)


# In[604]:


plt.scatter(df['HPI'],df['CI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('CI')
plt.show()
a9= np.corrcoef(df.HPI,df.CI)                          #상관관계가 없다
print(a9)


# In[ ]:


"""
결과: 행복 지수인 종속변수에 영향을 미치는 독립변수는 없었다. 
즉, 행복은 삶의 질, 구매력, 오염, 기후 지수 등 위에서 말한 독립변수들에 따라
움직인다고 말할 수 없다.
"""


# In[ ]:


"""
    🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓🠓
"""


# In[ ]:


# 이제는 반대로 PPI,SI,HCI,CLI ... CI에 따른 행복 지수의 값에 대한 예측모델을 세울 것울 것이며
# 테스트 및 정확도 등을 알아볼 것이다.


# In[225]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[303]:


"""y의 값"""

dependent_variable = 'HPI'


# In[304]:


independent_variables= df.columns.tolist()


# In[305]:


independent_variables.remove(dependent_variable)


# In[307]:


independent_variables.remove('Country')


# In[664]:


"""x의 값 """

independent_variables


# In[665]:


x = df[independent_variables].values               #x값 설정
y= df[dependent_variable].values                   #y값 설정
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)

regressor=LinearRegression()                       #데이터 셋을 트레인 셋과 테스트 셋으로 분할
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)                    #y 예측값 설정
print(y_pred)

r2_score(y_test,y_pred)                             # 41% 점수를 보임


# In[668]:


regressor.predict([[130.02, 76.6, 73.14, 82.3, 81.2, 23.63, 39.88, 61.85, 68.39]])  
                  
                  #독립변수 다 있을 때 대한민국 행복 지수 값 6.56096171


# In[644]:


# 정확도를 높이기 위해 위에 나온 변수들 중에 행복지수와 관련이없다고 생각한 'CI'를 제거해 보았다
independent_variables_2 = ['QLI', 'PPI', 'SI', 'HCI', 'CLI', 'PPIR', 'TCTI', 'PI']
x = df[independent_variables_2].values               
y= df[dependent_variable].values                   
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)

regressor=LinearRegression()                       
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)                   
print(y_pred)

r2_score(y_test,y_pred)                                # 46% 의 점수를 보여줌 (5%향상)


# In[645]:


# 정확도를 또 높이는 실험을 위해 위에 나온 변수들 중에 'SI'를 제거해 보았다.
independent_variables_3= ['QLI', 'PPI', 'HCI', 'CLI', 'PPIR', 'TCTI', 'PI','CI']
x = df[independent_variables_3].values               
y= df[dependent_variable].values                   
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)
regressor=LinearRegression()                       
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
print(y_pred)
                  
r2_score(y_test,y_pred)                        


# In[656]:


# 정확도를 또 높이는 실험을 위해 위에 나온 변수들 중에 'QLI'를 제거해 보았다.
independent_variables_4= [ 'PPI', 'HCI', 'CLI', 'PPIR', 'TCTI', 'PI','CI']
x = df[independent_variables_4].values               
y= df[dependent_variable].values                   
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)
regressor=LinearRegression()                       
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
print(y_pred)
                  
r2_score(y_test,y_pred)                        


# In[647]:


# 정확도를 또 높이는 실험을 위해 위에 나온 변수들 중에 'TCTI'를 제거해 보았다.
independent_variables_5= ['QLI', 'PPI', 'HCI', 'CLI', 'SI','PPIR', 'PI','CI']
x = df[independent_variables_5].values               
y= df[dependent_variable].values                   
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)
regressor=LinearRegression()                       
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
print(y_pred)
                  
r2_score(y_test,y_pred)                        


# In[657]:


# 정확도를 또 높이는 실험을 위해 위에 나온 변수들 중에 'HCI'를 제거해 보았다.
independent_variables_6= ['QLI', 'PPI', 'CLI','SI', 'PPIR', 'TCTI', 'PI','CI']
x = df[independent_variables_6].values               
y= df[dependent_variable].values                   
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)
regressor=LinearRegression()                       
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
print(y_pred)
                  
r2_score(y_test,y_pred)                        


# In[649]:


''' 

반복

'''


# In[650]:


#독립변수 한개 제거했을 때는 'CI'를 제거했을 때 제일 높게 나타남


# In[651]:


#이제 제일 제외했을 때 높게 나타난 CI와 함께 한가지 변수를 더 제외해보자


# In[669]:


independent_variables_7 = ['QLI', 'PPI', 'HCI', 'SI', 'PPIR', 'TCTI', 'PI']
x = df[independent_variables_7].values               
y= df[dependent_variable].values                   
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)

regressor=LinearRegression()                       
regressor.fit(x_train,y_train)                    #여러가지 시행중
                                                  #CLI CI를 제거했을 때 높은 점수가 나타남

y_pred=regressor.predict(x_test)                   
print(y_pred)

r2_score(y_test,y_pred)


# In[670]:


regressor.predict([[130.02,76.6,73.14,82.3,23.63,39.88,61.85]])  
                                             #가장 높은 점수 모델을 가지고 한국 값을 넣었더니
                                             # 실제 값=5.935, 아까보다 매우 근접한 결과


# In[659]:


independent_variables_8 = ['QLI', 'PPI', 'HCI', 'PPIR', 'TCTI', 'PI']
x = df[independent_variables_8].values               
y= df[dependent_variable].values                   
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)

regressor=LinearRegression()                       
regressor.fit(x_train,y_train)                     
                                                 #CL CI포함해 3개를 제외하면 아까보다 떨어짐

y_pred=regressor.predict(x_test)                   
print(y_pred)

r2_score(y_test,y_pred)


# In[655]:


pred_y_df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred,'Difference':y_test-y_pred})
pred_y_df[0:20]                                   
                                                   #실제 값과 예측값을 통한 가치 확인
                                                   #차이 값 확인


# In[12]:


plt.scatter(df['HPI'],df['QLI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('QLI')
plt.show()
a1= np.corrcoef(df.HPI,df.QLI)                     #매우 높은 양의 상관관계를 나타냄
print(a1)

"""
같은 방식으로
H0: 구매력 지수, 건강 관리 지수, 생활비지수 (PPI, HCI, CLI)에 따른 행복 지수의 차이가 없다.
H1: 구매력 지수, 건강 관리 지수, 생활비지수 (PPI, HCI, CLI)에 따른 행복 지수의 차이가 있다.
유의수준=0.05, P-value값 다 0.05보다 높게 측정됨, 즉 P>0.05이므로 귀무가설 채택

결론:구매력 지수, 건강 관리 지수, 생활비지수에 따른 행복 지수의 차이가 있다는 근거가 충분하지 않다.  
"""
plt.scatter(df['HPI'],df['PPI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('PPI')
plt.show()
a2= np.corrcoef(df.HPI,df.PPI)                     #높은 양의 상관관계를 나타냄

print(a2)
plt.scatter(df['HPI'],df['HCI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('HCI')
plt.show()
a4= np.corrcoef(df.HPI,df.HCI)                    #보통인 양의 상관관계를 나타냄

print(a4)
plt.scatter(df['HPI'],df['CLI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('CLI')
plt.show()
a5= np.corrcoef(df.HPI,df.CLI)                    #높은 양의 상관관계를 나타냄
print(a5)
plt.scatter(df['HPI'],df['SI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('SI')
plt.show()


# In[13]:


a3= np.corrcoef(df.HPI,df.SI)                   #낮은 양의 상관관계를 나타냄
print(a3)
plt.scatter(df['HPI'],df['PPIR'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('PPIR')
plt.show()
a6= np.corrcoef(df.HPI,df.PPIR)                   #보통인 양의 상관관계를 나타냄
print(a6)
plt.scatter(df['HPI'],df['TCTI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('TCTI')
plt.show()
a7= np.corrcoef(df.HPI,df.TCTI)                   #보통인 양의 상관관계를 나타냄
print(a7)
plt.scatter(df['HPI'],df['PI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('PI')
plt.show()
a8= np.corrcoef(df.HPI,df.PI)                  #매우 높은 음의 상관관계를 나타냄
print(a8)
plt.scatter(df['HPI'],df['CI'],alpha=0.5)
plt.title('scatter plot')
plt.xlabel('HPI')
plt.ylabel('CI')
plt.show()
a9= np.corrcoef(df.HPI,df.CI)                          #상관관계가 없다
print(a9)


# In[ ]:




