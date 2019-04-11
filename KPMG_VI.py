
# coding: utf-8

# In[106]:


#Get all packages needed
import pandas as pd
import datetime


# In[77]:


kmpgex = pd.ExcelFile('KPMG.xlsx')
kmpgex.sheet_names
df = pd.read_excel('KPMG.xlsx', sheetname="CustomerDemographic")
list(df)


# In[88]:


# rename for easier analysis
df.rename(columns={"Note: The data and information in this document is reflective of a hypothetical situation and client. This document is to be used for KPMG Virtual Internship purposes only. ":"customer_id"}, inplace = True)
df.rename(columns={"Unnamed: 1":"fname",
                   "Unnamed: 2":"lname",
                   "Unnamed: 3":"gender",
                   "Unnamed: 4":"3y_bike_purchases",
                   "Unnamed: 5":"DOB",
                   "Unnamed: 6":"JT"}, inplace = True)
df.rename(columns={"Unnamed: 7":"Category",
                   "Unnamed: 8":"wealth_segement",
                   "Unnamed: 9":"D_Indicator",
                   "Unnamed: 10":"default",
                   "Unnamed: 11":"owns_car",
                   "Unnamed: 12":"tencure"}, inplace = True)

df.head(35)


# In[139]:


def check_NA():
    ret = []
    temp = list(df)
    for each in temp:
        ret.append(df[each].isna().sum())
    return ret

NaNlist = check_NA()
print(NaNlist, len(NaNlist))
# need to check columns
check = []
k     = -1
for i in NaNlist:
    k += 1
    if i > 0:
        check.append(k)
print(check)


# In[6]:


#dictionary where keys is column name and values is the unique elements in list
def check_unique():
    temp = list(df)
    mydict = {}
    for i in range(3, 13):
        ct = 0
        if NaNlist[i] > 0:
            ct = len(df[temp[i]].unique()) - 1
        else:
            ct = len(df[temp[i]].unique())
        mydict[temp[i]] = (ct, df[temp[i]].unique())
    return mydict
print(check_unique())


# In[13]:


check_unique()["gender"]
#As we can see, there are 6 different values in gender but it's not well organized
#Miss spelled Femal. 
#should be only Female/Male or F/M


# In[127]:


#DOB has NaN
#check_unique()["DOB"] doesn't work since there will be many different days
#check for data type instead
print(isinstance(df["DOB"][34], str))
lenn = len(df["DOB"])
ret = []
j = 0
for i in range(1, lenn):
    if type(df["DOB"][i]) not in ret:
        ret.append(type(df["DOB"][i]))
print(ret)
#type of DOB is also not consistent. 


# In[130]:


#JT has NaN
#check_unique()["JT"] doesn't work since there will be many different job  tittles
#Category has NaN
check_unique()["Category"] #not much errors to fix


# In[141]:


print(check_unique()["wealth_segement"])
print(check_unique()["D_Indicator"])


# In[143]:


print(check_unique()["default"]) # Has NaN need more description on this because it looks like it has weird symbols


# In[142]:


print(check_unique()["owns_car"]) # Not much to fix
print(check_unique()["tencure"]) # Has NaN, not much to fix

