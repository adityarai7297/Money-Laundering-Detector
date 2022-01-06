#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.express as px
from IPython.display import display, HTML
import pandas as pd
import sqlite3
from sqlite3 import Error
import csv
import pprint
from random import random



# In[2]:


### Utility Functions

def create_connection(db_file, delete_db=False):
    import os
    if delete_db and os.path.exists(db_file):
        os.remove(db_file)

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql, drop_table_name=None):
    
    if drop_table_name: # You can optionally pass drop_table_name to drop the table. 
        try:
            c = conn.cursor()
            c.execute("""DROP TABLE IF EXISTS %s""" % (drop_table_name))
        except Error as e:
            print(e)
    
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)
        
def execute_sql_statement(sql_statement, conn):
    cur = conn.cursor()
    cur.execute(sql_statement)

    rows = cur.fetchall()

    return rows


# In[3]:


normalized_database_filename = 'normalized.db'
conn = create_connection(normalized_database_filename, delete_db=True)


# In[4]:


with open('PS_20174392719_1491204439457_log.csv', 'r') as file:
    header = None
    my_list = []
    for ele in file:
        ele = ele.strip()
        if header is None:
            header = ele.split(',')
            continue
        line = ele.split(',')
        my_list.append(line)  


# In[7]:


qry = 'CREATE TABLE Test_New({} INT, {} TEXT, {} REAL , {} TEXT , {} REAL , {} REAL , {} TEXT ,{} REAL ,{} REAL ,{} INT,{} INT)'.format(*header)
#execute_sql_staatement(qry , conn ,  drop_table_name='Test_New')
create_table(conn , qry ,drop_table_name='Test_New')


# In[8]:


conn.executemany("INSERT INTO Test_New VALUES (?,?,?,?,?,?,?,?,?,?,?)", my_list)


# In[9]:


execute_sql_statement('select count(*) from Test_New' , conn )


# In[10]:


del my_list


# In[11]:


df = pd.read_sql_query("SELECT * from Test_New", conn)


# In[12]:


fig = px.histogram(df, x="type")
fig.show()


# In[12]:


#Taking customer realted transactions and removing Merchant transactions

qry = "select * from Test_New WHERE nameDest like 'C%'";#limit 1000
customer_transactions = execute_sql_statement(qry , conn)
print(len(customer_transactions))


# In[13]:


#Checking isFlaggedFraud column as it shows only 16 values in 6 million records
qry = "select isFlaggedFraud ,count(*) from Test_New WHERE nameDest like 'C%'group by 1"
print(execute_sql_statement(qry , conn))


# In[14]:


df = pd.read_sql_query("SELECT * from Test_New", conn)


# In[15]:


df.head()


# In[16]:


df.info()


# In[17]:


df.describe()


# In[19]:


df.isnull().values.any()


# In[20]:


# Number of transaction which is flagged as fraud per transaction type
ax = df.groupby(['type', 'isFlaggedFraud']).size().plot(kind='bar')
ax.set_title("Number of transaction which is flagged as fraud per transaction type")
ax.set_xlabel("(Type, isFlaggedFraud)")
ax.set_ylabel("Count of transaction")
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))


# In[21]:


# Number of transaction which are actual fraud per transaction type
ax = df.groupby(['type', 'isFraud']).size().plot(kind='bar')
ax.set_title("Number of transaction which are actual fraud per transaction type")
ax.set_xlabel("(Type, isFraud)")
ax.set_ylabel("Count of transaction")
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))


# In[83]:


sns.scatterplot(x=df_preprocessed['amount'], y=df_preprocessed['step'], hue=fraud)


# In[22]:


# Adding new feature Column to headers to create new table for customer related transactions, removing isFlaggedFraud
# columns as it has only 16 entries has flaged 1 compared to 6 million records 
header.append('rank')
header.append('Dest_txn_count')
header.remove('isFlaggedFraud')
print(header)


# In[23]:


#Creating new temporary table to remove isflagged column
qry = 'CREATE TABLE Cust_Transaction({} INT, {} TEXT, {} REAL , {} TEXT , {} REAL , {} REAL , {} TEXT ,{} REAL ,{} REAL ,{} INT, {} INT, {} INT)'.format(*header[0:12]);
create_table(conn, qry, drop_table_name= 'Cust_Transaction')


# In[24]:


#Inserting Customer related transactions and removing the isflaggedfraud 
qry= '''INSERT INTO Cust_Transaction 

        SELECT step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,a.nameDest,oldbalanceDest,newbalanceDest,isFraud,
        ROW_NUMBER() OVER (PARTITION BY a.nameDest ORDER BY a.nameDest, step ASC) as rank, Dest_txn_count
        FROM Test_New a 
        inner join (select nameDest, count(1) as Dest_txn_count from Test_New group by 1) b
        on a.nameDest = b.nameDest
        WHERE a.nameDest like 'C%'
        '''

execute_sql_statement(qry , conn)


# In[25]:


#Adding two feature columns
qry = "ALTER TABLE Cust_Transaction ADD COLUMN Source_final_balance"
execute_sql_statement(qry , conn)
qry = "ALTER TABLE Cust_Transaction ADD COLUMN Dest_final_balance"
execute_sql_statement(qry , conn)


# In[26]:


# Checking the number of customer records of new customer transactions table
qry_count = 'Select count(1) from Cust_Transaction'
print(execute_sql_statement(qry_count , conn))


# In[27]:


#Updating two feature columns - which are the final balances of origin and source accounts
qry = '''Update Cust_Transaction
set Source_final_balance = (amount + oldbalanceOrg)
where type = 'CASH_IN' '''
execute_sql_statement(qry , conn)

qry = '''Update Cust_Transaction
set Source_final_balance = (amount - oldbalanceOrg)
where type != 'CASH_IN' '''
execute_sql_statement(qry , conn)
 


# In[28]:


#Updating two feature columns - which are the final balances of origin and destination accounts
qry = '''Update Cust_Transaction
set Dest_final_balance = (amount - oldbalanceOrg)
where type = 'CASH_IN' '''
execute_sql_statement(qry , conn)

qry = '''Update Cust_Transaction
set Dest_final_balance = (amount + oldbalanceOrg)
where type != 'CASH_IN' '''
execute_sql_statement(qry ,conn)


# In[30]:


cust_df = pd.read_sql_query("SELECT * from Cust_Transaction", conn)


# In[32]:


len(cust_df[cust_df['isFraud']==0])
len(cust_df[cust_df['isFraud']==1])
# Total Customer ratio of fraud to non-fraud 
cust_df_ratio = len(cust_df[cust_df['isFraud']==0]) / len(cust_df[cust_df['isFraud']==1])
print(cust_df_ratio)


# In[33]:


df_sample = cust_df.sample(frac = 0.2)


# In[34]:


### Observing Number of transactions over each timestep


import plotly.graph_objects as go
import numpy as np

# Create figure
fig = go.Figure()

per_step_value_dict = {}

for step in range(1, 745):
    per_step_value_dict[step] = df_sample.loc[df['step'] == step]
    step+=1



# Add traces, one for each slider step
for step in per_step_value_dict.keys():
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=3),
            name="𝜈 = " + str(step),
            x= np.arange(len(per_step_value_dict[step]['amount'].to_numpy())),
            y= per_step_value_dict[step]['amount'].to_numpy()  ))

## Make 10th trace visible
fig.data[10].visible = True
#
## Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

fig.show()


# In[35]:


len(df_sample[df_sample['isFraud']==0])
len(df_sample[df_sample['isFraud']==1])
# 20% Customer ratio of fraud to non-fraud 
cust_df_sample_ratio = len(df_sample[df_sample['isFraud']==0]) / len(df_sample[df_sample['isFraud']==1])
print(cust_df_sample_ratio)


# In[74]:


#### Anomaly Detection #####


# In[40]:


qry = '''select  step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, Source_final_balance, Dest_final_balance, isFraud, nameDest, rank, Dest_txn_count        
         from Cust_Transaction 
         order by nameDest, rank ASC
         '''

anomaly_df = pd.read_sql_query(qry, conn)


# In[41]:


##Feature Engineering: Adding derived features for a particular Destination account transaction 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

def preprocessing(dataframe):    
    dataframe['diff_source_final'] = dataframe['Source_final_balance'] - dataframe['newbalanceOrig']
    dataframe['Dest_final_balance_as_per_next_txn'] = np.where(dataframe['rank'] == dataframe['Dest_txn_count'], dataframe['newbalanceDest'], dataframe[(dataframe['nameDest']==dataframe['nameDest'])]['oldbalanceDest'].shift(-1))
    dataframe['diff_Dest_final'] = dataframe['Dest_final_balance'] - dataframe['Dest_final_balance_as_per_next_txn']

    df_cust_amount_scaled = pd.DataFrame(scaler.fit_transform(np.array(dataframe['amount']).reshape(-1,1)), columns = ['amount'])
    dataframe['amount_scaled'] = np.array(df_cust_amount_scaled)

    df_cust_diff_source_final_scaled = pd.DataFrame(scaler.fit_transform(np.array(dataframe['diff_source_final']).reshape(-1,1)), columns = ['diff_source_final'])
    dataframe['diff_source_final_scaled'] = np.array(df_cust_diff_source_final_scaled)

    df_cust_diff_Dest_final_scaled = pd.DataFrame(scaler.fit_transform(np.array(dataframe['diff_Dest_final']).reshape(-1,1)), columns = ['diff_Dest_final'])
    dataframe['diff_Dest_final_scaled'] = np.array(df_cust_diff_Dest_final_scaled)
    
    return dataframe;


# In[42]:


anomaly_df_sample = anomaly_df.sample(frac = 0.2,random_state = 0)
df_preprocessed = preprocessing(anomaly_df_sample)


# In[43]:


df_preprocessed.head()


# In[44]:


print(len(df_preprocessed[df_preprocessed['isFraud']==1]))
print(len(df_preprocessed[df_preprocessed['isFraud']==0]))


# In[47]:


df_corr_variables = df_preprocessed[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud','Source_final_balance','Dest_final_balance','diff_source_final','diff_Dest_final']]
corrMatrix = df_corr_variables.corr()
import matplotlib.pyplot as plt
import seaborn as sn
sn.heatmap(corrMatrix, annot=True)
plt.show()


# In[49]:


df_anomaly = df_preprocessed[['diff_source_final','diff_Dest_final']]
df_anomaly.head()


# In[50]:


plt.figure()
plt.xlabel('diff_source_final_scaled')
plt.ylabel('diff_Dest_final_scaled')
plt.scatter(df_preprocessed['diff_source_final_scaled'], df_preprocessed['diff_Dest_final_scaled'], c= df_preprocessed['isFraud'],  cmap=plt.cm.autumn)
plt.show() 


# In[90]:


plt.figure()
plt.scatter(df_preprocessed['diff_source_final_scaled'], df_preprocessed['amount_scaled'], c= df_preprocessed['isFraud'],  cmap=plt.cm.autumn)
plt.show() 


# In[91]:


plt.figure()
plt.scatter(df_preprocessed['diff_Dest_final_scaled'], df_preprocessed['amount_scaled'], c= df_preprocessed['isFraud'],  cmap=plt.cm.autumn)
plt.show() 


# In[51]:


s = np.sum(df_anomaly, axis=0)
mu = s/len(df_anomaly)
mu


# In[52]:


vr = np.sum((df_anomaly - mu)**2, axis=0)
variance = vr/len(df_anomaly)
variance


# In[53]:


var_dia = np.diag(variance)
var_dia


# In[54]:


X= df_preprocessed['diff_source_final_scaled']
Y = df_preprocessed['amount_scaled']
fraud = df_preprocessed['isFraud']
sns.scatterplot(x= X, y=Y, hue= fraud)


# In[55]:


from scipy.stats import norm

prob_dist = norm.pdf(df_anomaly, mu, variance)
prob = []

for i in range(0,len(prob_dist)):
    prob.append(prob_dist[i][0]*prob_dist[i][1])
    


# In[56]:


print(len(prob))
print(len(df_preprocessed))


# In[57]:


prob_df = pd.DataFrame(prob_dist, columns = ['P1','P2'])
prob_df['probability'] = prob_df['P1'] * prob_df['P2']


# In[60]:


prob_df = pd.merge(prob_df, df_preprocessed['isFraud'], how = 'inner',left_index=True, right_index=True)


# In[63]:


fraud_df = prob_df[(prob_df['isFraud'] == 1)]
fraud_df = pd.concat([fraud_df, prob_df[(prob_df['isFraud'] == 0)].head(400)])


# In[ ]:


sns.scatterplot(x= list(range(0,842832225)), y=fraud_df['probability'], hue= fraud_df['isFraud'])


# In[ ]:





# In[ ]:


#### Random Forest #####


# In[ ]:


# Taking only tansfer and cash out records as all other types has no fraud transactions for X 

X = df_sample.loc[(df_sample.type == 'TRANSFER') | (df_sample.type == 'CASH_OUT')]
Y = pd.DataFrame(data = X['isFraud'], columns = ['isFraud'])
del X['isFraud']
# input data

# X.drop(['step', 'nameOrig', 'nameDest'], axis=1, inplace=True)
X = X.drop(['nameOrig', 'nameDest', 'step'], axis = 1)
# Categorising transaction types into two classes(e.g. 0 and 1)
# Transfr = 0, Cashout = 1
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int)


# In[ ]:


# Dividing totat data into 70% train and 30% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30)


# In[ ]:


print('Number of train transactions', len(X_train))
print('Number of test transactions', len(X_test))
print('Number of total transactions', len(X))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred= classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[ ]:




