import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




# Title the app
st.title('Credit Score Classification')


# Load data
df_train = pd.read_csv("train.csv", low_memory=False)
df_test = pd.read_csv("test.csv", low_memory=False)


# Set up the sidebar
st.sidebar.title('Developer Information')
st.sidebar.text('Welcome to my Streamlit app!')

# Display name at the top of the sidebar
st.sidebar.subheader('Developer:')
st.sidebar.text('Zohreh Taghibakhsh')

#link to GitHub profile
st.sidebar.markdown('[GitHub](https://github.com/your_username)')

st.subheader('Plot')
# dropdown menu to the sidebar
selected_option = st.sidebar.selectbox('Select a Plot to Display', ['Histogram for Age', 'Count Plot for Occupation', 'Correlation Heatmap', 'Bar Plot of Annual Income', 
                                                            'Scatterplot of Monthly Inhand Salary vs. Amount invested monthly' , 'Pie Chart of Occupation Counts',
                                                            'Count Plot of Credit Mix'])


#dropdown menu for machine learning models
selected_model = st.sidebar.selectbox('Select Machine Learning Model', ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'KNN'])


# Main content
st.subheader(f'{selected_option}')

# Display the head of the data
st.subheader('Raw Data - Head')
st.write(df_train.head())

# Display list of columns
st.subheader('List of Columns')
st.write(df_train.columns.tolist())


# --------------------------------------------
# data preprocessing

# Drop irelevent columns
columns_to_drop = ["Name", "SSN", "ID"]
df_train.drop(columns=columns_to_drop, inplace=True)
df_test.drop(columns=columns_to_drop, inplace=True)

# Convert "Month" column to numerical format
df_train['Month'] = pd.to_datetime(df_train['Month'], format='%B', errors='coerce').dt.month
df_test['Month'] = pd.to_datetime(df_test['Month'], format='%B', errors='coerce').dt.month

# Remove non-numeric characters from "Age" column
df_train['Age'] = df_train['Age'].str.replace(r'[^0-9-]', '', regex=True)
df_test['Age'] = df_test['Age'].str.replace(r'[^0-9-]', '', regex=True)

# Convert "Age" column to numeric
df_train['Age'] = pd.to_numeric(df_train['Age'], errors='coerce').astype('Int64')
df_test['Age'] = pd.to_numeric(df_test['Age'], errors='coerce').astype('Int64')

# Make negative values in "Age" column positive
df_train['Age'] = df_train['Age'].abs()
df_test['Age'] = df_test['Age'].abs()

# Replace ages greater than 75 with NaN
df_train['Age'] = df_train['Age'].apply(lambda x: x if 0 <= x <= 75 else np.nan)
df_test['Age'] = df_test['Age'].apply(lambda x: x if 0 <= x <= 75 else np.nan)

# Drop rows with NaN values in the "Age" column
df_train = df_train.dropna(subset=['Age']).reset_index(drop=True)
df_test = df_test.dropna(subset=['Age']).reset_index(drop=True)

# Replace NaN values in the "Occupation" "Type_of_Loan" "Credit_Mix" column with ""
df_train['Occupation'].fillna("", inplace=True)
df_test['Occupation'].fillna("", inplace=True)
df_train['Type_of_Loan'].fillna("Not Specified", inplace=True)
df_test['Type_of_Loan'].fillna("Not Specified", inplace=True)
df_train['Credit_Mix'].fillna("", inplace=True)
df_test['Credit_Mix'].fillna("", inplace=True)

# Replace values less than 0 in "Num_Bank_Accounts" with 0
df_train['Num_Bank_Accounts'] = df_train['Num_Bank_Accounts'].apply(lambda x: max(0, x))
df_test['Num_Bank_Accounts'] = df_test['Num_Bank_Accounts'].apply(lambda x: max(0, x))

# Drop "Customer_ID" column
df_train.drop(columns=['Customer_ID'], inplace=True)
df_test.drop(columns=['Customer_ID'], inplace=True)

# Replace "_______" with "Not Specified" in the Occupation column
df_train['Occupation'].replace("_______", "Not Specified", inplace=True)
df_test['Occupation'].replace("_______", "Not Specified", inplace=True)

# Remove leading and trailing underscores from "Annual_Income" column
df_train['Annual_Income'] = df_train['Annual_Income'].str.replace('_', '')
df_test['Annual_Income'] = df_test['Annual_Income'].str.replace('_', '')

# Convert "Annual_Income" column to numeric
df_train['Annual_Income'] = pd.to_numeric(df_train['Annual_Income'], errors='coerce')
df_test['Annual_Income'] = pd.to_numeric(df_test['Annual_Income'], errors='coerce')

# Convert "Num_of_Loan" column to numeric and then to integers, replacing NaN with 0
df_train['Num_of_Loan'] = pd.to_numeric(df_train['Num_of_Loan'], errors='coerce').astype('Int64').fillna(0)
df_test['Num_of_Loan'] = pd.to_numeric(df_test['Num_of_Loan'], errors='coerce').astype('Int64').fillna(0)

# Remove non-numeric characters from "Num_of_Delayed_Payment" column
df_train['Num_of_Delayed_Payment'] = df_train['Num_of_Delayed_Payment'].str.replace(r'\D', '', regex=True)
df_test['Num_of_Delayed_Payment'] = df_test['Num_of_Delayed_Payment'].str.replace(r'\D', '', regex=True)

# Fill NaN values with 0 and convert the column to integers
df_train['Num_of_Delayed_Payment'] = pd.to_numeric(df_train['Num_of_Delayed_Payment'], errors='coerce').fillna(0).astype('Int64')
df_test['Num_of_Delayed_Payment'] = pd.to_numeric(df_test['Num_of_Delayed_Payment'], errors='coerce').fillna(0).astype('Int64')

# Remove non-numeric characters from "Changed_Credit_Limit" column
df_train['Changed_Credit_Limit'] = df_train['Changed_Credit_Limit'].str.replace(r'\D', '', regex=True)
df_test['Changed_Credit_Limit'] = df_test['Changed_Credit_Limit'].str.replace(r'\D', '', regex=True)

# Replace empty strings with 0 before converting to float
df_train['Changed_Credit_Limit'] = pd.to_numeric(df_train['Changed_Credit_Limit'], errors='coerce').fillna(0).astype('float64')
df_test['Changed_Credit_Limit'] = pd.to_numeric(df_test['Changed_Credit_Limit'], errors='coerce').fillna(0).astype('float64')

# Drop rows with missing values in 'Monthly_Inhand_Salary' column
df_train = df_train.dropna(subset=['Monthly_Inhand_Salary']).reset_index(drop=True)
df_test = df_test.dropna(subset=['Monthly_Inhand_Salary']).reset_index(drop=True)

# Fill missing values with 0 for 'Num_Credit_Inquiries' column
df_train['Num_Credit_Inquiries'] = df_train['Num_Credit_Inquiries'].fillna(0)
df_test['Num_Credit_Inquiries'] = df_test['Num_Credit_Inquiries'].fillna(0)

# Replace '_' with 'Not Specified' in 'Credit_Mix' column
df_train['Credit_Mix'] = df_train['Credit_Mix'].replace('_', 'Not Specified')
df_test['Credit_Mix'] = df_test['Credit_Mix'].replace('_', 'Not Specified')

# Remove non-numeric characters except decimal point from "Outstanding_Debt" column
df_train['Outstanding_Debt'] = df_train['Outstanding_Debt'].str.replace(r'[^0-9.]', '', regex=True)
df_test['Outstanding_Debt'] = df_test['Outstanding_Debt'].str.replace(r'[^0-9.]', '', regex=True)

# Replace empty strings with 0 before converting to float
df_train['Outstanding_Debt'] = pd.to_numeric(df_train['Outstanding_Debt'], errors='coerce').fillna(0)
df_test['Outstanding_Debt'] = pd.to_numeric(df_test['Outstanding_Debt'], errors='coerce').fillna(0)

# Extract years and months
df_train['Years'] = df_train['Credit_History_Age'].str.extract(r'(\d+) Years').astype('float64')
df_train['Months'] = df_train['Credit_History_Age'].str.extract(r'(\d+) Months').astype('float64')

df_test['Years'] = df_test['Credit_History_Age'].str.extract(r'(\d+) Years').astype('float64')
df_test['Months'] = df_test['Credit_History_Age'].str.extract(r'(\d+) Months').astype('float64')

# Convert years to months and add to months
df_train['Credit_History_Age_Months'] = df_train['Years'] * 12 + df_train['Months']
df_test['Credit_History_Age_Months'] = df_test['Years'] * 12 + df_test['Months']

# Drop intermediate columns
df_train = df_train.drop(['Years', 'Months', 'Credit_History_Age'], axis=1)
df_test = df_test.drop(['Years', 'Months', 'Credit_History_Age'], axis=1)

# Replace NaN values with 0 and convert to Int64
df_train['Credit_History_Age_Months'] = df_train['Credit_History_Age_Months'].fillna(0).astype('Int64')
df_test['Credit_History_Age_Months'] = df_test['Credit_History_Age_Months'].fillna(0).astype('Int64')

# Convert "Amount_invested_monthly" to float
df_train['Amount_invested_monthly'] = pd.to_numeric(df_train['Amount_invested_monthly'], errors='coerce').astype('float64')
df_test['Amount_invested_monthly'] = pd.to_numeric(df_test['Amount_invested_monthly'], errors='coerce').astype('float64')

# Replace NaN values with 0
df_train['Amount_invested_monthly'].fillna(0, inplace=True)
df_test['Amount_invested_monthly'].fillna(0, inplace=True)

# Convert "Amount_invested_monthly" to float
df_train['Monthly_Balance'] = pd.to_numeric(df_train['Monthly_Balance'], errors='coerce').astype('float64')
df_test['Monthly_Balance'] = pd.to_numeric(df_test['Monthly_Balance'], errors='coerce').astype('float64')

# Replace NaN values with 0
df_train['Monthly_Balance'].fillna(0, inplace=True)
df_test['Monthly_Balance'].fillna(0, inplace=True)

df_train['Payment_Behaviour'] = df_train['Payment_Behaviour'].replace('!@9#%8', 'Not specified')
df_test['Payment_Behaviour'] = df_test['Payment_Behaviour'].replace('!@9#%8', 'Not specified')

df_train['Delay_from_due_date'] = df_train['Delay_from_due_date'].apply(lambda x: max(0, x))
df_test['Delay_from_due_date'] = df_test['Delay_from_due_date'].apply(lambda x: max(0, x))


# Plots
if selected_option == 'Histogram for Age':
    st.subheader('Histogram for Age')
    fig, ax = plt.subplots()
    sns.histplot(df_train['Age'], kde=True, ax=ax)
    ax.set_title('Distribution of Age')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)


elif selected_option == 'Count Plot for Occupation':
    st.subheader('Count Plot for Occupation')
    fig, ax = plt.subplots()
    sns.countplot(x='Occupation', data=df_train, hue='Occupation', palette='pastel', ax=ax)
    ax.set_title('Count of Individuals in Each Occupation')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)


elif selected_option == 'Correlation Heatmap':
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(12, 10))
    numeric_columns = df_train.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df_train[numeric_columns].corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

elif selected_option == 'Bar Plot of Annual Income':
    st.subheader('Bar Plot of Annual Income considering Credit score')
    fig, ax = plt.subplots()
    sns.barplot(x='Credit_Score', y='Annual_Income', data=df_train, estimator=np.median, ax=ax)
    st.pyplot(fig)

elif selected_option == 'Scatterplot of Monthly Inhand Salary vs. Amount invested monthly':
    st.subheader('Scatterplot of Monthly Inhand Salary vs. Amount invested monthly')
    fig, ax = plt.subplots()
    sns.scatterplot(x='Monthly_Inhand_Salary', y='Amount_invested_monthly', data=df_train, ax=ax)
    st.pyplot(fig)


elif selected_option == 'Pie Chart of Occupation Counts':
    st.subheader('Pie Chart of Occupation Counts')
    occupation_counts = df_train['Occupation'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(occupation_counts, labels=occupation_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

elif selected_option == 'Count Plot of Credit Mix':
    st.subheader('Count Plot of Credit Mix')
    fig, ax = plt.subplots()
    sns.countplot(x='Credit_Mix', data=df_train, order=df_train['Credit_Mix'].value_counts().index, hue='Credit_Mix', ax=ax, palette='pastel')
    st.pyplot(fig)
