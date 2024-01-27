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

# Add a picture to the header
header_image = "credit-score.jpeg" 
st.image(header_image, use_column_width=True)


# Load data
df_train = pd.read_csv("train.csv", low_memory=False)
df_test = pd.read_csv("test.csv", low_memory=False)


# Set up the sidebar
st.sidebar.title('Developer Information')
st.sidebar.text('Welcome to my Streamlit app!')
# Display name at the top of the sidebar
st.sidebar.subheader('Developer:')
st.sidebar.text('Zohreh Taghibakhshi')
#link to GitHub profile
st.sidebar.markdown('[GitHub](https://github.com/zizo155)')





# Display the head of the data
st.subheader('Raw Data - Head')
st.write(df_train.head())

# Display list of columns
st.subheader('List of Columns')
columns_table = pd.DataFrame(df_train.columns.tolist(), columns=["Columns"])
st.table(columns_table)


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

st.markdown("<hr style='border:2px solid black'>", unsafe_allow_html=True)


st.markdown("<h2 style='text-align: center;'>Plots</h2>", unsafe_allow_html=True)


# dropdown menu to the sidebar
selected_option = st.sidebar.selectbox('Select a Plot to Display', ['Histogram for Age', 'Count Plot for Occupation', 'Correlation Heatmap', 'Bar Plot of Annual Income', 
                                                            'Scatterplot of Monthly Inhand Salary vs. Amount invested monthly' , 'Pie Chart of Occupation Counts',
                                                            'Count Plot of Credit Mix'])

# Main content
st.subheader(f'{selected_option}')


# Plots
if selected_option == 'Histogram for Age':
    fig, ax = plt.subplots()
    sns.histplot(df_train['Age'], kde=True, ax=ax)
    ax.set_title('Distribution of Age')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)


elif selected_option == 'Count Plot for Occupation':
    fig, ax = plt.subplots()
    sns.countplot(x='Occupation', data=df_train, hue='Occupation', palette='pastel', ax=ax)
    ax.set_title('Count of Individuals in Each Occupation')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)


elif selected_option == 'Correlation Heatmap':
    fig, ax = plt.subplots(figsize=(12, 10))
    numeric_columns = df_train.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df_train[numeric_columns].corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

elif selected_option == 'Bar Plot of Annual Income':
    fig, ax = plt.subplots()
    sns.barplot(x='Credit_Score', y='Annual_Income', data=df_train, estimator=np.median, ax=ax)
    st.pyplot(fig)

elif selected_option == 'Scatterplot of Monthly Inhand Salary vs. Amount invested monthly':
    fig, ax = plt.subplots()
    sns.scatterplot(x='Monthly_Inhand_Salary', y='Amount_invested_monthly', data=df_train, ax=ax)
    st.pyplot(fig)


elif selected_option == 'Pie Chart of Occupation Counts':
    occupation_counts = df_train['Occupation'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(occupation_counts, labels=occupation_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

elif selected_option == 'Count Plot of Credit Mix':
    fig, ax = plt.subplots()
    sns.countplot(x='Credit_Mix', data=df_train, order=df_train['Credit_Mix'].value_counts().index, hue='Credit_Mix', ax=ax, palette='pastel')
    st.pyplot(fig)

# Draw a line to separate the plots from machine learning part
st.markdown("<hr style='border:2px solid black'>", unsafe_allow_html=True)

# Machine Learning Section Header
st.markdown("<h2 style='text-align:center;'>Machine Learning</h2>", unsafe_allow_html=True)

machine_learning_image = "Machine_Learning.jpeg" 
st.image(machine_learning_image, use_column_width=True)

st.markdown("<p style='text-align:center;'>Please select a machine learning model from the sidebar dropdown and click on the button below it to run the selected model.</p>", unsafe_allow_html=True)


st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

# Machine Learning 

#dropdown menu for machine learning models
selected_model = st.sidebar.selectbox('Select a machine learning model',
                                      ['Decision Tree', 
                                       'Random Forest', 
                                       'Gradient Boosting', 
                                       'k-Nearest Neighbors'])

df_train.drop(['Month', 'Occupation', 'Type_of_Loan', 'Payment_of_Min_Amount'], axis=1, inplace=True)
df_test.drop(['Month', 'Occupation', 'Type_of_Loan', 'Payment_of_Min_Amount'], axis=1, inplace=True)

# One-hot encode the object columns
df_train = pd.get_dummies(df_train, columns=['Credit_Mix', 'Payment_Behaviour'])
df_test = pd.get_dummies(df_test, columns=['Credit_Mix', 'Payment_Behaviour'])

# Initialize StandardScaler
scaler = StandardScaler()

# Define numeric columns
cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary','Changed_Credit_Limit', 'Num_Credit_Inquiries',
                'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
                'Amount_invested_monthly', 'Monthly_Balance', 'Credit_History_Age_Months']

# Apply StandardScaler to training data
df_train[cols] = scaler.fit_transform(df_train[cols])

# Apply the same scaler to test data
df_test[cols] = scaler.transform(df_test[cols])

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Apply label encoding to the "Credit_Score" column
df_train['Credit_Score'] = label_encoder.fit_transform(df_train['Credit_Score'])

# Prepare Data
X = df_train.drop('Credit_Score', axis=1)
y = df_train['Credit_Score']
X_test = df_test.copy()

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Perform the selected machine learning model
if st.sidebar.button('Run Selected Model'):
    if selected_model == 'Decision Tree':
        # Run Decision Tree model
        st.markdown("<p style='text-align:center;'>Performing Decision Tree model...</p>", unsafe_allow_html=True)

        # Build and Train the Decision Tree Model
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_train, y_train)
        # Predictions on Validation Set
        y_pred_val_dt = dt_model.predict(X_val)
        # Evaluation
        val_accuracy_dt = accuracy_score(y_val, y_pred_val_dt)
        st.write(f'Validation Accuracy: {val_accuracy_dt:.4f}')
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred_val_dt)
        st.write("Confusion Matrix:")
        st.write(cm)

        # Classification Report
        # Parse and format the classification report
        report_dict_dt = classification_report(y_val, y_pred_val_dt, output_dict=True)
        report_df_dt = pd.DataFrame(report_dict_dt).transpose()

        # Display the formatted classification report
        st.write("Classification Report:")
        st.write(report_df_dt)
        
        # Predict for Unseen Data (df_test)
        predictions_test_dt = dt_model.predict(X_test)

    elif selected_model == 'Random Forest':
        # Run Random Forest model
        st.markdown("<p style='text-align:center;'>Performing Random Forest model...</p>", unsafe_allow_html=True)

        # Build and Train the Random Forest Model
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        # Predictions on Validation Set
        y_pred_val_rf = rf_model.predict(X_val)
        # Evaluation
        val_accuracy_rf = accuracy_score(y_val, y_pred_val_rf)
        st.write(f'Validation Accuracy (Random Forest): {val_accuracy_rf:.4f}')
        # Confusion Matrix
        cm_rf = confusion_matrix(y_val, y_pred_val_rf)
        st.write("Confusion Matrix (Random Forest):")
        st.write(cm_rf)

        # Classification Report
        # Parse and format the classification report
        report_dict_rf = classification_report(y_val, y_pred_val_rf, output_dict=True)
        report_df_rf = pd.DataFrame(report_dict_rf).transpose()

        # Display the formatted classification report
        st.write("Classification Report:")
        st.write(report_df_rf)


        # Predict for Unseen Data (df_test) using Random Forest
        predictions_test_rf = rf_model.predict(X_test)


    elif selected_model == 'Gradient Boosting':
        # Run Gradient Boosting model
        st.markdown("<p style='text-align:center;'>Performing Gradient Boosting model...</p>", unsafe_allow_html=True)

        # Build and Train the Gradient Boosting Model
        gb_model = GradientBoostingClassifier(random_state=42)
        gb_model.fit(X_train, y_train)
        # Predictions on Validation Set
        y_pred_val_gb = gb_model.predict(X_val)
        # Evaluation
        val_accuracy_gb = accuracy_score(y_val, y_pred_val_gb)
        st.write(f'Validation Accuracy (Gradient Boosting): {val_accuracy_gb:.4f}')
        # Confusion Matrix
        cm_gb = confusion_matrix(y_val, y_pred_val_gb)
        st.write("Confusion Matrix (Gradient Boosting):")
        st.write(cm_gb)
        
        # Classification Report
        # Parse and format the classification report
        report_dict_gb = classification_report(y_val, y_pred_val_gb, output_dict=True)
        report_df_gb = pd.DataFrame(report_dict_gb).transpose()

        # Display the formatted classification report
        st.write("Classification Report:")
        st.write(report_df_gb)

        # Predict for Unseen Data (df_test) using Gradient Boosting
        predictions_test_gb = gb_model.predict(X_test)

    elif selected_model == 'k-Nearest Neighbors':
        # Run k-Nearest Neighbors model
        st.markdown("<p style='text-align:center;'>Performing k-Nearest Neighbors model...</p>", unsafe_allow_html=True)

        # Build and Train the k-Nearest Neighbors Model
        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)
        # Predictions on Validation Set
        y_pred_val_knn = knn_model.predict(X_val)
        # Evaluation
        val_accuracy_knn = accuracy_score(y_val, y_pred_val_knn)
        st.write(f'Validation Accuracy (k-Nearest Neighbors): {val_accuracy_knn:.4f}')
        # Confusion Matrix
        cm_knn = confusion_matrix(y_val, y_pred_val_knn)
        st.write("Confusion Matrix (k-Nearest Neighbors):")
        st.write(cm_knn)

        # Classification Report
        # Parse and format the classification report
        report_dict_knn = classification_report(y_val, y_pred_val_knn, output_dict=True)
        report_df_knn = pd.DataFrame(report_dict_knn).transpose()

        # Display the formatted classification report
        st.write("Classification Report:")
        st.write(report_df_knn)

        # Predict for Unseen Data (df_test) using k-Nearest Neighbors
        predictions_test_knn = knn_model.predict(X_test)






st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Models Accuracy Comparison</h2>", unsafe_allow_html=True)
model_accuracy_image = "model_accuracy.png" 
st.image(model_accuracy_image, use_column_width=True)

st.markdown("<br>", unsafe_allow_html=True)  
st.markdown("<hr style='border:2px solid gray'>", unsafe_allow_html=True) 


st.markdown("<p style='text-align: center; font-size: 16px;'>Explore the full dataset on Kaggle:</p>", unsafe_allow_html=True)


st.markdown("<p style='text-align: center; font-size: 16px;'><a href='https://www.kaggle.com/datasets/parisrohan/credit-score-classification/' target='_blank'>Kaggle Dataset</a></p>", unsafe_allow_html=True)
