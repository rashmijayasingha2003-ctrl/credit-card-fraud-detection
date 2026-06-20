import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# ========================================
# LOAD AND PREPARE DATA
# ========================================

@st.cache_data
def load_data():
    df1 = pd.read_csv('Credit_Card_Dataset_2025_Sept_1.csv')
    df2 = pd.read_csv('Credit_Card_Dataset_2025_Sept_2.csv')
    df = pd.merge(df1, df2, left_on='ID', right_on='User')
    df = df.drop(columns=['User', 'Unnamed: 0'])
    number_columns = ['NO_OF_CHILD', 'INCOME', 'AGE',
                      'YEARS_EMPLOYED', 'FAMILY SIZE']
    for col in number_columns:
        df[col] = df[col].fillna(df[col].mean())
    text_columns = ['GENDER', 'CAR', 'REALITY', 'INCOME_TYPE',
                    'EDUCATION_TYPE', 'FAMILY_TYPE', 'HOUSE_TYPE']
    for col in text_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    df = df.drop_duplicates()
    return df

@st.cache_data
def train_models(df):
    df2 = df.copy()
    text_columns = ['GENDER', 'CAR', 'REALITY', 'INCOME_TYPE',
                    'EDUCATION_TYPE', 'FAMILY_TYPE', 'HOUSE_TYPE']
    for col in text_columns:
        le = LabelEncoder()
        df2[col] = le.fit_transform(df2[col])
    X = df2.drop(columns=['TARGET', 'ID'])
    y = df2['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_sc, y_train)
    model1 = LogisticRegression(random_state=42, max_iter=1000)
    model1.fit(X_train_smote, y_train_smote)
    model2 = DecisionTreeClassifier(random_state=42)
    model2.fit(X_train_smote, y_train_smote)
    return model1, model2, scaler, X_test_sc, y_test, X.columns

# ========================================
# MAIN APP
# ========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

df = load_data()
model1, model2, scaler, X_test, y_test, feature_cols = train_models(df)

# ========================================
# SIDEBAR NAVIGATION
# ========================================

st.sidebar.title("💳 Fraud Detection App")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go to Page:",
    ["📊 Data Overview",
     "📈 Charts & EDA",
     "🤖 Predict Fraud"]
)

# ========================================
# PAGE 1 - DATA OVERVIEW
# ========================================

if page == "📊 Data Overview":
    st.title("📊 Data Overview")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Fraud Cases", sum(df['TARGET'] == 1))
    with col4:
        st.metric("Normal Cases", sum(df['TARGET'] == 0))

    st.markdown("---")
    st.subheader("First 10 rows of data:")
    st.dataframe(df.head(10))

    st.markdown("---")
    st.subheader("Column Information:")
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Missing Values': df.isnull().sum().values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(info_df)

    st.markdown("---")
    st.subheader("Basic Statistics:")
    st.dataframe(df.describe())

# ========================================
# PAGE 2 - CHARTS AND EDA
# ========================================

elif page == "📈 Charts & EDA":
    st.title("📈 Charts & Exploratory Data Analysis")
    st.markdown("---")

    # Chart 1 - Fraud vs Normal (Univariate)
    st.subheader("1. Fraud vs Normal Transactions (Univariate)")
    fig, ax = plt.subplots(figsize=(6,4))
    df['TARGET'].value_counts().plot(kind='bar', color=['green','red'], ax=ax)
    ax.set_title('Fraud vs Not Fraud')
    ax.set_xlabel('0 = Normal, 1 = Fraud')
    ax.set_ylabel('Count')
    plt.xticks(rotation=0)
    st.pyplot(fig)
    st.info("📌 Only 420 out of 25,134 transactions are fraud (1.6%)")
    st.markdown("---")

    # Chart 2 - Gender (Univariate)
    st.subheader("2. Gender Distribution (Univariate)")
    fig, ax = plt.subplots(figsize=(6,4))
    df['GENDER'].value_counts().plot(kind='bar', color=['blue','pink'], ax=ax)
    ax.set_title('Male vs Female Customers')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Count')
    plt.xticks(rotation=0)
    st.pyplot(fig)
    st.markdown("---")

    # Chart 3 - Age Distribution (Univariate)
    st.subheader("3. Age Distribution (Univariate)")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(df['AGE'], bins=20, color='skyblue', edgecolor='black')
    ax.set_title('Age Distribution of Customers')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    st.info(f"📌 Average age: {round(df['AGE'].mean(), 1)} years")
    st.markdown("---")

    # Chart 4 - Income Distribution (Univariate)
    st.subheader("4. Income Distribution (Univariate)")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(df['INCOME'], bins=20, color='orange', edgecolor='black')
    ax.set_title('Income Distribution of Customers')
    ax.set_xlabel('Income')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    st.info(f"📌 Average income: {round(df['INCOME'].mean(), 1)}")
    st.markdown("---")

    # Chart 5 - Education Type (Univariate)
    st.subheader("5. Education Type (Univariate)")
    fig, ax = plt.subplots(figsize=(10,4))
    df['EDUCATION_TYPE'].value_counts().plot(kind='bar', color='purple', ax=ax)
    ax.set_title('Education Type of Customers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("---")

    # Chart 6 - Fraud by Gender (Bivariate)
    st.subheader("6. Fraud Count by Gender (Bivariate)")
    fig, ax = plt.subplots(figsize=(6,4))
    fraud_gender = df.groupby('GENDER')['TARGET'].sum()
    fraud_gender.plot(kind='bar', color=['blue','pink'], ax=ax)
    ax.set_title('Fraud Count by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Number of Frauds')
    plt.xticks(rotation=0)
    st.pyplot(fig)
    st.info("📌 Bivariate Analysis: Relationship between Gender and Fraud")
    st.markdown("---")

    # Chart 7 - Income by Fraud (Bivariate)
    st.subheader("7. Income by Fraud Status (Bivariate)")
    fig, ax = plt.subplots(figsize=(8,4))
    df.boxplot(column='INCOME', by='TARGET', ax=ax)
    ax.set_title('Income Distribution by Fraud Status')
    plt.suptitle('')
    ax.set_xlabel('0 = Normal, 1 = Fraud')
    ax.set_ylabel('Income')
    plt.tight_layout()
    st.pyplot(fig)
    st.info("📌 Bivariate Analysis: Shows relationship between Income and Fraud Status")
    st.markdown("---")

    # Chart 8 - Fraud by Income Type (Multivariate)
    st.subheader("8. Fraud vs Normal by Income Type (Multivariate)")
    fig, ax = plt.subplots(figsize=(12,5))
    fraud_income = df.groupby(['INCOME_TYPE', 'TARGET']).size().unstack(fill_value=0)
    fraud_income.plot(kind='bar', color=['green','red'], ax=ax)
    ax.set_title('Fraud vs Normal by Income Type')
    ax.set_xlabel('Income Type')
    ax.set_ylabel('Count')
    ax.legend(['Normal', 'Fraud'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    st.info("📌Multivariate Analysis: Relationship between Income Type, Count and Fraud Status")
    st.markdown("---")

    # Chart 9 - Correlation Heatmap (Multivariate)
    st.subheader("9. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,8))
    number_cols = df.select_dtypes(include='number')
    sns.heatmap(number_cols.corr(), annot=True, fmt='.2f',
                cmap='coolwarm', ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    st.info("📌 Multivariate Analysis: Values close to 1 or -1 = strong relationship. Values close to 0 = no relationship")

# ========================================
# PAGE 3 - PREDICT FRAUD
# ========================================

elif page == "🤖 Predict Fraud":
    st.title("🤖 Predict Fraud")
    st.markdown("---")

    st.subheader("Model Performance (Improved with SMOTE):")
    st.info("📌 SMOTE was applied to fix imbalanced data - helps model detect fraud better!")

    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    acc1 = round(accuracy_score(y_test, y_pred1) * 100, 2)
    acc2 = round(accuracy_score(y_test, y_pred2) * 100, 2)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Logistic Regression Accuracy", str(acc1) + "%")
    with col2:
        st.metric("Decision Tree Accuracy", str(acc2) + "%")

    st.markdown("---")
    st.subheader("Enter Customer Details:")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        car = st.selectbox("Has Car?", ["Y", "N"])
        reality = st.selectbox("Has Property?", ["Y", "N"])
        children = st.number_input("Number of Children", 0, 10, 0)
        income = st.number_input("Income", 0, 1000000, 150000)

    with col2:
        income_type = st.selectbox("Income Type",
            ["Commercial associate", "Pensioner",
             "State servant", "Student", "Working"])
        education = st.selectbox("Education Type",
            ["Academic degree",
             "Higher education",
             "Incomplete higher",
             "Lower secondary",
             "Secondary / secondary special"])
        family_type = st.selectbox("Family Type",
            ["Civil marriage", "Married",
             "Separated", "Single / not married", "Widow"])

    with col3:
        house_type = st.selectbox("House Type",
            ["Co-op apartment", "House / apartment",
             "Municipal apartment", "Office apartment",
             "Rented apartment", "With parents"])
        family_size = st.number_input("Family Size", 1, 10, 2)
        begin_month = st.number_input("Begin Month", -100, 0, -10)
        age = st.number_input("Age", 18, 100, 30)
        years_employed = st.number_input("Years Employed", 0, 50, 5)

    st.markdown("---")

    model_choice = st.selectbox(
        "Choose Model for Prediction:",
        ["Logistic Regression (Improved with SMOTE)",
         "Decision Tree (Improved with SMOTE)"]
    )

    if st.button("🔍 PREDICT NOW!", use_container_width=True):

        gender_enc = 1 if gender == "M" else 0
        car_enc = 1 if car == "Y" else 0
        reality_enc = 1 if reality == "Y" else 0

        income_map = {"Commercial associate":0, "Pensioner":1,
                      "State servant":2, "Student":3, "Working":4}
        edu_map = {"Academic degree":0, "Higher education":1,
                   "Incomplete higher":2, "Lower secondary":3,
                   "Secondary / secondary special":4}
        family_map = {"Civil marriage":0, "Married":1,
                      "Separated":2, "Single / not married":3, "Widow":4}
        house_map = {"Co-op apartment":0, "House / apartment":1,
                     "Municipal apartment":2, "Office apartment":3,
                     "Rented apartment":4, "With parents":5}

        input_data = np.array([[
            gender_enc,               # GENDER
            car_enc,                  # CAR
            reality_enc,              # REALITY
            children,                 # NO_OF_CHILD
            family_map[family_type],  # FAMILY_TYPE
            house_map[house_type],    # HOUSE_TYPE
            1,                        # FLAG_MOBIL
            0,                        # WORK_PHONE
            0,                        # PHONE
            0,                        # E_MAIL
            family_size,              # FAMILY SIZE
            begin_month,              # BEGIN_MONTH
            age,                      # AGE
            years_employed,           # YEARS_EMPLOYED
            income,                   # INCOME
            income_map[income_type],  # INCOME_TYPE
            edu_map[education]        # EDUCATION_TYPE
        ]])

        input_scaled = scaler.transform(input_data)

        if "Logistic" in model_choice:
            prediction = model1.predict(input_scaled)[0]
            prob = model1.predict_proba(input_scaled)[0]
            model_name = "Logistic Regression"
        else:
            prediction = model2.predict(input_scaled)[0]
            prob = model2.predict_proba(input_scaled)[0]
            model_name = "Decision Tree"

        st.markdown("---")
        st.subheader("Prediction Result:")

        if prediction == 1:
            st.error("🚨 FRAUD DETECTED! This transaction looks suspicious!")
            st.write("Model used:", model_name)
            st.write("Fraud Probability:", round(prob[1] * 100, 2), "%")
            st.write("Normal Probability:", round(prob[0] * 100, 2), "%")
        else:
            st.success("✅ NORMAL TRANSACTION! This looks safe!")
            st.write("Model used:", model_name)
            st.write("Normal Probability:", round(prob[0] * 100, 2), "%")
            st.write("Fraud Probability:", round(prob[1] * 100, 2), "%")