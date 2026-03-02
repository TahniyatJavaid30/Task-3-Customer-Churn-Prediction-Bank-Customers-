# Task-3-Customer-Churn-Prediction-Bank-Customers-
Identify customers who are likely to leave the bank.


# 🏦 Customer Churn Prediction – Bank Customers

## 📌 Objective

The objective of this project is to:

* Identify customers who are likely to **leave the bank (churn)**.
* Perform **data cleaning and preprocessing**.
* Encode categorical features properly.
* Train a **classification model**.
* Analyze **feature importance** to understand what drives churn.

This is a **Supervised Binary Classification** problem.

---

## 📊 Dataset

Dataset used: **Churn Modelling Dataset**

The dataset contains customer information such as:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Estimated Salary
* Exited (Target Variable)

Target Variable:

* `Exited`

  * 1 → Customer Left the Bank
  * 0 → Customer Stayed

---

## 🛠️ Technologies Used

* Python 🐍
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📂 Project Structure

```id="ozh7d8"
customer-churn-prediction/
│
├── churn_model.ipynb
├── Churn_Modelling.csv
├── requirements.txt
└── README.md
```

---

# 🔎 Project Workflow

---

## 1️⃣ Data Loading

```python id="3slc9x"
import pandas as pd

df = pd.read_csv("Churn_Modelling.csv")
df.head()
```

---

## 2️⃣ Data Cleaning & Preparation

Steps performed:

* Removed irrelevant columns (RowNumber, CustomerId, Surname)
* Checked missing values
* Verified data types

```python id="7kclq2"
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
df.isnull().sum()
```

---

# 🔄 Encoding Categorical Features

We encoded:

* **Gender** → Label Encoding
* **Geography** → One-Hot Encoding

```python id="4x9nla"
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

df = pd.get_dummies(df, columns=["Geography"], drop_first=True)
```

✔ Label Encoding converts categories into numeric labels.
✔ One-Hot Encoding prevents ordinal relationship issues.

---

# 📊 Exploratory Data Analysis (EDA)

## 📍 Churn Distribution

![Image](https://www.researchgate.net/publication/384244960/figure/fig1/AS%3A11431281279488747%401727050372810/Bar-plot-for-customer-churn-distribution.jpg)

![Image](https://miro.medium.com/1%2AAMHr4qoxJa4r3XLY2XYa1w.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1016/1%2AheddYfxvMZ7CR3TaO1uORA.png)

![Image](https://365datascience.com/resources/blog/jq4m1s0l9g-customer-churn-screenshot6.png)

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="Exited", data=df)
plt.title("Churn Distribution")
plt.show()
```

---

## 📍 Age vs Churn

![Image](https://www.kaggleusercontent.com/kf/50448306/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..hxlbIQVGgJZK-trLNu8VQA.jgndRZKpxrfXSI8ULYEYZHKHn-xsfbEKbZMmgs896ZrFUUKbYKQKb3cgPGnba-14xnx1pSXSWvEfRBxKw5dL04uII6Pta97qy5owuye2f61vuZs8ISh7_MCJ3OEPiCS3YHNXOpHnSJ8j_UVJAgaSOpgHtnGsL39InABWbwwC1b5OO6uzO4F1tS8cQ1mbuqzJCaWtT5oIOEGoQkcYeA_DAbDex6j0eTU2pLnQfgY-YDssnxCdEarDYfLBzqAtm8aKJffFCvh4K1tUrOSEm66bmTFJAW5rVnGGuuLB0Jq-nTboBvldhV206rOUQnnCsze7ESzJ1tea9ffBypcXzEeCnkIytHGWnED5hDQado27ja7hwv-ohs7X2f3iHHiFpbIlCQeKHnGSCTlx40qsXlMIfQOmQl3kuiB6KutaYBHr3HnDg1HQClmCYT3H0qJOOrhRiAChdNRbXg3BTbLzCXOCT21XifSGzpJmWILhtOv8oRhy7IkZg3XEa7oQQW7C5KuUVy0OfIFAF0zfPYPZS4svqJIiRfgEPiVDnh73IvtnMfMQbjSMaNV89Qgg4Zeh2Cm6WeYW2eBBFXemAG0Jo8u4TA-7_83BxGJ07cxVKwU8ESr5WBltlAb_xX25ROApEGW6IEc0R6i8_ejFlL7jcIdg7A.qBTpI3-ArBRNxeKlT0FsqQ/__results___files/__results___23_0.png)

![Image](https://www.ijraset.com/images/text_version_uploads/4_467.png)

![Image](https://media.licdn.com/dms/image/v2/D5612AQEWPE7SUxrAZQ/article-cover_image-shrink_720_1280/B56ZfRKjKeHQAI-/0/1751560893737?e=2147483647\&t=oxydO0uFZRM5LLq0EbHWbsm7iGpnN2yUSiyhp0t9hXg\&v=beta)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2AFu9B0S6vI8PVWp6HGLZyZA.png)

```python
sns.boxplot(x="Exited", y="Age", data=df)
plt.title("Age vs Churn")
plt.show()
```

---

# 🤖 Model Training

We trained a **Random Forest Classifier** to predict churn.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 📈 Model Evaluation

```python
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

Model achieved good classification accuracy in predicting customer churn.

---

# 🔍 Feature Importance Analysis

One major advantage of Random Forest is feature importance extraction.

```python
import pandas as pd

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance)
```

## 📊 Feature Importance Visualization

![Image](https://www.researchgate.net/publication/384017993/figure/fig2/AS%3A11431281282857456%401728526545583/Feature-importance-plot-of-the-random-forest-model-according-to-variables-weights.png)

![Image](https://www.researchgate.net/publication/396717958/figure/fig2/AS%3A11431281685183631%401761025726443/Feature-importance-for-churn-prediction-using-Grabit-model.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2A-GGDhUQV8ZTzsiSR8iEfWQ.png)

![Image](https://www.researchgate.net/publication/335778141/figure/fig2/AS%3A802452496478208%401568330992277/Feature-importance-plot-for-the-A-machine-learning-model-and-B-machine-learning-model.png)

```python
feature_importance.plot(kind="bar")
plt.title("Feature Importance")
plt.show()
```

---

# 🎯 Key Insights

* Age is a strong indicator of churn.
* Customers with higher balances are more likely to churn.
* Active members are less likely to leave.
* Geography impacts churn probability.
* Number of products influences customer retention.

---

# 🧠 Skills Demonstrated

✔ Data Cleaning & Preprocessing
✔ Categorical Data Encoding
✔ Supervised Machine Learning
✔ Random Forest Classification
✔ Feature Importance Interpretation
✔ Model Evaluation

---

# 🚀 Future Improvements

* Try Logistic Regression / XGBoost
* Perform Hyperparameter Tuning
* Handle class imbalance (SMOTE)
* Deploy using Flask / FastAPI
* Build dashboard using Streamlit

---

# 🏁 Conclusion

This project demonstrates a complete **end-to-end churn prediction pipeline**, including:

* Data preprocessing
* Encoding categorical variables
* Model training
* Feature importance analysis

It provides practical experience in solving real-world **Customer Retention Problems** used in banking and fintech industries.
