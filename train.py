import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import GradientBoostingClassifier

path = "dataset/Bank Customer Churn Prediction.csv"
df = pd.read_csv(path)

drop(columns=['customer_id'], inplace=True)

scaler = StandardScaler()

df = pd.get_dummies(df, columns=['gender', 'country'], drop_first=True)
df["credit_score"] = scaler.fit_transform(df[["credit_score"]])
target = 'churn'
#dropping the churn and credit_score column to make up the fetures
x = df.drop(columns=['churn'])
#target variable
y = df[target]
#splitting the data into tarin and test data
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                   random_state =42)
params2 = {
    "gradientboostingclassifier__n_estimators": range(20, 31, 5),
    "gradientboostingclassifier__max_depth": range(2, 5)
}
GBC = make_pipeline(GradientBoostingClassifier())

model = GridSearchCV(
    GBC,
    param_grid=params2,
    cv=5,
    n_jobs=-1,
    verbose=1
)
model
model.fit(X_train, y_train)
#checking the accuracy
train = model.score(X_train, y_train)
test =  model.score(X_test, y_test)

# print("Training Accuracy:", round(train3, 4))
# print("Test Accuracy:", round(test3, 4))
# scores3 = cross_val_score(clf2, X_train, y_train, cv=5, n_jobs=-1)
# print(scores3)
