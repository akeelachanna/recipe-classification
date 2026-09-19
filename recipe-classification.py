import pandas as pd
import numpy as np
from sklearn import svm

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

# Allows charts to appear in the notebook
#%matplotlib inline

# Pickle package
import pickle

# Read in muffin and cupcake ingredient data
recipes = pd.read_csv('cake,muffie.csv')
recipes

# =========================================
# 1. Import Libraries
# ==============================cake,muffie===========
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================================
# 2. Load Dataset
# =========================================
file_path = "recipess.xlsx"
df = pd.read_excel(file_path)

print("First 5 Rows:\n", df.head())

# =========================================
# 3. Basic EDA
# =========================================
print("\nDataset Info:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

# Fill missing values
df.fillna(0, inplace=True)

# =========================================
# 4. EDA GRAPHS
# =========================================

# Graph 1: Recipe Count
plt.figure()
df["Recipes"].value_counts().plot(kind='bar')
plt.title("Recipe Distribution")
plt.xlabel("Recipes")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Graph 2: Ingredient Usage
plt.figure()
df.drop("Recipes", axis=1).sum().plot(kind='bar')
plt.title("Ingredients Usage")
plt.xlabel("Ingredients")
plt.ylabel("Total")
plt.xticks(rotation=45)
plt.show()

# =========================================
# 5. Encoding Target
# =========================================
le = LabelEncoder()
df["Recipes"] = le.fit_transform(df["Recipes"])

# =========================================
# 6. Features & Target
# =========================================
X = df.drop("Recipes", axis=1)
y = df["Recipes"]

# =========================================
# 7. Train-Test Split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 8. MODEL 1: Decision Tree
# =========================================
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\n===== Decision Tree =====")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("\nClassification Report:\n", classification_report(y_test, dt_pred))

# =========================================
# 9. MODEL 2: Random Forest
# =========================================
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n===== Random Forest =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("\nClassification Report:\n", classification_report(y_test, rf_pred))

# =========================================
# 10. GRAPH: Accuracy Comparison
# =========================================
dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)

plt.figure()
plt.bar(["Decision Tree", "Random Forest"], [dt_acc, rf_acc])
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.show()

# =========================================
# 11. GRAPH: Feature Importance (Random Forest)
# =========================================
importance = rf_model.feature_importances_

plt.figure()
plt.bar(X.columns, importance)
plt.title("Feature Importance")
plt.xlabel("Ingredients")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()
cm = confusion_matrix(y_test, rf_pred)

plt.figure()
plt.imshow(cm)

# Plot two ingredients
sns.lmplot(x='Flour', y='Sugar', data=recipes, hue='Type',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70});
# Specify inputs for the model
# ingredients = recipes[['Flour', 'Milk', 'Sugar', 'Butter', 'Egg', 'Baking Powder', 'Vanilla', 'Salt']].as_matrix()
ingredients = recipes[['Flour','Sugar']].values
type_label = np.where(recipes['Type']=='Muffin', 0, 1)

# Feature names
recipe_features = recipes.columns.values[1:].tolist()
recipe_features
# Fit the SVM model
model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)
# Get the separating hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - (model.intercept_[0]) / w[1]

# Plot the parallels to the separating hyperplane that pass through the support vectors
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])
# Plot the hyperplane
sns.lmplot(x='Flour', y='Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black');
# Look at the margins and support vectors
sns.lmplot(x='Flour', y='Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='pink')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=80, facecolors='none');
# Create a function to guess when a recipe is a muffin or a cupcake
def muffin_or_cupcake(flour, sugar):
    if(model.predict([[flour, sugar]]))==0:
        print('You\'re looking at a muffin recipe!')
    else:
        print('You\'re looking at a cupcake recipe!')
# Predict if 50 parts flour and 20 parts sugar
muffin_or_cupcake(50, 20)
# Plot the point to visually see where the point lies
sns.lmplot(x='Flour', y='Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='pink')
plt.plot(50, 20, 'yo', markersize='9');
# Predict if 40 parts flour and 20 parts sugar
muffin_or_cupcake(40,20)
muffin_cupcake_dict = {'muffin_cupcake_model': model, 'muffin_cupcake_features': ['Flour','Sugar'], 'all_features': recipe_features}
muffin_cupcake_dict
# Pickle
pickle.dump(muffin_cupcake_dict, open("muffin_cupcake_dict.p", "wb"))
# S = String
pickle.dumps(muffin_cupcake_dict)

