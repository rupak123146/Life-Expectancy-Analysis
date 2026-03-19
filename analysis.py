import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# -------------------------------
# Load Dataset
# -------------------------------
# Clean column names
data = pd.read_csv("Life Expectancy Data (1).csv")
data.columns = data.columns.str.strip()

# Select important columns
data = data[['Country','Life expectancy', 'GDP', 'Population', 'Schooling', 'BMI']]
# Drop missing values
data = data.dropna()

# -------------------------------
# Features and Target
# -------------------------------
X = data[['GDP', 'Population', 'Schooling', 'BMI']]
y = data['Life expectancy']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------------
# Model Training
# -------------------------------
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
error = mean_absolute_error(y_test, y_pred)
print("\nMean Absolute Error:", error)

# -------------------------------
# Sample Predictions
# -------------------------------
print("\nSample Predictions:")
for i in range(5):
    print("Predicted:", round(y_pred[i],2), "| Actual:", y_test.iloc[i])

# -------------------------------
# VISUALIZATIONS
# -------------------------------

# 1. Histogram
data['Life expectancy'].plot(kind='hist', bins=20)
plt.title("Life Expectancy Distribution")
plt.xlabel("Life Expectancy")
plt.show()

# 2. Scatter Plot (GDP vs Life Expectancy)
plt.scatter(data['GDP'], data['Life expectancy'])
plt.title("GDP vs Life Expectancy")
plt.xlabel("GDP")
plt.ylabel("Life Expectancy")
plt.show()

# 3. Bar Chart (Average Life Expectancy by Top Countries)
top = data.groupby('Country')['Life expectancy'].mean().sort_values(ascending=False).head(10)
top.plot(kind='bar')
plt.title("Top 10 Countries Life Expectancy")
plt.xticks(rotation=45)
plt.show()

# 4. Correlation Heatmap
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# USER INPUT PREDICTION
# -------------------------------
print("\n--- Try Your Own Input ---")

gdp = float(input("Enter GDP: "))
population = float(input("Enter Population: "))
schooling = float(input("Enter Schooling: "))
bmi = float(input("Enter BMI: "))

user_data = pd.DataFrame([[gdp, population, schooling, bmi]],
                         columns=['GDP', 'Population', 'Schooling', 'BMI'])

result = model.predict(user_data)

print("\nPredicted Life Expectancy:", round(result[0],2))