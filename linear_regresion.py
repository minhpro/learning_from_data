from sklearn.datasets import load_diabetes 
dataset = load_diabetes(as_frame=True, scaled=False) 
input_features = dataset.feature_names 
print("The features in the input dataset are:", input_features) 
print(dataset.DESCR) 
data_df=dataset.data 
print("The input dataset is:") 
print(data_df.head()) 
disease_progression = dataset.target 
print("The output feature values are:") 
print(disease_progression.head()) 

# import matplotlib.pyplot as plt 
# plt.scatter(data_df["bmi"], disease_progression, color="midnightblue") 
# plt.title("BMI vs Diabetes Progression One Year After Baseline") 
# plt.xlabel("BMI") 
# plt.ylabel("Diabetes Progression") 
# plt.show()

X = data_df[["bmi"]] 
y = disease_progression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
untrained_model = LinearRegression()
trained_model = untrained_model.fit(X_train, y_train)

intercept=trained_model.intercept_ 
coefficient= trained_model.coef_ 
print("The intercept is:",intercept) 
print("The coefficient is:",coefficient) 

# Making predictions
y_predicted= trained_model.predict(X_test) 
print("The predicted values are:") 
print(y_predicted) 

# Evaluating the model
from sklearn.metrics import mean_squared_error, r2_score 
 
y_predicted= trained_model.predict(X_test) 
mse = mean_squared_error(y_test, y_predicted) 
r2 = r2_score(y_test, y_predicted) 
print("Mean squared error (MSE): %.2f" % mse) 
print("R² score: %.2f" % r2) 

# Multiple Linear Regression

X = data_df[["age","bmi","bp"]] 
y = disease_progression 

# Correlation analysis
correlation_df=X.corr() 
print("Correlation between features:") 
print(correlation_df) 


