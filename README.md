# Used Car Price Prediction 🚗💰


A machine learning regression project that predicts the price of used cars based on vehicle specifications, ownership history, condition, and other categorical attributes.


## 📌 Project Overview


This project uses a used-car dataset containing numerical and categorical features such as:


- Make year
- Mileage
- Engine capacity
- Fuel type
- Number of owners
- Brand
- Transmission
- Color
- Service history
- Accidents reported
- Insurance validity


The target variable is:


```text
price_usd

The project follows a complete machine learning workflow including data preprocessing, train-test splitting, feature transformation, model training, evaluation, model saving, and inference.

📊 Dataset

The dataset contains 10,000 car records.

Features
Feature	Type	Description
make_year	Numerical	Manufacturing year
mileage_kmpl	Numerical	Mileage in km/l
engine_cc	Numerical	Engine capacity
fuel_type	Categorical	Type of fuel
owner_count	Numerical	Number of previous owners
brand	Categorical	Car manufacturer
transmission	Categorical	Manual or Automatic
color	Categorical	Car color
service_history	Categorical	Service history information
accidents_reported	Numerical	Number of reported accidents
insurance_valid	Categorical	Whether insurance is valid
price_usd	Target	Used car price in USD
⚙️ Machine Learning Workflow

The project follows these steps:

Load the dataset
Check missing values
Analyze numerical feature correlations
Separate features and target
Split the dataset into training and testing sets
Identify numerical and categorical features
Build preprocessing pipelines
Handle missing values
Standardize numerical features
Apply One-Hot Encoding to categorical features
Train machine learning models
Evaluate models using 10-fold cross-validation
Select the best-performing model
Train the selected model on the full training set
Evaluate it on unseen test data
Save the trained model and preprocessing pipeline using Joblib
Perform inference on new/test data
🔧 Data Preprocessing
Numerical Features

The numerical pipeline uses:

SimpleImputer(strategy="median")
        ↓
StandardScaler()

Missing numerical values are replaced using the median, and numerical features are standardized.

Categorical Features

The categorical pipeline uses:

SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")

Missing categorical values are replaced by the most frequent category, and categorical features are converted into numerical binary columns.

handle_unknown="ignore" allows the model to safely process categories that were not seen during training.

🤖 Models Evaluated

The following regression models were evaluated:

Linear Regression
Decision Tree Regressor
Random Forest Regressor

Model performance was compared using Root Mean Squared Error (RMSE).

Evaluation Metric

RMSE measures the average magnitude of prediction errors, with larger errors receiving more penalty.

Lower RMSE = Better Model
📈 Model Evaluation

The models were evaluated using 10-fold cross-validation.

Current results:

Model	Mean CV RMSE
Linear Regression	~1005
Decision Tree Regressor	~1509
Random Forest Regressor	~1092

Based on the current cross-validation results, Linear Regression performed best with the lowest RMSE.

The final Linear Regression model achieved a test RMSE of approximately:

990.62
💾 Model Saving

The trained model and preprocessing pipeline are saved using Joblib:

joblib.dump(model, "model.pkl")
joblib.dump(full_pipeline, "pipeline.pkl")

Two files are created:

model.pkl
pipeline.pkl
model.pkl → trained machine learning model
pipeline.pkl → fitted preprocessing pipeline
🔮 Inference

During inference:

Load the saved model
Load the saved preprocessing pipeline
Read new input data
Separate the actual target value when evaluating test data
Apply the saved pipeline using transform()
Generate predictions
Compare predictions with actual values
Calculate final RMSE
Save predictions to output.csv

Example:

model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")


input_data = pd.read_csv("input.csv")


actual_values = input_data["price_usd"].copy()
input_features = input_data.drop("price_usd", axis=1)


transformed_input = pipeline.transform(input_features)
predictions = model.predict(transformed_input)


input_data = input_data.drop("price_usd", axis=1)


input_data["actual_price_usd"] = actual_values
input_data["predicted_price_usd"] = predictions
📁 Project Structure
DS_Proj-1/
│
├── main.py
├── used_car_price_dataset_extended.csv
├── input.csv
├── output.csv
├── model.pkl
├── pipeline.pkl
└── README.md
🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Joblib
🚀 Future Improvements

Possible improvements include:

Hyperparameter tuning
Feature importance analysis
Trying additional regression algorithms
Building a prediction interface
Deploying the model as a web applicatio