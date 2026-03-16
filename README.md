Car Price Prediction Using Linear Regression
A beginner-friendly machine learning project that predicts the selling price of a car based on its features like brand, year, mileage, and fuel type.

About the Project
Ever wondered how much a used car should cost? This project tries to answer that using a machine learning model trained on real car data. You give it details about a car and it gives back an estimated price. It was built as a learning project to understand how Linear Regression works on a real dataset from data cleaning all the way to making predictions.

Dataset

Total Records — 2,500 cars
Source — CSV file (car_price_prediction_.csv)
Target Variable — Price (in USD)

FeatureDescriptionBrandCar manufacturer (Toyota, BMW, Tesla, etc.)YearYear the car was madeEngine SizeEngine size in litresFuel TypePetrol, Diesel, Electric, or HybridTransmissionManual or AutomaticMileageTotal kilometres drivenConditionNew, Used, or Like NewModelSpecific model name (Civic, Model S, etc.)

How It Works

Load the dataset and check for missing values
Drop unnecessary columns (Car ID)
Convert text columns to numbers using Label Encoding
Split data — 80% for training, 20% for testing
Train a Linear Regression model
Evaluate using MAE, RMSE, and R² score
Visualise actual vs predicted prices
Predict price interactively by entering car details


 Model Performance
MetricValueMAE$ 23,692RMSE$ 27,550R² Score− 0.0019

The R² score indicates that Linear Regression alone is not the strongest fit for this dataset — which is a useful finding in itself. Future versions could explore Random Forest or Gradient Boosting for better accuracy.


Project Structure
car_price_project/
├── main.py               # Runs the full training pipeline
├── data_loader.py        # Loads and inspects the dataset
├── preprocessor.py       # Cleans and encodes the data
├── model.py              # Trains, evaluates and plots results
├── predict_car.py        # Interactive price predictor
└── car_price_prediction_.csv   # Dataset

 Technologies Used

Python 3
Pandas
NumPy
Scikit-learn
Matplotlib


Getting Started
1. Clone the repository
git clone https://github.com/DivyanshDutt001/car-price-prediction.git
2. Install dependencies
pip install pandas numpy matplotlib scikit-learn
3. Run the training pipeline
python main.py
4. Run the interactive predictor
python predict_car.py

 Future Improvements

Apply feature scaling for better model performance
Try advanced models like Random Forest or XGBoost
Build a simple web interface using Flask or Streamlit
Add more features like car color, number of owners, and location


 Author
Made by Divyansh as part of a machine learning learning project.
Feel free to fork, use, or improve this project!

