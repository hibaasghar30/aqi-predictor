#what this file does
#it opens our saved data and two different methods try to learn the pattern
#  and saves the pattern that learnedthe pattern well is used by the predictor
import hopsworks
import json   # saves notes about the winning model
import pandas as pd   #opens and reads our saved feature store data as a table
from sklearn.model_selection import train_test_split       # splits data into "practice" and "quiz" portions, so testing is fair
from sklearn.linear_model import Ridge                       # METHOD 1: one way of learning the aqi pattern
from sklearn.ensemble import RandomForestRegressor            # METHOD 2: a different way of learning the aqi pattern
from sklearn.preprocessing import StandardScaler               # adjusts numbers to a similar scale - Ridge needs this to learn well
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score   # STEP 3: three ways to score how good each method's guesses were
#model predicted AQI as 80, but the real AQI was 75. That's a mistake of 5.
#   1.  mean_absolute_error (MAE)
#Takes every mistake, ignores whether it was too high or too low, and just averages how far off 
#you were on average. If your model is consistently off by  5 points across all its predictions, MAE ≈ 5.



#     2.    mean_squared_error (used to get RMSE)
#Similar idea, but squares each mistake before averaging. Squaring a mistake of 5 gives 25;
#  squaring a mistake of 20 gives 400. This means big mistakes get punished much more harshly than small ones


#     3.     r2 score
#  instead of measuring the size of mistakes it measures what percentage of the pattern your  
#  model actually captured. It's a number from roughly 0 to 1:
#1= best 0= not good just guessing avg , negative= worst



import joblib                        # STEP 4: saves the winning method's learned pattern to a file
from src import config               # tells this file where to find our data, and where to save results




#opens file"feature_store" and loads the rows we have collected so far
def load_training_data():
    #log into hopsworks and get the feature store
    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()

    #get the same feature group we've been saving to
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        description="AQI and weather features for Karachi",
        time_travel_format="HUDI",
    )

    #read all the data currently in the feature group
    df = fg.read()
    print(f"Loaded {len(df)} rows from the feature store")

    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"{after} rows usable after dropping incomplete ones")

    return df



#--------------------------     TESTING    -------------------------
#testing
#df = load_training_data()
#print(df)


def train_and_evaluate():
    df = load_training_data()

#model willlearn form these columns (take input)
    feature_columns = ["pm2_5", "pm10", "co", "no2","so2", "o3", "temperature", "humidity" ,
                        "pressure" , "wind_speed", "hour", "day", "month" , "day_of_week"]


# model willpredict (output)

    target_column = "aqi"
    #saves the feature columns
    x = df[feature_columns]  
    #save the answers
    y = df[target_column]    

 

#split into practice data(train) and quiz data (test)
    x_train, x_test, y_train , y_test = train_test_split( x, y , test_size=0.2, random_state=42)


#----------------------------------      TRAINING RIDGE      -----------------------------

    scaler = StandardScaler()  #center values around zero because rigidlearns it better this way
    x_train_scaled = scaler.fit_transform(x_train)  #read the x_train values using .fit and center them around 0  usning _transform

    x_test_scaled = scaler.transform(x_test) #center the x_test values around zero and uses the same average


    ridge = Ridge()   #creates a empty ridge nmodel
    ridge.fit(x_train_scaled, y_train)  #reads wtv is stored in x_train_sclaed(the converted values that were changed to center around zero) and y_train (the naswers)


#predicts using only X_test_scaled — the quiz clues, which the already-trained ridge model has never seen before. The result (the guessed AQI values) gets stored in ridge_predictions.

    ridge_predictions = ridge.predict(x_test_scaled)

#----------        RMSE AND MAE ASKS HOW BIG WERE THE MISTAKES ON AVERAGE         ---------------
#it checks the "(y_test) the correct answers" to "(ridge_predictions) the predicted answers"  by squaring the mistakes and then take its square root
    ridge_rmse =mean_squared_error(y_test, ridge_predictions) **0.5


#instead of squaring the mistakes, mean_absolute_error just takes the plain, straightforward size of each mistake
    ridge_mae = mean_absolute_error(y_test, ridge_predictions)

#--------     R2 scores how smart the model's guesses were, from 0 (dumb) to 1 (perfect).     ----
    ridge_r2 = r2_score(y_test, ridge_predictions)
#how much did the model actually understand, not just how close were the guesses
#below 0 = the model is doing worse than just guessing the average


    print(f"Ridge - RMSE: {ridge_rmse:.2f}, MAE: {ridge_mae:.2f}, R2: {ridge_r2:.2f}")


#----------------------------------      TRAINING RANDOM FOREST      -----------------------------
    forest = RandomForestRegressor(random_state=42)
    forest.fit(x_train, y_train)

#test Random Forest on the quiz data
    forest_predictions = forest.predict(x_test)
#score how good Random Forest's guesses were
    forest_rmse = mean_squared_error(y_test, forest_predictions) ** 0.5
    forest_mae = mean_absolute_error(y_test, forest_predictions)
    forest_r2 = r2_score(y_test, forest_predictions)
    print(f"Random Forest - RMSE: {forest_rmse:.2f}, MAE: {forest_mae:.2f}, R2: {forest_r2:.2f}")



#compare both models - lower RMSE wins
    if forest_rmse < ridge_rmse:
        best_model = forest
        best_name = "random_forest"
        needs_scaler = False
    else:
        best_model = ridge
        best_name = "ridge"
        needs_scaler = True

    print(f"Winner: {best_name}")

    #save the winning model to disk
    joblib.dump(best_model, config.MODEL_FILE)

    #if ridge won, also save the scaler (predictor.py needs it to convert live data)
    if needs_scaler:
        joblib.dump(scaler, config.MODEL_DIR / "scaler.joblib")

    #save notes about the winner, so predictor.py knows how to use it
    metadata = {
        "best_model": best_name,
        "needs_scaler": needs_scaler,
        "feature_columns": feature_columns,
    }
    with open(config.MODEL_METADATA_FILE, "w") as f:
        json.dump(metadata, f)

    print("Model and metadata saved.")


#upload the saved model to Hopsworks Model Registry
    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )
    mr = project.get_model_registry()

    hw_model = mr.python.create_model(
        name="aqi_best_model",
        metrics={"rmse": min(ridge_rmse, forest_rmse)},
        description=f"Best AQI model - {best_name}",
    )

    #uploads everything inside the models/ folder (the joblib file + metadata json)
    hw_model.save(str(config.MODEL_DIR))

    print("Model uploaded to Hopsworks Model Registry.")

train_and_evaluate()
 