import pandas as pd
import numpy as np
import joblib

# We don't need to import MinMaxScaler, since the loaded 'scaler' object
# is already an instance of it with all its learned data.

# --- 1. Load All Your Saved Assets ---
# You've already done this part perfectly.
print("Loading model assets...")
model_path = 'C:/Users/ravic/OneDrive/Desktop/PJS/ML/RL/Beginner/Project-1/artifacts/model_data.joblib'
model_data = joblib.load(model_path)

k_model = model_data['k_model']
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']  # This is your X_train.columns
cols_to_scale = model_data['cols_to_scale']
print("Assets loaded successfully.")


def make_prediction(location, area_type, availability, size, bath, balcony, converted_sqft):
    """
    Takes all raw user inputs and returns a price prediction.
    This function performs the *entire* pipeline:
    1. Scales the numerical inputs
    2. Predicts the cluster
    3. One-hot encodes the categorical inputs
    4. Aligns all columns to match the model's training data
    5. Predicts the final price
    """

    # --- Step 1: Create an Input DataFrame ---
    # We create a DataFrame with one row to match the structure of your original data
    # before you did any processing.

    # Create a dictionary of the raw inputs
    input_data = {
        'location': [location],
        'area_type': [area_type],
        'availability': [availability],
        'size': [size],
        'bath': [bath],
        'balcony': [balcony],
        'converted_sqft': [converted_sqft]
    }
    df_input = pd.DataFrame.from_dict(input_data)

    # --- Step 2: Scale Numerical Features (for Clustering) ---
    # The K-Means model (k_model) was trained on scaled data.
    # We must scale the user's inputs *before* we can get a cluster.

    # We must be *very* careful to only scale the columns that were
    # originally scaled. The 'cols_to_scale' list is crucial here.

    # Create a copy of the numerical data to scale
    numerical_inputs_for_cluster = df_input[cols_to_scale].copy()

    # Use the loaded 'scaler' to transform this data
    scaled_numerical_inputs = scaler.transform(numerical_inputs_for_cluster)

    # --- Step 3: Predict the Cluster ---
    # Use the K-Means model to predict which cluster this new house belongs to.
    cluster_prediction = k_model.predict(scaled_numerical_inputs)

    # --- Step 4: Add Cluster to Input DataFrame ---
    # This cluster is now a new feature, just like in your notebook.
    df_input['cluster'] = cluster_prediction

    # --- Step 5: Apply Scaling to the DataFrame ---
    # Your main model (XGBoost) was also trained on scaled data.
    # We'll replace the raw numerical values in our input DataFrame
    # with their scaled versions.

    # We create a new DataFrame from the scaled data, with the correct column names
    scaled_df = pd.DataFrame(scaled_numerical_inputs, columns=cols_to_scale)

    # Update the input DataFrame with these scaled values
    df_input[cols_to_scale] = scaled_df[cols_to_scale]

    # --- Step 6: One-Hot Encoding ---
    # Now we convert categorical columns (like 'location') into
    # many 0/1 columns (like 'location_Whitefield').
    # By default, pd.get_dummies() will correctly ignore the numerical
    # columns (like 'size', 'bath', 'cluster') and only encode text.
    df_encoded = pd.get_dummies(df_input)

    # --- Step 7: Align DataFrame to Training Columns ---
    # This is the most important step!
    # The user's input (df_encoded) might only have 10 columns
    # (e.g., 'location_Whitefield', 'area_type_Plot Area').
    # But your model was trained on 270+ columns (features).
    # We use .reindex() to create a final DataFrame that has the *exact*
    # same columns, in the *exact* same order, as your training data.
    # Any columns not in df_encoded (like 'location_Sarjapur Road')
    # will be filled with 0.

    df_final = df_encoded.reindex(columns=features, fill_value=0)

    # --- Step 8: Make the Final Prediction ---
    # Use the main XGBoost model to predict the price.
    final_price_prediction = model.predict(df_final)

    # The model returns a list, so we get the first (and only) item.
    return final_price_prediction[0]


# --- 9. Example of how to use this function ---
# if __name__ == '__main__':
#     # This is a test run to see if it works.
#     # Your Streamlit app will get these values from the user.
#     print("Running a test prediction...")
#
#     # Example inputs (you must use real values from your data)
#     test_location = 'Whitefield'  # Must be a location in your original data
#     test_area_type = 'Super built-up  Area'  # Must be a real area_type
#     test_availability = 'Ready To Move'  # Must be a real value
#     test_size = 3
#     test_bath = 3
#     test_balcony = 2
#     test_sqft = 1760
#
#     predicted_price = make_prediction(
#         location=test_location,
#         area_type=test_area_type,
#         availability=test_availability,
#         size=test_size,
#         bath=test_bath,
#         balcony=test_balcony,
#         converted_sqft=test_sqft
#     )
#
#     print(f"Test prediction successful.")
#     print(f"Predicted Price (in Lakhs): {predicted_price:.2f}")

