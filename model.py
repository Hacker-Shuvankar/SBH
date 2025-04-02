import base64
import io
from io import BytesIO
import json
from supabase import create_client
import tenseal as ts
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
label_encoder = LabelEncoder()

url='https://lemgbotvsawfhnhvslsm.supabase.co'
key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxlbWdib3R2c2F3ZmhuaHZzbHNtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM0NDYzMTcsImV4cCI6MjA1OTAyMjMxN30.liKQcW9kuC4JQKdtQhnRF721OWIoygFpLuGSQ0YfRrA'
supabase = create_client(url, key)

# model_save_path = "D:/My_Projects/SBH/model"

def upload_files(file_name ,file_content):
    bucket_name = "encrypted-file"  # Specify your bucket name

    try:
        # List all files in the bucket
        existing_files = supabase.storage.from_(bucket_name).list()

        # Check if the file already exists in the bucket
        file_exists = any(f['name'] == file_name for f in existing_files)

        # If the file exists, delete it first
        if file_exists:
            delete_res = supabase.storage.from_(bucket_name).remove([file_name])
            print(f"Deleted existing file: {file_name} - Response: {delete_res}")

        # Upload the new file
        upload_re = supabase.storage.from_(bucket_name).upload(file_name, file_content)
        print(upload_re)

        return {"success": True, "message": f"File '{file_name}' uploaded successfully"}

    except Exception as e:
        return {"error": str(e)}

# Load multiple CSV files from a folder
def load_multiple_csv():
    bucket_name = "encrypted-file"

    try:
        # Fetch all files in the Supabase bucket
        files_res = supabase.storage.from_(bucket_name).list()
        # Ensure `files_res` is a list (not a response object)
        if not isinstance(files_res, list):
            return {"error": "Unexpected response format from Supabase"}

        # Extract only CSV files
        csv_files = [file['name'] for file in files_res if file['name'].endswith('.csv')]

        # List to store DataFrames
        all_dfs = []

        # Download each CSV file and convert to DataFrame
        for csv_file in csv_files:
            file_content = supabase.storage.from_(bucket_name).download(csv_file)

            if file_content:  # Ensure content is not empty
                df = pd.read_csv(io.BytesIO(file_content))  # Read CSV from binary data
                all_dfs.append(df)
                print(f"✅ Downloaded & Loaded: {csv_file}")
            else:
                print(f"⚠️ Error downloading {csv_file}")

        # Combine all DataFrames
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            return combined_df
        else:
            return {"error": "No CSV files found in the bucket"}

    except Exception as e:
        return {"error": str(e)}

def load():
    # Specify the folder path containing CSV files
    # folder_path = "D:/My_Projects/SBH/uploads"  # Update with the correct path
    df = load_multiple_csv()

    # Identify the outcome column (last column assumed as target)
    outcome_column = df.columns[-1]  # Automatically detect last column as target


    label_encoders = {}
    # Encode all columns with dtype 'object'

    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])
        if column == outcome_column:
            label_encoders[column] = label_encoder
    if outcome_column not in label_encoders:
        label_encoders[outcome_column] = label_encoder.fit(df[outcome_column])  # Convert categories to numbers

    # Print label_encoders to verify
    print(label_encoders)

    # Save only the classes of each LabelEncoder (not the full LabelEncoder object)
    encoded_labels = {col: le.classes_.tolist() for col, le in label_encoders.items()}
    # # Save the label encoders as Pickle, but only the class labels (not the entire LabelEncoder object)
    # joblib.dump(encoded_labels, os.path.join(model_save_path, "label_encoders.pkl"))

    # Convert the encoded_labels to JSON string for Supabase insertion
    encoded_labels_json = json.dumps(encoded_labels)

    # Store Pickle File in Memory (No Local Save)
    buffer = BytesIO()
    joblib.dump(encoded_labels, buffer)
    buffer.seek(0)  # Reset buffer position to the beginning
    pickle_data = buffer.getvalue()

    # Delete duplicate file from Supabase if exists
    try:
        delete_res = supabase.storage.from_("encrypted-file").remove(["label_encoders.pkl"])
        print(f"🗑️ Attempted to delete label_encoders.pkl. Response: {delete_res}")
    except Exception as e:
        print(f"⚠️ Could not delete : {e}")

    # Upload the Pickle File to Supabase Storage
    res = supabase.storage.from_("encrypted-file").upload(
        "label_encoders.pkl",
        pickle_data,
        file_options={"content-type": "application/octet-stream"}
    )

    supabase.table('file_storage').delete().neq("labelencoder_pkl", None).execute()
    res=supabase.table('file_storage').insert({"labelencoder_pkl": encoded_labels_json}).execute()

    response = supabase.storage.from_("encrypted-file").download("label_encoders.pkl")

    # Load pickle file from BytesIO
    encoded_labels = joblib.load(BytesIO(response))
    print(res)
    print("reterive data: ",encoded_labels)

    # Create encryption context
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, coeff_mod_bit_sizes=[40, 20, 40])
    context.global_scale = 2 ** 15
    context.generate_galois_keys()
    context.generate_relin_keys()
    return df , outcome_column, context

# Step 1: Encrypt the dataset
def encrypt_data(dataf, context, outcome_col):
    encrypted_data = []
    for _, row in dataf.iterrows():
        encrypted_vector = ts.ckks_vector(context, row.drop(outcome_col).tolist()).serialize()
        encrypted_data.append(base64.b64encode(encrypted_vector).decode('utf-8'))  # Convert to base64 string
    return encrypted_data, dataf[outcome_col].tolist()

def encrypt_data1(df, context, outcome_column):
    encrypted_data = []
    for _, row in df.iterrows():
        encrypted_data.append(ts.ckks_vector(context, row.drop(outcome_column).tolist()))
    return encrypted_data, df[outcome_column].tolist()

def fetch_encrypted_data():
    bucket_name = "encrypted-file"

    try:
        # List all files in the bucket
        files_res = supabase.storage.from_(bucket_name).list()

        if not isinstance(files_res, list):
            return {"error": "Unexpected response format from Supabase"}

        # Filter only JSON files
        json_files = [file['name'] for file in files_res if file['name'].endswith('.json')]

        if not json_files:
            return {"error": "No JSON files found in the bucket"}


        return json_files # Dictionary with file names as keys and content as values

    except Exception as e:
        return {"error": str(e)}

def fetch_content(filename):
    file_res = supabase.storage.from_("encrypted-file").download(filename)
    return file_res

def save_encrypted_data(encrypted_data, labels, filename):
    # os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist
    # filepath = os.path.join(folder, filename)  # Full path
    # with open(filepath, "w") as f:
    #     json.dump({"features": encrypted_data, "labels": labels}, f)  # Save as JSON


    json_data = json.dumps({"features": encrypted_data, "labels": labels})
    # Convert JSON string to raw bytes
    file_bytes = json_data.encode("utf-8")

        # Define the Supabase Storage bucket name
    bucket_name = "encrypted-file"  # Change this to your actual bucket name

    try:
        delete_res = supabase.storage.from_(bucket_name).remove([filename])
        print(f"🗑️ Attempted to delete {filename}. Response: {delete_res}")
    except Exception as e:
        print(f"⚠️ Could not delete {filename}: {e}")

        # Upload the file directly from memory using binary data
    res = supabase.storage.from_(bucket_name).upload(filename, file_bytes, file_options={"content-type": "application/json"})
    print("upload encrpted :", res)
    # print(f"✅ Encrypted data saved at: {filepath}")



# Encrypt the dataset
# encrypted_features, labels = encrypt_data(df, context, outcome_column)
def load_model(df, outcome_column, context):
    # Split dataset into training and testing sets (80-20 split)
    X = df.drop(columns=[outcome_column])
    y = df[outcome_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Scale test data with same scaler

    # Train logistic regression model
    model = LogisticRegression(solver='saga', max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # **Compute Accuracy**
    y_pred = model.predict(X_test_scaled)
    print("Y predict ---:",y_pred)
    accuracy = accuracy_score(y_test, y_pred)  # Accuracy calculation

    # Encrypt model coefficients
    encrypted_weights = ts.ckks_vector(context, model.coef_[0])
    encrypted_bias = ts.ckks_vector(context, [model.intercept_[0]])

    # Save model
    # model_save_path = "D:/My_Projects/SBH/model"
    # joblib.dump(model, os.path.join(model_save_path, "logistic_regression_model.pkl"))

    model_metadata = json.dumps({"model_filename": "logistic_regression_model.pkl"})
    supabase.table('file_storage').delete().eq("model_pkl", "logistic_regression_model.pkl").execute()
    res = supabase.table('file_storage').insert({"model_pkl": model_metadata}).execute()
    print(res)

    return encrypted_weights, encrypted_bias, accuracy*100, y_pred

# Step 2: Perform encrypted inference
def encrypted_predict(encrypted_features, encrypted_weights, encrypted_bias):
    encrypted_predictions = []
    for enc_x in encrypted_features:
        enc_y = enc_x.dot(encrypted_weights) + encrypted_bias
        encrypted_predictions.append(enc_y)
    return encrypted_predictions

# encrypted_predictions = encrypted_predict(encrypted_features, encrypted_weights, encrypted_bias)


# Step 3: Decrypt the predictions
def decode(encrypted_predictions):
    response = supabase.storage.from_("encrypted-file").download("label_encoders.pkl")

    # Load the pickle file from the response
    encoded_labels = joblib.load(BytesIO(response))

    # Print out encoded_labels to debug
    print("Encoded Labels:", encoded_labels)

    # Ensure it's a valid dictionary with list of classes
    if not isinstance(encoded_labels, dict):
        raise ValueError("encoded_labels should be a dictionary")

    # Recreate the LabelEncoders using the loaded data
    label_encoders = {}
    for col, classes in encoded_labels.items():
        if isinstance(classes, list) or isinstance(classes, np.ndarray):
            label_encoders[col] = LabelEncoder().fit(classes)
        else:
            print(f"Invalid classes for column {col}: {classes}")
            raise ValueError(f"Classes for column {col} must be a list or ndarray")

    # Get any label encoder (assuming all columns use the same classes, or pick the first one)
    first_label_encoder = next(iter(label_encoders.values()))

    # Get the number of classes in the encoder
    num_classes = len(first_label_encoder.classes_)

    # Ensure encrypted predictions are within valid range (0 to num_classes-1)
    decrypted_predictions = np.clip(encrypted_predictions, 0, num_classes - 1)

    # Inverse transform the predictions to the original labels
    decoded_labels = []
    for p in decrypted_predictions:
        try:
            decoded_labels.append(first_label_encoder.inverse_transform([p])[0])
        except ValueError:
            decoded_labels.append("Unknown")  # If the value is out of bounds for inverse_transform

    print("Decoded labels:", decoded_labels)
    return decoded_labels


# print("Decrypted Predictions:", decoded_predictions)


# # Load the saved model and test with new unseen data
# def load_and_test_model(model_folder_path, test_folder_path):
#     model = joblib.load(os.path.join(model_folder_path, "logistic_regression_model.pkl"))
#     label_encoder = joblib.load(os.path.join(model_folder_path, "label_encoder.pkl"))
#     test_df = load_multiple_csv(test_folder_path)
#     test_features = test_df.values  # Assuming no target column in test set
#     predictions = model.predict(test_features)
#     decoded_predictions = label_encoder.inverse_transform(predictions)
#     print("Test Predictions:", decoded_predictions)
#
# # Example usage
# model_folder_path = "D:/My_Projects/SBH/model"  # Update with the correct model path
# test_folder_path = "D:/My_Projects/SBH/test"  # Update with the correct test dataset path
# # load_and_test_model(model_folder_path, test_folder_path)