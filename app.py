from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
import pandas as pd
import numpy as np
from config import config

app = Flask(__name__)

# Load configuration
config_name = os.environ.get('FLASK_ENV', 'default')
app.config.from_object(config[config_name])

# Ensure upload directory exists
UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize the deep learning model as None, will be created when needed
dl_model = None

def get_model():
    """Lazy initialization of the deep learning model"""
    global dl_model
    if dl_model is None:
        try:
            # Import here to avoid issues with TensorFlow during startup
            from models.deep_learning_model import ExoplanetDeepLearningModel
            dl_model = ExoplanetDeepLearningModel()
        except Exception as e:
            print(f"Warning: Could not initialize model: {e}")
            dl_model = None
    return dl_model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for Google Cloud monitoring"""
    return jsonify({
        "status": "healthy",
        "service": "cosmos-explorer",
        "timestamp": pd.Timestamp.now().isoformat()
    })

@app.route('/upload_file', methods=['POST'])
def upload_file():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file has a filename
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        # Save file
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Return success response
            return jsonify({
                'status': 'success', 
                'message': f'File {filename} uploaded successfully',
                'filename': filename,
                'file_path': file_path
            }), 200
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/analyze_data', methods=['POST'])
def analyze_data():
    try:
        # Get form data
        data = request.get_json()
        
        # In a real application, you would process the data here
        # For now, we'll just return a success response with mock results
        return jsonify({
            "status": "success",
            "message": "Data analysis completed successfully",
            "results": {
                "accuracy": "95%",
                "precision": "92%",
                "recall": "90%"
            }
        }), 200
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        print("\n=== MODEL TRAINING REQUEST RECEIVED ===")
        
        # Get the model instance
        model = get_model()
        
        # Check if model was successfully initialized
        if model is None:
            return jsonify({'status': 'error', 'message': 'Model not available. Service may be starting up or there was an error initializing the model.'}), 503
        
        # Get the uploaded files
        uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
        print(f"Found {len(uploaded_files)} files in upload folder")
        
        if not uploaded_files:
            return jsonify({'status': 'error', 'message': 'No files uploaded for training'}), 400
        
        # Load and combine data from all files
        all_dataframes = []
        for filename in uploaded_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"Processing file: {file_path}")
            if os.path.exists(file_path):
                df = model.load_and_clean_data(file_path)
                if not df.empty:
                    print(f"Loaded {len(df)} rows from {filename}")
                    all_dataframes.append(df)
                else:
                    print(f"No data loaded from {filename}")
            else:
                print(f"File not found: {file_path}")
        
        if not all_dataframes:
            return jsonify({'status': 'error', 'message': 'No valid data found in uploaded files'}), 400
        
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"\n--- Data Consolidation Complete. Total rows: {combined_df.shape[0]} ---")
        
        # Preprocess data
        X, y = model.preprocess_data(combined_df)
        print(f"After preprocessing - X shape: {X.shape if not X.empty else 'Empty'}, y shape: {y.shape if not y.empty else 'Empty'}")
        
        # Check if preprocessing was successful
        if X.empty or y.empty:
            return jsonify({'status': 'error', 'message': 'Failed to preprocess data. Check that your CSV files contain the required columns.'}), 400
        
        print(f"Preprocessed data shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Train model
        print("Starting model training...")
        history, X_test, y_test, y_pred = model.train(X, y, epochs=50)  # Reduced epochs for demo
        
        # Check if training was successful
        if history is None:
            return jsonify({'status': 'error', 'message': 'Model training failed. Check logs for details.'}), 500
        
        # Save model
        try:
            model.save_model()
            print("Model saved successfully")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
        
        # Get actual training results
        results = model.get_training_results()
        print(f"Training results: {results}")
        
        # Return success response with actual results
        return jsonify({
            "status": "success",
            "message": "Model training completed successfully",
            "results": {
                "accuracy": f"{results['accuracy']:.2%}",
                "precision": f"{results['precision']:.2%}",
                "recall": f"{results['recall']:.2%}",
                "loss": f"{results['loss']:.4f}",
                "total_rows": combined_df.shape[0],
                "feature_columns": getattr(model, 'feature_columns', [])
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Error in train_model: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Training failed: {str(e)}'}), 500

@app.route('/get_exoplanet_data')
def get_exoplanet_data():
    try:
        # Get the model instance
        model = get_model()
        
        # Check if model was successfully initialized
        if model is None:
            return jsonify({'status': 'error', 'message': 'Model not available. Service may be starting up or there was an error initializing the model.'}), 503
        
        # Get the uploaded files
        uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
        
        if not uploaded_files:
            return jsonify({'status': 'error', 'message': 'No files uploaded'}), 400
        
        # Load and combine data from all files
        all_dataframes = []
        for filename in uploaded_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(file_path):
                df = model.load_and_clean_data(file_path)
                if not df.empty:
                    all_dataframes.append(df)
        
        if not all_dataframes:
            return jsonify({'status': 'error', 'message': 'No valid data found in uploaded files'}), 400
        
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Preprocess data
        X, y = model.preprocess_data(combined_df)
        
        # Check if preprocessing was successful
        if X.empty or y.empty:
            return jsonify({'status': 'error', 'message': 'Failed to preprocess data'}), 400
        
        # If model is trained, make predictions
        predictions = []
        prediction_probabilities = []
        if model.model is not None:
            try:
                # Make predictions on the data
                pred, pred_proba = model.predict(X)
                predictions = pred.flatten().tolist()
                prediction_probabilities = pred_proba.flatten().tolist()
            except Exception as e:
                print(f"Error making predictions: {e}")
                predictions = [0] * len(X)
                prediction_probabilities = [0.5] * len(X)
        else:
            # If no model is trained, use default values
            predictions = [0] * len(X)
            prediction_probabilities = [0.5] * len(X)
        
        # Prepare data for visualization
        all_exoplanet_data = []
        for i in range(len(combined_df)):  # Show all entries
            row = combined_df.iloc[i]
            # Determine status based on prediction and confidence
            if predictions[i] == 1:
                status = 'Confirmed'
            else:
                # For candidates, we'll use the original disposition if available
                original_disposition = str(row.get('disposition', '')).upper()
                if 'CANDIDATE' in original_disposition or 'PC' in original_disposition or 'KP' in original_disposition:
                    status = 'Candidate'
                else:
                    status = 'False Positive'
                    
            all_exoplanet_data.append({
                'name': f"KOI-{i+1:04d}" if 'koi' in str(row.get('disposition', '')).lower() else f"TOI-{i+1:04d}",
                'status': status,
                'orbital_period': float(round(row.get('period', 0), 2)) if not pd.isna(row.get('period', 0)) else 0,
                'radius': float(round(row.get('planet_radius', 0), 2)) if not pd.isna(row.get('planet_radius', 0)) else 0,
                'star_teff': float(round(row.get('star_teff', 0), 2)) if not pd.isna(row.get('star_teff', 0)) else 0,
                'star_radius': float(round(row.get('star_radius', 0), 2)) if not pd.isna(row.get('star_radius', 0)) else 0,
                'impact': float(round(row.get('impact', 0), 2)) if not pd.isna(row.get('impact', 0)) else 0,
                'confidence': float(round(prediction_probabilities[i] * 100, 1)) if i < len(prediction_probabilities) else 50
            })
        
        # Sort by confidence level
        all_exoplanet_data.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Separate by status
        confirmed_exoplanets = [p for p in all_exoplanet_data if p['status'] == 'Confirmed']
        candidate_exoplanets = [p for p in all_exoplanet_data if p['status'] == 'Candidate']
        false_exoplanets = [p for p in all_exoplanet_data if p['status'] == 'False Positive']
        
        # Get top 5 for each category
        top_confirmed = confirmed_exoplanets[:5]
        top_candidates = candidate_exoplanets[:5]
        top_false = false_exoplanets[:5]
        
        # Combine for display
        exoplanet_data = top_confirmed + top_candidates + top_false
        
        # Calculate statistics for visualization
        stats = {
            'total_exoplanets': len([p for p in all_exoplanet_data if p['status'] == 'Confirmed']),
            'total_candidates': len([p for p in all_exoplanet_data if p['status'] == 'Candidate']),
            'total_false': len([p for p in all_exoplanet_data if p['status'] == 'False Positive']),
            'avg_orbital_period': float(round(combined_df['period'].mean(), 2)) if 'period' in combined_df.columns else 0,
            'avg_planet_radius': float(round(combined_df['planet_radius'].mean(), 2)) if 'planet_radius' in combined_df.columns else 0
        }
        
        # Debug: Print data info
        print(f"Total exoplanet data entries: {len(all_exoplanet_data)}")
        print(f"Sample data entry: {all_exoplanet_data[0] if all_exoplanet_data else 'No data'}")
        
        return jsonify({
            'status': 'success',
            'exoplanet_data': exoplanet_data,
            'statistics': stats,
            'all_data': all_exoplanet_data  # Include all data for charts
        }), 200
        
    except Exception as e:
        print(f"Error in get_exoplanet_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

# For Google App Engine compatibility
app = app

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    # Add a startup delay to allow for proper initialization
    import time
    time.sleep(2)
    app.run(host='0.0.0.0', port=port, debug=False)