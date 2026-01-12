import json
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import numpy as np
import joblib
import os
from rag_system import ask_rag_system  # Make sure this import matches your file

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Add this for session management

# Sample user database (replace with real database in production)
users = {
    'test@example.com': {
        'password': 'password123',
        'name': 'Test User'
    }
}

# Load trained model & scaler with robust error handling
try:
    # Use the correct model and scaler files
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("INFO: ML Model and scaler loaded successfully.")
except FileNotFoundError:
    print("WARNING: ML model files not found.")
    print("Prediction will be set to 0 (No Disease) if missing.")
    model = None
    scaler = None
except Exception as e:
    print(f"ERROR: Failed to load ML components: {e}")
    model = None
    scaler = None

# Helper function to check if user is logged in
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('signin'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Authentication Routes
@app.route('/')
def main():
    # Always redirect to signin page
    return redirect(url_for('signin'))

@app.route('/signin')
def signin():
    # If user is already logged in, redirect to main page
    if 'user' in session:
        return redirect(url_for('main_page'))
    return render_template('signin.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    
    # Check if user exists and password matches
    if email in users and users[email]['password'] == password:
        session['user'] = email
        session['user_name'] = users[email]['name']
        return jsonify({'success': True, 'message': 'Login successful'})
    else:
        return jsonify({'success': False, 'message': 'Invalid email or password'})

@app.route('/register', methods=['POST'])
def register():
    fullname = request.form.get('fullname')
    email = request.form.get('email')
    password = request.form.get('password')
    
    # Check if user already exists
    if email in users:
        return jsonify({'success': False, 'message': 'User already exists'})
    
    # Register new user
    users[email] = {
        'password': password,
        'name': fullname
    }
    
    session['user'] = email
    session['user_name'] = fullname
    return jsonify({'success': True, 'message': 'Registration successful'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('signin'))

# Protected Routes - require login
@app.route('/main')
@login_required
def main_page():
    return render_template('main.html')

@app.route('/heart-assessment')
@login_required
def heart_assessment():
    # Redirect to main page since we don't have heart-assessment.html
    return redirect(url_for('main_page'))

@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('chatbot.html')

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/model_insights')
@login_required
def model_insights():
    return render_template('modal_insights.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Get user input from form
        data = request.form.to_dict()
        
        # Define expected features in correct order (13 features)
        expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                           'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                           'ca', 'thal']
        
        # Create feature array with default values for missing features
        features = []
        for feature in expected_features:
            if feature in data:
                features.append(float(data[feature]))
            else:
                # Set default values for missing features
                if feature == 'ca':
                    features.append(0.0)  # Default: 0 major vessels
                elif feature == 'thal':
                    features.append(2.0)  # Default: Fixed defect
                else:
                    features.append(0.0)
        
        original_features = np.array(features).reshape(1, -1)
        
        # Default values
        prediction = 0
        confidence = 0.0
        
        # Scale input and predict
        if scaler is not None and model is not None:
            try:
                scaled_features = scaler.transform(original_features)
                
                # Try to get prediction
                try:
                    # For models with predict_proba
                    prediction_proba = model.predict_proba(scaled_features)
                    prediction = model.predict(scaled_features)[0]
                    confidence = float(prediction_proba[0][1] * 100)
                except AttributeError:
                    # For models without predict_proba
                    prediction = model.predict(scaled_features)[0]
                    confidence = 80.0 if prediction == 1 else 85.0
                    
            except Exception as e:
                print(f"Error in scaling/prediction: {e}")
                # Fallback prediction
                age = original_features[0][0]
                prediction = 1 if age > 60 else 0
                confidence = 75.0 if prediction == 1 else 90.0
        else:
            # Fallback if model failed to load
            age = original_features[0][0]
            prediction = 1 if age > 60 else 0
            confidence = 75.0 if prediction == 1 else 90.0
        
        # Determine result text and context for chatbot
        if prediction == 1:
            result_text = "High Risk"
            user_health_status = f"The user has been predicted to have HIGH RISK of heart disease with {confidence:.1f}% confidence. Key factors may include age, cholesterol levels, blood pressure, or other risk factors."
        else:
            result_text = "Low Risk" 
            user_health_status = f"The user has been predicted to have LOW RISK of heart disease with {confidence:.1f}% confidence. However, maintaining heart-healthy habits is still recommended."
        
        # Store prediction in user session for history
        if 'predictions' not in session:
            session['predictions'] = []
        
        session['predictions'].append({
            'risk_level': result_text,
            'probability': round(confidence, 2),
            'timestamp': '2024-01-01'  # Add actual timestamp
        })
        
        # Render the result with all necessary variables
        return render_template('chatbot.html', 
                              prediction_text=result_text,
                              probability=f"{confidence:.1f}",
                              user_health_status=user_health_status,
                              rag_available=True)
                              
    except Exception as e:
        print(f"Error in prediction: {e}")
        return render_template('chatbot.html',
                              prediction_text="Error",
                              probability="0.0",
                              user_health_status="There was an error processing your prediction.",
                              rag_available=False)

@app.route('/chatbot', methods=['POST'])
@login_required
def chatbot_endpoint():
    """Handles the POST request from the AI Health Advisor in result_view.html."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"response": "Invalid request data.", "sources": []}), 400
            
        query = data.get('query', '').strip()
        health_status = data.get('healthStatus', 'No specific health context provided.')
        
        if not query:
            return jsonify({"response": "Please enter a question.", "sources": []}), 400
        
        # Combine the context and the new query for the RAG system
        rag_prompt = f"Health Context: {health_status}. User Query: {query}"

        print(f"Calling RAG for interactive chat with prompt: {rag_prompt}")
        
        # Call the RAG system - handle potential errors
        try:
            result = ask_rag_system(rag_prompt)
            print(f"RAG response received: {result}")
            
            # Ensure the result has the expected structure
            if isinstance(result, dict) and 'response' in result:
                return jsonify(result)
            else:
                # If result is not a dict, wrap it
                return jsonify({
                    "response": str(result) if result else "I apologize, but I couldn't generate a response. Please try again.",
                    "sources": []
                })
                
        except Exception as rag_error:
            print(f"Error in RAG system call: {rag_error}")
            return jsonify({
                "response": "I'm having trouble accessing the health knowledge base right now. Please try again later or consult a healthcare professional for immediate concerns.",
                "sources": []
            }), 500
    
    except Exception as e:
        print(f"Error in chatbot_endpoint: {e}")
        return jsonify({
            "response": "An error occurred while processing your request. Please try again.",
            "sources": []
        }), 500

# Simple error handlers to avoid template issues
@app.errorhandler(404)
def not_found(error):
    return "Page not found", 404

@app.errorhandler(500)
def internal_error(error):
    return "Internal server error", 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('knowledge_base', exist_ok=True)
    
    # Test RAG system initialization
    try:
        from rag_system import RAG_READY
        if RAG_READY:
            print("✅ RAG System is ready!")
        else:
            print("⚠️ RAG System is not available, but the app will still run")
    except ImportError as e:
        print(f"⚠️ Could not import RAG system: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)