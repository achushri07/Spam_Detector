import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from gensim.utils import simple_preprocess

# Initialize the Flask application
app = Flask(__name__)

# Load the trained models
try:
    word2vec_model = joblib.load('word2vec_model.pkl')
    classifier = joblib.load('classifier_model.pkl')
except FileNotFoundError:
    print("Error: Model files not found. Please run the training script first.")
    # You might want to handle this more gracefully in a real application
    exit()

def avg_word2vec(doc, model):
    """Calculates the average word vector for a document."""
    # Filter out words that are not in the vocabulary
    valid_words = [model.wv[word] for word in doc if word in model.wv.index_to_key]
    if not valid_words:
        # Return a zero vector if no valid words are found
        return np.zeros(model.vector_size)
    return np.mean(valid_words, axis=0)

@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if request.method == 'POST':
        try:
            # Get the message from the POST request
            message = request.json['message']

            # Preprocess the input message
            processed_input = simple_preprocess(message)
            X_input = np.array(avg_word2vec(processed_input, word2vec_model)).reshape(1, -1)

            # Make a prediction
            prediction = classifier.predict(X_input)
            prediction_proba = classifier.predict_proba(X_input)

            # Get the probability for the predicted class
            spam_probability = prediction_proba[0][1] if prediction[0] == 'spam' else prediction_proba[0][0]


            # Return the prediction as a JSON response
            return jsonify({
                'prediction': prediction[0],
                'spam_probability': f"{spam_probability:.2%}"
            })
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)