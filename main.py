from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the SVM model from the file
with open('content/svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer from the file
with open('content/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def index():
    return render_template('index.html', emotion=None)

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    
    # Transform input text using the loaded vectorizer to create feature array for prediction
    processed_text = vectorizer.transform([input_text])  # Transform input text to feature array
    
    # Make prediction using the loaded model
    predicted_emotion = model.predict(processed_text)

    return render_template('index.html', emotion=predicted_emotion[0])

if __name__ == '__main__':
    app.run(debug=True)