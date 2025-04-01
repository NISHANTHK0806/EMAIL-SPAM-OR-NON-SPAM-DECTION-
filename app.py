from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load the trained model and vectorizer
model_path = "model/spam_classifier.pkl"
vectorizer_path = "model/vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
else:
    print("❌ Model files not found. Please train the model first.")
    exit()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form.get("message", "")
    
    if email_text.strip() == "":
        return render_template("result.html", prediction="No text provided.")

    # Transform input text
    input_data = vectorizer.transform([email_text])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    result = "Spam" if prediction == 1 else "Not Spam"
    
    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)