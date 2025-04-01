import pandas as pd
import nltk
import string
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download("stopwords")

# Load the dataset
df = pd.read_csv("dataset/email.csv", encoding="latin-1")

# Check for missing values
print("Missing values before cleaning:\n", df.isnull().sum())

# Drop NaN values
df.dropna(inplace=True)

# Verify column names (Adjust based on CSV structure)
print(df.head())

# Rename columns if needed
df.columns = ["label", "message"]

# Convert labels to numerical values (Check unique values)
print("Unique labels:", df["label"].unique())

df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Remove any remaining NaN values after mapping
df.dropna(inplace=True)

# Data preprocessing function
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(words)

# Apply preprocessing
df["message"] = df["message"].apply(preprocess_text)

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# Verify if any NaN remains
print("Missing values after cleaning:", df.isnull().sum())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model and vectorizer
with open("model/spam_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved successfully!")