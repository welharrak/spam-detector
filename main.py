import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from models.logistic_regression import logistic_regression
from prediction import predict_log_reg
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv("./data/spam_ham_dataset.csv")

# Tokenize each mail text into single words for easier handling
df["text"] = df.apply(lambda row: row["text"].split(" "), axis=1)#

# Remove stopwords and special charachters from text
nltk.download('stopwords')
stop_words = stopwords.words("english")
df["text"] = df["text"].apply(lambda words: [word for word in words if word not in stop_words and word.isalpha()])

# Combine the words back into strings for vectorization afterwards
df["text"] = df["text"].apply(lambda words: ' '.join(words))

# Split to train- , test-set
X = df["text"]
y = df["label_num"]

# Convert text to numerical values using TF-IDF Vectorization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
vectorizer = TfidfVectorizer()
X_train_n = vectorizer.fit_transform(X_train)
X_test_n = vectorizer.transform(X_test)

weights, bias = logistic_regression(X_train_n.toarray(), y_train,lr=1)

# Now we predict and see the accurcy of model
print("Time for evaluation:\n")
prediction = predict_log_reg(X_test_n.toarray(), weights, bias)
accuracy = accuracy_score(y_test, prediction)
print(f"Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, prediction)
print("Confussion matrix:\n",cm)
