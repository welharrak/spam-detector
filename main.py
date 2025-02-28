import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
# Load dataset
df = pd.read_csv("spam_ham_dataset.csv")

# Tokenize each mail text into single words for easier handling
df["text"] = df.apply(lambda row: row["text"].split(" "), axis=1)#

# Remove stopwords and special charachters from text
nltk.download('stopwords')
stop_words = stopwords.words("english")
df["text"] = df["text"].apply(lambda words: [word for word in words if word not in stop_words and word.isalpha()])
print(df.head())

# Split to train- , test-set
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)