import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# make nn
class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=100, batch_size=10):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = Sequential([
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # Handling classes_
        y_encoded = np.array(y == self.classes_[1], dtype=int)  # Binary encoding
        self.model.fit(X, y_encoded, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(np.int32)

    def predict_proba(self, X):
        proba = self.model.predict(X)
        return np.hstack((1 - proba, proba))  # Returning probabilities for both classes

# read data
data = pd.read_csv("https://raw.githubusercontent.com/OfficialMarvin/PredictBrainTumor/main/TCGA_GBM_LGG_Mutations_all.csv")

X = data.drop('Grade', axis=1)
y = data['Grade']
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# make models
classifier1 = RandomForestClassifier(n_estimators=100, random_state=123)
classifier2 = LogisticRegression()
classifier3 = SVC(probability=True)
classifier4 = KerasClassifierWrapper(epochs=100, batch_size=10)

# stack em
stacked_model = StackingClassifier(
    estimators=[
        ('rf', classifier1),
        ('lr', classifier2),
        ('svm', classifier3),
        ('nn', classifier4)],
    final_estimator=LogisticRegression(),
    stack_method='predict_proba'
)

stacked_model.fit(X_train_scaled, y_train)

# predict on test
y_pred = stacked_model.predict(X_test_scaled)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
