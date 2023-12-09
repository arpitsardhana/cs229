from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class CustomEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier1=None, classifier2=None):
        self.classifier1 = classifier1
        self.classifier2 = classifier2
        self.ensemble_classifier = VotingClassifier(estimators=[('clf1', classifier1), ('clf2', classifier2)], voting='soft')

    def fit(self, X, y):
        # Train both classifiers on the input data
        self.classifier1.fit(X, y)
        self.classifier2.fit(X, y)

        # Train the ensemble classifier on the input data
        self.ensemble_classifier.fit(X, y)
        return self

    def predict(self, X):
        # Predict using both classifiers
        predictions1 = self.classifier1.predict(X)
        predictions2 = self.classifier2.predict(X)

        # Combine the predictions using soft voting
        combined_predictions = self.ensemble_classifier.predict_proba(X)

        return combined_predictions

# Example usage
# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate data by combining 11 and 00 by matching age and sex.
# at time of training extract required params, train separately
# at time of prediction fit separately and combine

# Create two classifiers
classifier1 = DecisionTreeClassifier(random_state=42)
classifier2 = SVC(probability=True, random_state=42)

# Create the custom ensemble classifier
custom_classifier = CustomEnsembleClassifier(classifier1=classifier1, classifier2=classifier2)

# Train the custom classifier
custom_classifier.fit(X_train, y_train)

# Make predictions
y_pred_proba = custom_classifier.predict(X_test)

# Use argmax to get the predicted classes
y_pred = y_pred_proba.argmax(axis=1)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

