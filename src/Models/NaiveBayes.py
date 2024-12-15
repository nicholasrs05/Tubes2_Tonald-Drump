import pandas as pd
import numpy as np
import json
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from google.colab import files


class CustomGaussianNB(BaseEstimator, ClassifierMixin):

    def __init__(self):

        self.smoothing = 1e-9
        self.label_encoder = LabelEncoder()
        self.model = {}
        self.classes = None
        self.class_priors = None
        self.fitted = False

    def fit(self, X, y):

        # if self.fitted:
        #     return self

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        y_encoded = self.label_encoder.fit_transform(y)
        self.classes = self.label_encoder.classes_
        class_counts = np.bincount(y_encoded)
        self.class_priors = np.log(class_counts / len(y_encoded))

        self.model = {}
        for label in np.unique(y_encoded):
            class_data = X[y_encoded == label]
            self.model[label] = {}

            for i in range(X.shape[1]):
                feature_data = class_data[:, i]
                mean = np.mean(feature_data)
                std = np.std(feature_data)

                std = max(std, self.smoothing)
                self.model[int(label)][int(i)] = {'mean': mean, 'std': std}

        self.fitted = True
        return self

    def predict(self, X):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        predictions = []

        for row in X:
            log_likelihoods = {}

            predicted_label_encoded = None
            max_log_likelihood = float('-inf')


            for label in self.model.keys():
                log_likelihood = self.class_priors[label]

                for i in range(len(row)):
                    mean = self.model[label][i]['mean']
                    std = self.model[label][i]['std']

                    coefficient = -np.log(std * np.sqrt(2 * np.pi))
                    exponent = -0.5 * ((row[i] - mean) / std) ** 2

                    log_likelihood += coefficient + exponent

                if predicted_label_encoded == None or log_likelihood > max_log_likelihood:
                    max_log_likelihood = log_likelihood
                    predicted_label_encoded = label

                log_likelihoods[label] = log_likelihood

            predicted_class = self.label_encoder.inverse_transform([predicted_label_encoded])[0]
            predictions.append(predicted_class)

        return np.array(predictions)

    def save_model(self, model_path, download=False):

        if not self.fitted:
            raise ValueError("Model is not fitted. Please fit the model before saving.")

        model_data = {
            'class_priors': self.class_priors.tolist(),
            'classes': self.classes.tolist(),
            'model': {},
        }

        for label in self.model:
            model_data['model'][str(label)] = {}
            for i in self.model[label]:
                model_data['model'][str(label)][str(i)] = {
                    'mean': str(self.model[label][i]['mean']),
                    'std': str(self.model[label][i]['std'])
                }

        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=4)

        print(f"\nModel saved to: {model_path}")

        if download:
            files.download(model_path)


    def load_model(self, model_path):
        with open(model_path, 'r') as f:
            model_data = json.load(f)

        self.class_priors = np.array(model_data['class_priors'])
        self.classes = np.array(model_data['classes'])
        self.model = model_data['model']

        new_model = {}
        for label_str, features in self.model.items():
            label = int(label_str)
            new_model[label] = {}
            for i_str, feature_data in features.items():
                i = int(i_str)
                feature_data['mean'] = float(feature_data['mean'])
                feature_data['std'] = float(feature_data['std'])
                new_model[label][i] = feature_data

        self.model = new_model
        self.label_encoder.classes_ = np.array(np.array(model_data['classes']))

        self.fitted = True


# if __name__ == "__main__":
#     nb = NaiveBayes()
    
#     # # Train model
#     # train_data = pd.read_csv("data/preprocessed_train_data.csv")
#     # nb.trainModel(df=train_data)
#     # nb.saveModel("src/Models/NB_model.json")
    
#     # Test model
#     validation_data = pd.read_csv("data/preprocessed_validation_data.csv")
#     split_index = int(len(validation_data) * 0.1)
#     validation_data = validation_data.iloc[:split_index]
    
#     nb.loadModel("src/Models/NB_model.json")
#     accuracy = nb.calculateAccuracy(validation_data)
#     print(f"Model Accuracy: {accuracy * 100:.2f}%")


    
    