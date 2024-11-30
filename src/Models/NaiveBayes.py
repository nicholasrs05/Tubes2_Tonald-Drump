import pandas as pd
import numpy as np
import json
import time
from pprint import pprint

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def convert_to_np_float64(obj):

        if isinstance(obj, float):
            return np.float64(obj)
        elif isinstance(obj, dict):
            return {k: NumpyEncoder.convert_to_np_float64(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [NumpyEncoder.convert_to_np_float64(i) for i in obj]
        else:
            return obj

class NaiveBayes:
    def __init__(self):
        self.model = {}
        
    def trainModel(self, df):
        start_time = time.time()
        print(f"{(time.time()-start_time):.2f}s : Start training model with Gaussian Naive Bayes with {len(df)} samples")
        
        features_name = df.columns[:-1] 
        labels = df['attack_cat'].unique()  

        for i, feature in enumerate(features_name):
            print(f"{(time.time()-start_time):.2f}s : {i}. Training model for feature \"{feature}\"")
            self.model[feature] = {}

            for label in labels:
                label_data = df[df['attack_cat'] == label]

                mean = np.float64(label_data[feature].mean())
                stdev = np.float64(label_data[feature].std())
                
                self.model[feature][label] = {"mean": mean, "stdev": stdev}

        pprint(self.model)
        
    def predict(self, df):
        
        print(f"Start predicting with Gaussian Naive Bayes with {len(df)} samples")

        features_name = df.columns[:-1]
        labels = df['attack_cat'].unique()  
        
        predictions = []
        
        for i, row in df.iterrows():
            print(f"{i}/{len(df)}", end="\r")
            highest_prob = float('-inf')
            predicted_label = None
            
            for label in labels:
                prob = 1
                for feature in features_name:
                    prob *= self.calcGaussian(
                        row[feature], 
                        self.model[feature][label]["mean"], 
                        self.model[feature][label]["stdev"]
                    )
                
                if prob > highest_prob:
                    highest_prob = prob
                    predicted_label = label
            
            predictions.append(predicted_label)
        
        return pd.Series(predictions, index=df.index, name='predicted_label')
    
    def calculateAccuracy(self, df):
        predicted_labels = self.predict(df)
        
        accuracy = (predicted_labels == df['attack_cat']).mean()
        
        return accuracy

    def saveModel(self, path):
        with open(path, 'w') as f:
            json.dump(self.model, f, cls=NumpyEncoder, indent=4)
        print(f"Model saved to {path}")

    def loadModel(self, path):
        with open(path, 'r') as f:
            loaded_model = json.load(f)
            self.model = NumpyEncoder.convert_to_np_float64(loaded_model)
        print(f"Model loaded from {path}")

    def calcGaussian(self, data_to_predict, mean, stdev):
    
        if stdev == 0:
            stdev = 1e-10
        
        exponent = np.exp(-((data_to_predict - mean) ** 2) / (2 * stdev ** 2))
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

    def printModel(self):
        pprint(self.model)

if __name__ == "__main__":
    nb = NaiveBayes()
    
    # # Train model
    # train_data = pd.read_csv("data/preprocessed_train_data.csv")
    # nb.trainModel(df=train_data)
    # nb.saveModel("src/Models/NB_model.json")
    
    # Test model
    validation_data = pd.read_csv("data/preprocessed_validation_data.csv")
    split_index = int(len(validation_data) * 0.1)
    validation_data = validation_data.iloc[:split_index]
    
    nb.loadModel("src/Models/NB_model.json")
    accuracy = nb.calculateAccuracy(validation_data)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")


    
    