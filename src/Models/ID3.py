import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

def print_tree(self, tree: Optional[Dict[str, Any]] = None, indent: str = "") -> None:
        """Recursively prints the decision tree in a readable format"""
        if tree is None:
            tree = self.tree

        if isinstance(tree, dict):
            for feature, branches in tree.items():
                print(f"{indent}{feature}")
                for value, subtree in branches.items():
                    print(f"{indent}  --> {value}:")
                    self.print_tree(subtree, indent + "    ")
        else:
            print(f"{indent}--> Class: {tree}")
            
class ID3DecisionTree:
    def __init__(self):
        self.tree: Optional[Dict[str, Any]] = None
        self.label: str = "" 

    def calculate_entropy(self, data: pd.DataFrame) -> float:
        label_counts = data.iloc[:, -1].value_counts()
        probabilities = label_counts / len(data)
        entropy = -sum(probabilities * np.log2(probabilities + 1e-9))
        return entropy

    def calculate_information_gain(self, data: pd.DataFrame, feature: str) -> float:
        total_entropy = self.calculate_entropy(data)
        unique_values, counts = np.unique(data[feature], return_counts=True)
        weighted_entropy = sum(
            (count / len(data)) * self.calculate_entropy(data[data[feature] == value])
            for value, count in zip(unique_values, counts)
        )
        return total_entropy - weighted_entropy

    def find_best_splits(self, data: pd.DataFrame, feature: str, min_gain: float = 0.01) -> List[float]:
        sorted_data = data.sort_values(feature)
        unique_values = sorted_data[feature].unique()
        split_points = []

        def find_split(start: int, end: int):
            best_split, best_gain = None, 0.0
            for i in range(start + 1, end):
                midpoint = (unique_values[i - 1] + unique_values[i]) / 2
                left, right = sorted_data[sorted_data[feature] <= midpoint], sorted_data[sorted_data[feature] > midpoint]

                if len(left) > 0 and len(right) > 0:
                    gain = self.calculate_information_gain_on_split(data, left, right)
                    if gain > best_gain:
                        best_gain, best_split = gain, midpoint

            if best_split and best_gain >= min_gain:
                split_points.append(best_split)
                idx = np.searchsorted(unique_values, best_split)
                find_split(start, idx)
                find_split(idx, end)

        find_split(0, len(unique_values))
        return sorted(split_points)

    def calculate_information_gain_on_split(self, data: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> float:
        total_entropy = self.calculate_entropy(data)
        left_weight = len(left) / len(data)
        right_weight = len(right) / len(data)
        return total_entropy - (left_weight * self.calculate_entropy(left) + right_weight * self.calculate_entropy(right))

    def convert_numeric_to_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in data.columns[:-1]:
            if pd.api.types.is_numeric_dtype(data[column]):
                split_points = self.find_best_splits(data, column)
                if split_points:
                    bins = [-np.inf] + split_points + [np.inf]
                    labels = [f'<= {split_points[0]}'] + \
                             [f'{split_points[i]} - {split_points[i + 1]}' for i in range(len(split_points) - 1)] + \
                             [f'> {split_points[-1]}']
                    data[column] = pd.cut(data[column], bins=bins, labels=labels)
        return data

    def best_feature_to_split(self, data: pd.DataFrame) -> str:
        gains = {feature: self.calculate_information_gain(data, feature) for feature in data.columns[:-1]}
        return max(gains, key=gains.get)

    def build_tree(self, data: pd.DataFrame) -> Dict[str, Any]:
        target = data.columns[-1]
        if len(data[target].unique()) == 1:
            return data[target].iloc[0]
        if len(data.columns) == 1:
            return data[target].mode().iloc[0]

        best_feature = self.best_feature_to_split(data)
        tree = {best_feature: {}}

        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value].drop(columns=[best_feature])
            tree[best_feature][value] = self.build_tree(subset)

        return tree

    def train_model(self, data: pd.DataFrame, label: str) -> Dict[str, Any]:
        self.label = label  # Store label name for clarity
        processed_data = self.convert_numeric_to_categorical(data)
        self.tree = self.build_tree(processed_data)
        return self.tree

    def predict_sample(self, tree: Dict[str, Any], sample: pd.Series) -> Any:
        while isinstance(tree, dict):
            feature = next(iter(tree))
            tree = tree.get(feature, {}).get(sample[feature], None)
        return tree

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series([self.predict_sample(self.tree, row) for _, row in data.iterrows()])
    
    def load_model(self):
        pass
    def save_model(self):
        pass


# Test the model with a label
if __name__ == "__main__":
    df = pd.DataFrame({
        'Temperature': [85, 80, 83, 70, 68, 65, 72, 75, 77, 81, 71, 73, 78, 69],
        'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 60, 65, 75, 80, 70, 80],
        'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'True'],
        'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    })

    tree_model = ID3DecisionTree()
    tree = tree_model.train_model(df, label='Play')
