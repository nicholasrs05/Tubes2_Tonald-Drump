from sklearn.model_selection import train_test_split
import pandas as pd
from pipelines import create_preprocessing_pipeline, handleMissingValue

def main(data_path, train_output_path, validation_output_path):

    df = pd.read_csv(data_path)
    df = df.rename(columns={"attack_cat ": "attack_cat"})

    metadata = pd.read_csv("data/metadata.csv")
    metadata = metadata.rename(columns={"Type ": "Type"})

    metadata["Type"] = metadata["Type"].str.lower()
    metadata[["Name", "Type"]]

    # Drop column with name = label in metadata
    if "label" in metadata["Name"].values:
        metadata = metadata[metadata["Name"] != "label"]

    numeric_columns = metadata.loc[
        (metadata["Type"] == "float") | (metadata["Type"] == "integer")
    ]["Name"]
    categorical_columns = metadata.loc[
        (metadata["Type"] != "float") & (metadata["Type"] != "integer")
    ]["Name"]


    df = df.drop(columns=["label"])
    
    preprocessing_pipeline = create_preprocessing_pipeline(
        df, numeric_columns, categorical_columns
    )
    
    # train_set, val_set = train_test_split(
    # df, test_size=0.33, random_state=42, stratify=df["attack_cat"]
    # )
    # preprocessing_pipeline = create_preprocessing_pipeline(
    #     train_set, numeric_columns, categorical_columns
    # )

    # Save the preprocessing pipeline as csv
    split_index = int(len(preprocessing_pipeline) * 0.8)
    
    train_set = preprocessing_pipeline.iloc[:split_index]
    valiadation_set = preprocessing_pipeline.iloc[split_index:]
    
    print("Saving preprocessed train set data to csv")
    pd.DataFrame(train_set).to_csv(
        train_output_path, index=False
    )
    
    print("Saving preprocessed validation set data to csv")
    pd.DataFrame(valiadation_set).to_csv(
        validation_output_path, index=False
    )

if __name__ == "__main__":
    
    data_path = "data/merged_train_data.csv"
    train_output_path = "data/preprocessed_train_data.csv"
    validation_output_path = "data/preprocessed_validation_data.csv"
    
    main(data_path, train_output_path, validation_output_path)