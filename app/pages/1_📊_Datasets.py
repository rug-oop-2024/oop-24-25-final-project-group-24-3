import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from pathlib import Path
import io

automl = AutoMLSystem.get_instance()

st.header("Available Datasets")
datasets = automl.registry.list(type="dataset")
if datasets:
    for dataset in datasets:
        st.subheader(f"{dataset.name} (Version: {dataset.version})")

        if f"show_{dataset.name}" not in st.session_state:
            st.session_state[f"show_{dataset.name}"] = False

        if st.button(f"Show/Hide Dataset: {dataset.name}"):
            st.session_state[
                f"show_{dataset.name}"] = not st.session_state[
                    f"show_{dataset.name}"]

        if st.session_state[f"show_{dataset.name}"]:
            data_bytes = dataset.read()
            data_str = data_bytes.decode()
            data_io = io.StringIO(data_str)
            data_df = pd.read_csv(data_io)

            st.write("Dataset Preview:")
            st.dataframe(data_df)
else:
    st.write("No datasets available.")

st.header("Upload and Save New Dataset")
uploaded_file = st.file_uploader("Choose a CSV file (Note: other "
                                 "file types are not allowed)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset Preview:")
    st.write(data)

    dataset_name = st.text_input("Enter a name for the dataset:")
    version = st.text_input("Version", value="1.0.0")

    if st.button("Save Dataset") and dataset_name:
        artifact_base_directory = Path("assets/dbo/artifacts")
        artifact_directory = artifact_base_directory / dataset_name / version
        artifact_directory.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_directory / f"{dataset_name}.csv"

        new_dataset = Dataset.from_dataframe(
            data=data,
            name=dataset_name,
            asset_path=str(artifact_path),
            version=version
        )

        automl.registry.register(new_dataset)
        st.success(f"Dataset '{dataset_name}' saved successfully.")
