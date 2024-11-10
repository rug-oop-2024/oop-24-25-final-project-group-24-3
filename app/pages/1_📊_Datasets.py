import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

st.set_page_config(
    page_title="Manage Datasets",
    page_icon="📊",
    )

st.title("Dataset Management")

datasets = automl.registry.list(type="dataset")

if datasets:
    dataset_info = []

    for dataset in datasets:
        if isinstance(dataset, Dataset):
            data = dataset.data
            columns = dataset.columns
        else:
            data = getattr(dataset, 'data', None)
            columns = getattr(dataset, 'columns', None)

        dataset_info.append({
            "Name": dataset.name,
            "Type": dataset.type,
            "Size (rows)":
                len(dataset.data) if data is not None else "Unknown",
            "Columns": ", ".join(columns) if columns is not None else "Unknown"
        })

    df = pd.DataFrame(dataset_info)
    st.dataframe(df)

choose_dataset = st.selectbox("Select a dataset to see the details",
                              df['Name'])

if choose_dataset:
    st.subheader(f"Details for {choose_dataset}")
    dataset_object = next(
        (dataset for dataset in datasets if dataset.name == choose_dataset),
        None)

    if dataset_object:
        st.write("Columns:")
        st.write(dataset_object.columns)
        st.write("Preview")
        st.write(pd.DataFrame(dataset_object.data).head())

if hasattr(dataset_object, 'data'):
    csv = pd.DataFrame(dataset_object.data).to_csv(index=False)
    st.download_button(
        label="Download dataset",
        data=csv,
        file_name=f"{choose_dataset}.csv",
        mime='text/csv',
    )
