import streamlit as st
import pandas as pd
import io
from app.core.system import AutoMLSystem

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(
        f"<p style=\"color: #888;\">{text}</p>",
        unsafe_allow_html=True
    )


def display_pipeline_summary():
    dataset = st.session_state.get('dataset', 'Not Set')
    task_type = st.session_state.get('task_type', 'Not Set')
    model = st.session_state.get('selected_model', 'Not Set')

    st.write(f"Session state dataset: {dataset}")
    st.write(f"Session state task_type: {task_type}")
    st.write(f"Session state model: {model}")

    if 'solver' not in st.session_state:
        st.session_state.solver = "liblinear"
    if 'n_estimators' not in st.session_state:
        st.session_state.n_estimators = 100
    if 'max_depth' not in st.session_state:
        st.session_state.max_depth = None
    if 'kernel' not in st.session_state:
        st.session_state.kernel = "rbf"

    keys = ['dataset', 'task_type', 'selected_model']
    if all(key in st.session_state for key in keys):
        summary = f"""
        ## 📊 Pipeline Summary

        **Dataset:** {st.session_state.dataset['name']}
        (Version: {st.session_state.dataset['version']})

        **Task Type:** {st.session_state.task_type}

        ### Selected Features:
        - Input Features: {', '.join(st.session_state.input_features)}
        - Target Feature: {st.session_state.target_feature}

        ### Data Split:
        - Train set size: {st.session_state.X_train.shape[0]} samples
        - Test set size: {st.session_state.X_test.shape[0]} samples
        - Train-Test Split Ratio: {st.session_state.split_ratio}

        ### Model:
        **{st.session_state.selected_model}**

        #### Hyperparameters:
        """

        if st.session_state.selected_model in [
            "Logistic Regression", "Linear Regression"
        ]:
            summary += f" - Solver: {st.session_state.solver}\n"
        elif st.session_state.selected_model in [
            "Random Forest", "Random Forest Regressor"
        ]:
            summary += (
                f" - Number of Estimators: {st.session_state.n_estimators}\n"
                )
            summary += f" - Max Depth: {st.session_state.max_depth}\n"
        elif st.session_state.selected_model in ["SVM", "SVR"]:
            summary += f" - Kernel: {st.session_state.kernel}\n"

        summary += f"""
        ### Evaluation Metrics:
        - {', '.join(st.session_state.selected_metrics)}
        """

        st.markdown(summary, unsafe_allow_html=True)
    else:
        st.warning("Please complete the steps before viewing the summary.")


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline"
    " to train a model on a dataset."
)

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

if 'data_split' not in st.session_state:
    st.session_state.data_split = False
if 'model_selected' not in st.session_state:
    st.session_state.model_selected = False
if 'metrics_selected' not in st.session_state:
    st.session_state.metrics_selected = False

if datasets:
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_names)

    selected_dataset = next(
        dataset for dataset in datasets
        if dataset.name == selected_dataset_name
    )

    st.write(
        f"### Dataset: {selected_dataset.name} "
        f"(Version: {selected_dataset.version})"
    )

    data_bytes = selected_dataset.read()

    data_str = data_bytes.decode()
    data_io = io.StringIO(data_str)
    data_df = pd.read_csv(data_io)

    st.write("Dataset Preview:")
    st.dataframe(data_df)

    features = data_df.columns.tolist()

    input_features = st.multiselect("Select input features (X)", features)
    target_feature = st.selectbox("Select target feature (Y)", features)

    if not input_features:
        st.warning("Please select at least one input feature.")
    if not target_feature:
        st.warning("Please select a target feature.")

    if input_features and target_feature:
        if (
            data_df[target_feature].dtype == "object"
            or len(data_df[target_feature].unique()) < 20
        ):
            task_type = "Classification"
        else:
            task_type = "Regression"

        st.write(f"### Detected Task Type: {task_type}")

        split_ratio = st.slider("Select train-test split ratio", 0.1, 0.9, 0.8)
        st.write(
            f"Training set size: {split_ratio*100:.2f}%, "
            f"Test set size: {(1-split_ratio)*100:.2f}%"
        )

        if not st.session_state.data_split:
            if st.button("Split Data"):
                shuffled_data = data_df.sample(
                    frac=1, random_state=42
                ).reset_index(drop=True)
                split_index = int(split_ratio * len(shuffled_data))
                train_data = shuffled_data[:split_index]
                test_data = shuffled_data[split_index:]

                X_train = train_data[input_features]
                y_train = train_data[target_feature]
                X_test = test_data[input_features]
                y_test = test_data[target_feature]

                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test

                st.session_state.data_split = True
                st.session_state.split_ratio = split_ratio

                st.write(
                    f"Train set size: {X_train.shape[0]}, "
                    f"Test set size: {X_test.shape[0]}"
                )

        if st.session_state.data_split:
            if not st.session_state.model_selected:
                if st.button("Proceed to Model Selection"):
                    if task_type == "Classification":
                        st.write(
                            "You selected a classification task. "
                            "Choose a classification model:"
                        )
                        model_type = st.selectbox(
                            "Select a model",
                            ["Logistic Regression", "Random Forest", "SVM"]
                        )
                    else:
                        st.write(
                            "You selected a regression task. "
                            "Choose a regression model:"
                        )
                        model_type = st.selectbox(
                            "Select a model",
                            ["Linear Regression",
                             "Random Forest Regressor", "SVR"]
                        )

                    st.session_state.model_selected = True
                    st.session_state.selected_model = model_type
                    st.session_state.task_type = task_type
                    st.session_state.input_features = input_features
                    st.session_state.target_feature = target_feature
                    st.session_state.dataset = {
                        "name": selected_dataset.name,
                        "version": selected_dataset.version
                    }
                    st.write(f"Selected model: {model_type}")

            else:
                st.write(
                    f"### Model Selected: {st.session_state.selected_model}")

                if task_type == "Classification":
                    st.write("Select evaluation metrics for classification:")
                    metrics = st.multiselect(
                        "Select metrics",
                        ["Accuracy", "Precision", "Recall", "F1-Score",
                         "AUC", "Confusion Matrix"]
                    )
                else:
                    st.write("Select evaluation metrics for regression:")
                    metrics = st.multiselect(
                        "Select metrics",
                        ["Mean Absolute Error", "Mean Squared Error",
                         "R-squared", "Root Mean Squared Error"]
                    )

                if metrics:
                    st.session_state.metrics_selected = True
                    st.session_state.selected_metrics = metrics
                    st.write(f"Selected metrics: {', '.join(metrics)}")

        if (
            st.session_state.model_selected
            and st.session_state.metrics_selected
        ):
            if st.button("Show Pipeline Summary"):
                st.empty()
                display_pipeline_summary()
else:
    st.write("No datasets available.")
