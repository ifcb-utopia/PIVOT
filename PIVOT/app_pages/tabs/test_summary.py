"""
Code that executes the contents of the Summary Metrics: Test Summary tab
and is called by the main dashboard.py script. This file formats data with the
aid of the dashboard_utils.py file.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import sql_utils
from utils import app_utils

from utils import dashboard_utils as ds
from utils import load_config

def main():
    """
    Executes the Streamlit formatted HTML displayed on the Test Summary tab and
    displays summary statistics and graphs for the test model. Users can select from
    the different test models stored on the SQL Database.
    """
    # Ensure the config file is not empty
    config_dict = load_config()
    if None in config_dict.values():
        st.error("""No database configuration found. Please update the database
                 configuration in the Settings page.""")
    else:
        # Check to see if models exist in the database
        model_list = app_utils.get_models()
        if not model_list:
            st.error("""Please ensure database configuration information is correct
                        and update on the Settings page.""")
        else:          
            two_columns = st.columns(2)
            with two_columns[0]:
                model_list = app_utils.get_models()
                model_dictionary = {}
                model_name = []

                for i in range(1,len(model_list)):
                    model_name.append(model_list[i]['m_id'])
                    model_dictionary[model_list[i]['m_id']] = model_list[i]['model_name']

                selected_model_sum = st.selectbox(
                    label='Select the model you wish to evaluate:',
                    options=tuple(model_name),
                    format_func=model_dictionary.__getitem__,
                    index=None)

            st.markdown("""<h1></h1>""", unsafe_allow_html=True)

            # Check that there are validated labels to be analyzed
            if selected_model_sum:
                validated_df = sql_utils.get_test_set_df(selected_model_sum)
            else:
                validated_df = None

            if validated_df is not None:

                with st.container(border=True):

                    st.markdown("""<h2 style='text-align: left; color: black;'>
                            Test Summary Dashboard</h2>""",
                            unsafe_allow_html=True)
                    
                    st.write("""This interactive dashboard shows the summary performance
                            metrics of the CNN on an unseen dataset. Filter the different
                            graphs by selecting items on the legends, click and drag over
                            graphs to zoom, and download graphs as PNGs by hovering over
                            graphs and selecting the camera icon.""")
                    
                    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

                    class_labels = ["Chloro",
                                        "Ciliate",
                                        "Crypto",
                                        "Diatom",
                                        "Dictyo",
                                        "Dinoflagellate",
                                        "Eugleno",
                                        "Other",
                                        "Prymnesio",
                                        "Null"]

                    # Load the train model summary data to compare
                    model_preds = pd.read_csv('data/model-summary-cnn-v1-b3.csv')
                    pred_label_counts = model_preds.groupby('pred_label').size().reset_index(name='count')['count'].values

                    with st.expander("View Labeling Progress:"):
                        # Display graphs showing the number and percent of images labeled by users
                        val_label_counts = validated_df.groupby('PRED_LABEL').size().reset_index(name='count')
                        val_label_counts = [val_label_counts[val_label_counts.PRED_LABEL == label]['count'].values[0] if len(val_label_counts[val_label_counts.PRED_LABEL == label]) > 0 else 0 for label in class_labels]

                        percent_df = pd.DataFrame({'class': class_labels,
                                               'total': [100 for i in range(10)],
                                               '% Images Labeled': (val_label_counts/pred_label_counts)*100})

                        count_df = pd.DataFrame({'class': class_labels,
                                                 '# Images Labeled': val_label_counts})

                        three_columns = st.columns([5, .2, 5])

                        with three_columns[0]:
                            target_options = [50, 100, 1000, 10000]
                            target = st.selectbox("Select the target number of images labeled per class:", options=target_options, index=0)
                            fig = ds.target_plot(count_df, target)
                            st.plotly_chart(fig, use_container_width=True)

                        with three_columns[2]:
                            st.markdown('#')
                            st.markdown('###')
                            st.markdown('###')
                            custom_colors = ['#0B5699', '#EDF8E6']
                            fig = ds.class_proportion_plot(percent_df)
                            st.plotly_chart(fig, use_container_width=True)

                    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

                    validated_df['IS_CORRECT'] = (validated_df['PRED_LABEL'] == \
                                                validated_df['CONSENSUS']).astype(int)
                    
                    # Get model summary statistics for both train and test subsets
                    val_stats = ds.get_acc_prec_recall(validated_df,
                                                    ['IS_CORRECT',
                                                        'CONSENSUS',
                                                        'PRED_LABEL'])

                    model_stats = ds.get_acc_prec_recall(
                                    pd.read_csv('data/model-summary-cnn-v1-b3.csv'),
                                    ['is_correct',
                                    'true_label',
                                    'pred_label'])

                    # Display the summary metrics
                    three_columns = st.columns([.75,2.5,2.5])
                    with three_columns[0]:
                        st.subheader("Metrics")
                        st.metric("Accuracy:",
                                    f"{val_stats[0]*100:.2f} %",
                                    delta=f"{(val_stats[0] - model_stats[0])*100:.2f} %")
                        st.metric("Precision:",
                                    f"{val_stats[1]*100:.2f} %",
                                    delta=f"{(val_stats[1] - model_stats[1])*100:.2f} %")
                        st.metric("Recall:",
                                    f"{val_stats[2]*100:.2f} %",
                                    delta=f"{(val_stats[2] - model_stats[2])*100:.2f} %")
                        st.metric("Images Validated:", f"{len(validated_df)}")

                    with three_columns[1]:
                        # Display the confusion matrix
                        st.subheader("Confusion Matrix", 
                         help="""A confusion matrix is a tabular representation that
                         summarizes the effectiveness of a machine learning model
                         when tested against a dataset. It provides a visual breakdown
                         of correct and incorrect predictions made by the model.""")
                        c_report_test = ds.get_classification_report(
                                        validated_df,
                                        ['CONSENSUS', 'PRED_LABEL', None])
                        st.plotly_chart(ds.plot_confusion_matrix(validated_df,
                                            ['CONSENSUS', 'PRED_LABEL'],
                                            classes=c_report_test.index,
                                            normalize=True), use_container_width=True)
                    with three_columns[2]:
                        # Display ROC Curve
                        st.subheader("ROC Curve",
                         help="""An ROC (Receiver Operating Characteristic) curve,
                         illustrates how well a classification model performs across
                         various classification thresholds. It showcases two key
                         parameters: True Positive Rate and False Positive Rate.
                         The curve plots the relationship between TPR and FPR as the
                         classification threshold changes. Lowering the threshold
                         identifies more items as positive, leading to an increase in
                         both False Positives and True Positives.""")
                        st.plotly_chart(ds.plot_roc_curve(validated_df['CONSENSUS'],
                                        pd.DataFrame(validated_df['PROBS'].tolist()), 
                                        c_report_test.index.sort_values(ascending=True)),
                                        use_container_width=True)
                    two_columns = st.columns([4,3])
                    with two_columns[0]:
                        # Display precision, recall, and f1 score plot
                        st.subheader("Model Performance: Precision, Recall, F1 Score", 
                         help="""Precision is the actual correct prediction divided by total
                        prediction made by model. Recall is the number of true positives
                        divided by the total number of true positives and false
                        negatives. F1 Score is the weighted average of precision and
                        recall.""")
                        c_report_test = c_report_test.sort_values(by=['f1-score'],
                                                                        ascending=False)
                        st.plotly_chart(ds.plot_precision_recall_f1(c_report_test),
                                    use_container_width=True)
                    with two_columns[1]:
                        # display sunburst plot unique to test summary data
                        st.subheader("Sunburst Plot", 
                         help="""This sunburst plot visualizes the CNN predicted labels
                         (inner circle) and the user verified labels (outer circle). """)                   
                        agg_df = validated_df.groupby(['PRED_LABEL', 'CONSENSUS']).size().reset_index(name='count')
                        fig = ds.plot_sunburst(agg_df)
                        st.plotly_chart(fig, use_container_width=True)
