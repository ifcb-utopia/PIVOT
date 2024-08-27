"""
Code that executes the contents of the Image Validation page
and is called by the main app.py script.

Functions:
    - get_user_experience: Calculates the experience of a user based on the number
                           number of images they have labeled and their domain experience.
    - main: Executes the Streamlit formatted HTML when called by app.py.
"""
import streamlit as st
import pandas as pd
import numpy as np

from utils import app_utils
from utils import sql_utils
from utils import load_config

import logging

logging.basicConfig(level=logging.DEBUG)

config_dict = load_config()

def update_counter(increment, user_label):
    """
    Updates the counter for which image the user is annotating.

    Args:
        - increment (bool): Indicates if the counter should increment (True) or decrement (False)
        - user_label (str): The label the user is giving to the image.
    """
    logging.info("Next or back button clicked, updating counter with update_counter()")
    if increment and (st.session_state.counter == st.session_state.session_number - 1):
        pass
    elif not increment and (st.session_state.counter == 0):
        pass
    else:
        try:
            is_checked = st.session_state['plankton_check_' + str(st.session_state.counter)]
            if is_checked:
                st.session_state.new_df.loc[st.session_state.counter] = [
                                            st.session_state.label_df.iloc[st.session_state.counter]['IMAGE_ID'],
                                            st.session_state.user_account['u_id'],
                                            st.session_state.user_account['experience'],
                                            user_label]
                logging.info("New dataframe created to store user experience information")

            elif not is_checked and st.session_state.counter in st.session_state.new_df.index:
                st.session_state.new_df = st.session_state.new_df.drop(st.session_state.counter)
                logging.info("Information was not verified, moving on")
        except KeyError:
            st.exception("Clicking next too clickly")

        if increment:
            st.session_state.counter += 1
            logging.info("Next button clicked, increasing counter")
        else:
            st.session_state.counter -= 1
            logging.info("Back button clicked, decreasing counter")
    logging.info("End of update_counter()")

def submit_labels(user_label):
    """
    Checks if there are labels to be submitted and gives a toast if there are none to submit

    Args:
        - user_label (str): The label the user is giving to the image.
    """
    logging.info("Submit button clicked, enacting submit_labels()")
    is_checked = st.session_state['plankton_check_' + str(st.session_state.counter)]
    if is_checked:
        st.session_state.new_df.loc[st.session_state.counter] = [
            st.session_state.label_df.iloc[st.session_state.counter]['IMAGE_ID'],
            st.session_state.user_account['u_id'],
            st.session_state.user_account['experience'],
            user_label]
        logging.info("New dataframe created to store user, image, and label information after submit button was clicked")
    elif not is_checked and st.session_state.counter in st.session_state.new_df.index:
        st.session_state.new_df = st.session_state.new_df.drop(st.session_state.counter)
    if len(st.session_state.new_df) == 0:
        st.toast("No labels to submit")
        logging.info("No labels to submit")
    logging.info("End of submit_labels()")

def get_user_experience(num_labels, domain):

    """
    This function calculates the experience of a user from a range of 1 to 5, where
    1 indicates no experience and 5 indicates an expert.

    Args:
        - num_labels (str): User prompted range indicating the number of images labeled.
        - domain (str): User prompted response determine their domain.

    Returns:
        - exp_level (int): The level of experience ranging from 1 to 5.
    """
    logging.info("Getting user experience with get_user_experience()")
    exp_level = 1

    if domain == "No" and num_labels == "None":
        exp_level = 1
    elif domain == "No" and num_labels == "25 to 100":
        exp_level = 2
    elif domain == "No" and num_labels == "100 to 500":
        exp_level = 3
    elif domain == "No" and num_labels == "500 to 1000":
        exp_level = 4
    elif domain == "No" and num_labels == "1000+":
        exp_level = 5
    elif domain == "Yes" and num_labels == "None":
        exp_level = 2
    elif domain == "Yes" and num_labels == "25 to 100":
        exp_level = 3
    elif domain == "Yes" and num_labels == "100 to 500":
        exp_level = 4
    elif domain == "Yes" and num_labels == "500 to 1000":
        exp_level = 5
    elif domain == "Yes" and num_labels == "1000+":
        exp_level = 5

    logging.info(f"Experience level is {exp_level}, end of get_user_experience()")
    return exp_level

def get_label_prob_options(label_df, count):
    """
    Converts the probabilities of each category into a list and sorts them to
    be displayed in descending order.

    Args:
        - label_df (DataFrame): A Pandas DataFrame containing images to be labeled
        - count (int): Counter for the position within the DataFrame

    Returns:
        - exp_level (int): The level of experience ranging from 1 to 5.
    """
    logging.info("Sorting probabilities with get_label_prob_options()")
    # Convert dictionary of probabilities to a DataFrame
    label_probs = label_df.iloc[count]['PROBS']
    label_probs = pd.DataFrame.from_dict(label_probs, orient='index')

    # Rename the column title to PROBS
    column_name = label_probs.columns[0]
    label_probs = label_probs.rename(columns={column_name: "PROBS"})

    # Sort in descending order and convert into final list
    label_probs = label_probs.sort_values(by='PROBS', ascending=False)
    label_probs_options = label_probs.index.values.tolist()

    logging.info("Probabilities sorted and stored as a list. End of get_label_prob_options()")

    return label_probs_options

def display_label_info(label_df, count):
    """
    Display the phytoplankton image, the predicted label, and the image ID.

    Args:
        - label_df (DataFrame): A Pandas DataFrame containing images to be labeled
        - count (int): Counter for the position within the DataFrame

    Returns:
        - exp_level (int): The level of experience ranging from 1 to 5.
    """
    logging.info("Displaying image and label information with display_label_info()")
    label_image = app_utils.get_image(label_df.iloc[count]['BLOB_FILEPATH'])
    logging.info('Image retrieved from blob')
    
    label_pred = label_df.iloc[count]['PRED_LABEL']
    label_id = label_df.iloc[count]['IMAGE_ID']
    logging.info("Predicted label and image id retrieved")
    im_col,_ = st.columns([1,3])
    with im_col:
        st.image(label_image,use_column_width='always')
        logging.info("Image displayed")

    # st.write('ML Generated Label: ', label_pred)
    # st.metric('ML Generated Label:', str(label_pred))

    # st.caption('Image ID: '+ str(label_id))
    # st.write('Image ID: ', label_id)
    # st.metric('Image ID:', str(label_id))
    st.caption('Prediction: ' + str(label_pred), help='Image ID: '+ str(label_id))
    logging.info("Predicted label and image ID displayed")
    logging.info("End of display_label_info()")

def header():
    """
    Executes the Streamlit formatted HTML banner for the page.
    """
    logging.info("Running header()")
    st.markdown("""
            <h1 style='text-align: center; color: white; background-image: url(https://img.freepik.com/premium-photo/cute-colorful-abstract-background_480962-11756.jpg);
            padding-top: 70px''>
            Phytoplankton Image Validation Optimization Toolkit<br><br></h1>""",
            unsafe_allow_html=True)
    logging.info("'Phytoplankton Image Validation Optimization Toolkit' displayed")

    st.markdown("""<h1></h1>""", unsafe_allow_html=True)
    logging.info("End of header()")

def main():
    """
    Executes the Streamlit formatted HTML displayed on the Image Validation subpage. Users
    are prompted to fill out several forms used to determine the images they will label. All
    forms link to the SQL Database using functions from both sql_utils and app_utils from
    the utils folder.
    """
    logging.info("Running main()")
    state = st.session_state
    logging.info(f"State: {state}")
    with st.expander("Dataset Selection", expanded=True):
        state.container = None

        # Retrieve and display the list of datasets
        dataset_df = sql_utils.run_sql_query("SELECT * FROM dbo.datasets")
        logging.info("Dataset table has been called")
        friendly_names = dataset_df['selectbox_name']
        state.container = st.selectbox(label='Choose your dataset:',
                                       options=tuple(friendly_names),
                                       index=None)
        logging.info("Dataset selectbox created")

        if state.container is not None:
            dataset_df_row = dataset_df[dataset_df['selectbox_name'] == state.container]
            image_container = str(np.array(dataset_df_row['container'].values)[0])
            
            logging.info("Dataset selected:", image_container)
            logging.info(type(image_container))

            with open("config/config.yaml", "w", encoding="utf-8") as file:
                file.write("connection_string: " + config_dict['connection_string'] + "\n")
                file.write("image_container: " + image_container + "\n")
                file.write("server: " + config_dict['server'] + "\n")
                file.write("database: " + config_dict['database'] + "\n")
                file.write("db_user: " + config_dict['db_user'] + "\n")
                file.write("db_password: " + config_dict['db_password'] + "\n")
                file.write("subscription_id: " + config_dict['subscription_id'] + "\n")
                file.write("resource_group: " + config_dict['resource_group'] + "\n")
                file.write("workspace_name: " + config_dict['workspace_name'] + "\n")
                file.write("experiment_name: " + config_dict['experiment_name'] + "\n")
                file.write("api_key: " + config_dict['api_key'] + "\n")
                file.write("model_name: " + config_dict['model_name'] + "\n")
                file.write("endpoint_name: " + config_dict['endpoint_name'] + "\n")
                file.write("deployment_name: " + config_dict['deployment_name'] + "\n")
            
            logging.info("Config file updated")
        elif state.container is None:
            image_container = None
            st.warning("Dataset has not been selected")

    with st.expander("Session Details", expanded=True):
        st.markdown("""<h3 style='text-align: left; color: black;'>
                User Information</h3>""",
                unsafe_allow_html=True)
        logging.info("Session details expander has been expanded")

        # Session State
        if 'counter' not in state:
            state.counter = 0
        logging.info(f"State counter: {state.counter}")

        # Retrieves users information if their email exists in the SQL Database,
        # or will prompt user to enter information if it does not
        state.user_account = None

        user_col = st.columns([2,1])
        with user_col[0]:
            user_email = st.text_input(label = "Please enter your email:")
            logging.info("The user has been prompted to enter their email")

            if user_email != '':
                state.user_account = app_utils.get_user(user_email)

                if state.user_account is None:
                    st.warning("User not found. Please create a new profile.")
                    two_columns = st.columns(2)
                    with two_columns[0]:
                        user_name = st.text_input(label = "Name:")
                    with two_columns[1]:
                        user_lab = st.text_input(label = "Lab:")
                    logging.info("User has been prompted to create a new profile")

                    two_columns = st.columns(2)
                    with two_columns[0]:
                        user_domain = st.radio(label="Do you have experience in this field?",
                                            options=['Yes', 'No'],
                                            index=None)
                    with two_columns[1]:
                        user_num_labels = st.radio(
                            label="Approximately how many images have you labeled?",
                            options=['None', '25 to 100',
                                    '100 to 500', '500 to 1000',
                                    '1000+'],
                                    index=None)
                        user_experience = get_user_experience(user_num_labels, user_domain)
                    logging.info("User has been asked about their experience level")

                    new_user = {
                        'email': user_email,
                        'name': user_name,
                        'experience': user_experience,
                        'lab': user_lab
                    }
                    logging.info("New user information stored")

                    user_confirm = st.button(label="Submit", key="user_button")
                    logging.info("New user information submitted")

                    # Create a new user once submitted
                    if user_confirm:
                        app_utils.create_user(new_user)
                        st.toast("User Added!")
                        app_utils.get_user.clear()
                        state.user_account = app_utils.get_user(user_email)
                        st.rerun()
                        logging.info("New user confirmed and added to the database")

                # Display User information if they exist in Database
                elif state.user_account is not None:
                    # with st.expander("User Information"):
                    # st.markdown(f"{str(state.user_account['name'])}")
                    # st.subheader(state.user_account['name'])
                    st.success("User Found")
                    st.markdown(f"<h4 style='text-align: left; color: black;'>{state.user_account['name']}</h4>",
                        unsafe_allow_html=True)
                    info_cols = st.columns(3)
                    with info_cols[0]:
                        st.markdown(f"**Experience:** {str(state.user_account['experience'])}")
                    with info_cols[1]:
                        st.markdown(f"**Lab:** {str(state.user_account['lab'])}")
                    with info_cols[2]:
                        st.markdown(f"**Email:** {str(state.user_account['email'])}")
                    logging.info("User information displayed")

        st.divider()

        st.markdown("""<h3 style='text-align: left; color: black;'>
                    Validation Specifications</h4>""",
                    unsafe_allow_html=True)

        state.label_df = pd.DataFrame()
        models = app_utils.get_models()

        # Confirm that the app is connected to the SQL Database
        if not models:
            with st.spinner('Connecting to database'):
                db_connected = app_utils.await_connection(max_time=60,step=5)
                if not db_connected:
                    st.error("""Please ensure database configuration information is correct
                    and update on the Settings page.""")
                    logging.info("Database connection error")
                else:
                    logging.info("Connected to the database")
                    app_utils.get_models.clear()
                    app_utils.get_user.clear()
                    app_utils.get_dissimilarities.clear()
                    st.rerun()
        else:
            logging.info("Models retrieved")
            two_columns = st.columns(3)
            with two_columns[0]:

                # Retrieve and display the list of current Azure ML models
                model_dic = {}
                model_names = []

                for i in range(1,len(models)):
                    model_names.append(models[i]['m_id'])
                    model_dic[models[i]['m_id']] = models[i]['model_name']
                logging.info("Models appended to dictionary and list")

                # Prompt user to select their model of interest
                st.session_state.session_model = st.selectbox(label='Select the model you wish to validate:',
                                            options=tuple(model_names),
                                            format_func=model_dic.__getitem__,
                                            index=None)
                logging.info("Model selectbox created")

            with two_columns[1]:

                # Retrieve the dissimilarity metrics from SQL database
                dissimilarities = app_utils.get_dissimilarities()
                logging.info("Dissimilarities retrieved")
                diss_dic = {}
                diss_names = []

                for j in range(1, len(dissimilarities)):
                    diss_names.append(dissimilarities[j]['d_id'])
                    diss_dic[dissimilarities[j]['d_id']] = dissimilarities[j]['name']
                logging.info("Dissimilarities appended to dictionary and list")

                # Prompt user to select the metric of interest
                state.session_dissim = st.selectbox(label='What selection method would you like to use?',
                                            options=tuple(diss_names),
                                            format_func=diss_dic.__getitem__,
                                            index=None,
                                            help = """
                        **Entropy Score**: Entropy is the level of disorder or uncertainty in a
                        given dataset or point, ranging from 0 to 1.

                        **Least Confident Score**: The confident score represents the probability that
                        the image was labeled correctly. Images with the lowest confidence scores
                        will be displayed.

                        **Least Margin Score**: The margin score quantifies the distance between
                        a single data point to the decision boundary. Images located close to
                        the decision boundary will be displayed.
                        """)
                logging.info(f"Dissimilarity selectbox created")

            with two_columns[2]:
                state.session_number = st.number_input(label='What is the preferred image batch size?',
                                        min_value=0,
                                        max_value=200,
                                        value=10,
                                        step=5)
                logging.info(f"Batch size: {state.session_number} images")
            # Hide Advanced Specifications with an expander
            # with st.expander("Advanced Specifications"):
            two_columns = st.columns([2,1])
            with two_columns[0]:
                # Prompt user for purpose using a scale
                purpose = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                state.session_purpose = st.select_slider(
                    label="For what purpose are the images being labeled?",
                    options=purpose,
                    value=purpose[10],
                    format_func = {0.0: 'Retraining',
                                0.1: '0.1',
                                0.2: '0.2',
                                0.3: '0.3',
                                0.4: '0.4',
                                0.5: '0.5',
                                0.6: '0.6',
                                0.7: '0.7',
                                0.8: '0.8',
                                0.9: '0.9',
                                1.0: "Evaluation"}.__getitem__,
                                help="""
                        The purpose of this question is to determine the level of randomness.
                        Moving the slider toward *Evaluation* will result in a more
                        randomized selection of images while moving the slider toward 
                        *Retraining* will ensure images with less certain labels are selected.""")
                logging.info("Session purpose slider created")

            # Ensure non-Advanced Specification prompts have been answered before retrieving
            # the images with those specifications
            valid_models = state.session_model is not None
            valid_dissim = state.session_dissim is not None
            valid_session_number = state.session_number is not None
            if valid_models and valid_dissim and valid_session_number:
                logging.info("Gathering image dataframe")
                state.label_df = sql_utils.get_label_rank_df(container=image_container,
                                                    model_id=state.session_model,
                                                    dissimilarity_id=state.session_dissim,
                                                    batch_size=state.session_number,
                                                    random_ratio=state.session_purpose,
                                                    )
                # st.toast("Retrieved Images!")
                logging.info("""Retrieved labels of interest from the database based on model, dissimilarity value, 
                             batch size, and purpose""")

    st.divider()

    st.markdown("""<h3 style='text-align: left; color: black;'>
                Image Validation</h3>""",
                unsafe_allow_html=True)
    logging.info("Initializing image validation section")

    if 'new_df' not in state:
        state.new_df = pd.DataFrame(columns=['i_id', 'u_id', 'weight', 'label'])
        logging.info("New state dataframe created to store image, user, and label information")
    # Create a form of images to be labeled if there are labels that meet the user specs

    # initializing the test counter
    if 'test_counter' not in state:
        state.test_counter = 0

    if state.label_df is not None:
        if not state.label_df.empty:
            with st.form('image_validation_form', clear_on_submit=True):
                logging.info("New form created - the image validation form")
                for count in range(0, len(state.label_df)):
                    # logging.info(f"State.counter: {state.counter}")
                # st.progress(state.counter/(state.session_number -1),  
                            # text=f"{len(state.new_df)}/{state.session_number} labeled")
                # logging.info("Progress bar created")
                    # Create unique keys for form widgets
                    widget_selectbox = 'plankton_select_' + str(count)
                    widget_checkbox = 'plankton_check_' + str(count)

                    if count in state.new_df.index:
                        is_checked = True
                        logging.info(f"image at location {count} is checked")
                    else:
                        is_checked = False
                        logging.info(f"image at location {count} is not checked")

                # Show relevant label info
                    display_label_info(state.label_df, count)
                    logging.info("label info displayed: state.label_df and count")
                    logging.info(f"What are these values? state.label_df: {state.label_df}, count: {count}")

                    label_probs_options = get_label_prob_options(
                        state.label_df, count)

                    # Prompt user to label image
                    user_label = st.selectbox(
                        label="Select the correct phytoplankton subcategory:",
                        key=widget_selectbox,
                        options = label_probs_options)
                    logging.info(f"Selectbox created for user to select label. Current label: {user_label}")

                    # Add validated label to a DataFrame
                    user_add = st.checkbox(label='Confirm label',
                                           key=widget_checkbox,
                                           value=is_checked)
                    logging.info(f"Checkbox created. Value: {user_add}")
                    if is_checked is True:
                        logging.info('Checkbox is selected')

                    if user_add and not state.user_account:
                        st.error("Please submit your user information!")
                    elif user_add and state.user_account:
                        state.new_df.loc[count] = [
                                            state.label_df.iloc[count]['IMAGE_ID'],
                                            state.user_account['u_id'],
                                            state.user_account['experience'],
                                            user_label]
                        logging.info("Dataframe to store validated information has been created")
                #     st.divider()
# 
                # # Use Submit button to insert label information to SQL Database
                # back_col, next_col = st.columns(2)
                # next_disabled = (state.user_account is None)
                # if next_disabled:
                #     logging.info("Next button disabled")
                # back_disabled = (state.user_account is None)
                # if back_disabled:
                #     logging.info("Back button disabled")
                # next_tip = None
                # back_tip = None
                # if state.counter == 0:
                #     back_tip = "Try Next"
                # if state.counter == state.session_number - 1:
                #     next_tip = "Try Back or Submit"
                # if state.user_account is None:
                #     next_tip = "Enter valid user information"
                # if state.user_account is None:
                #     back_tip = "Enter valid user information"
# 
                # with back_col:
                #     back_button = st.form_submit_button("Back", disabled=back_disabled, 
                #                                         on_click=update_counter,
                #                                         args=(False, user_label),
                #                                         help=back_tip, use_container_width=True,
                #                                         type='secondary')
                #     logging.info(f"Back button created. Value: {back_button}")
                # with next_col:
                #     next_button = st.form_submit_button("Next", #disabled=next_disabled, 
                #                                         on_click=update_counter,
                #                                         args=(True, user_label),
                #                                         help=next_tip,use_container_width=True,
                #                                         type='secondary')
                #     logging.info(f"Next button created. Value: {next_button}")
# 
                submit_disabled = state.user_account is None
                if submit_disabled:
                    logging.info("Submit button disabled")
                submit_tip = None
                # if (len(st.session_state.new_df) <= 0) and not user_add:
                #     submit_tip = "No labels to submit"
                if state.user_account is None:
                    submit_tip = "Enter valid user information"

                submitted = st.form_submit_button("Submit", type='primary', use_container_width=True,
                                                    disabled=submit_disabled, help=submit_tip,
                                                    on_click=submit_labels, args=(user_label,))
                logging.info(f"Submit button created. Value: {submitted}")

                if submitted and state.user_account:
                    if len(state.new_df) > 0:
                        st.success("Your labels have been recorded!")
                        app_utils.insert_label(state.new_df)
                        state.counter = 0
                        sql_utils.get_label_rank_df.clear()
                        state.label_df = sql_utils.get_label_rank_df(container=image_container, model_id=state.session_model,
                                                    dissimilarity_id=state.session_dissim,
                                                    batch_size=state.session_number,
                                                    random_ratio=state.session_purpose)
                        state.new_df = pd.DataFrame(columns=['i_id', 'u_id', 'weight', 'label'])
                        logging.info("Labels have been recorded in the database!")

                        st.rerun()
                elif submitted and not state.user_account:
                    st.markdown("""<h5 style='text-align: left; color: black;'>
                    Please resubmit once your user information has been recorded.</h5>""",
                    unsafe_allow_html=True)
                    logging.info("No user information so cannot submit labels")
    else:
        st.error("""No images match the specification.""")
        logging.info("Error: no images match the specification")
    logging.info("End of main()")
