import streamlit as st
import pandas as pd
import os
import datetime
from transformers import pipeline

# --- Configuration and Setup ---

# 1. Set the Streamlit page configuration
st.set_page_config(
    page_title="AI Fake News Detector",
    layout="centered",
    initial_sidebar_state="expanded",
)

# 2. Define the pre-trained HuggingFace Model ID
MODEL_ID = "vikram71198/distilroberta-base-finetuned-fake-news-detection"
FEEDBACK_FILE = "feedback.csv"
LOW_CONFIDENCE_THRESHOLD = 0.60 # 60%

# 3. Load the NLP model using Streamlit's resource caching
@st.cache_resource
def load_model(model_name):
    """Loads the HuggingFace text classification pipeline."""
    try:
        # Load the pipeline for text classification
        nlp_pipeline = pipeline(
            "text-classification",
            model=model_name,
        )
        return nlp_pipeline
    except Exception as e:
        st.error(f"Error loading NLP model '{model_name}'. Check installations: {e}")
        return None

# Load the model globally
classifier = load_model(MODEL_ID)

# --- Data Saving Function ---

def save_feedback(user_feedback):
    """Saves the user feedback to a local CSV file by reading from session state."""
    # Retrieve prediction details from session state
    if 'latest_prediction' not in st.session_state:
        st.toast("❌ Error: No prediction found to save.", icon="🛑")
        return

    pred = st.session_state.latest_prediction
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a new feedback record
    new_record = {
        'timestamp': timestamp,
        'text': pred['user_input'],
        'prediction': pred['label'],
        'confidence': pred['confidence'],
        'user_feedback': user_feedback
    }
    new_df = pd.DataFrame([new_record])

    # Check if the CSV file exists
    if os.path.exists(FEEDBACK_FILE):
        # Read the existing data and append the new record
        try:
            existing_df = pd.read_csv(FEEDBACK_FILE)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            # Handle case where file exists but might be corrupted or empty
            updated_df = new_df
    else:
        # If the file doesn't exist, use the new record as the dataframe
        updated_df = new_df

    # Save the updated dataframe back to the CSV
    updated_df.to_csv(FEEDBACK_FILE, index=False)
    
    # Update state and show toast (This now works reliably inside on_click)
    st.session_state['feedback_saved'] = True
    st.toast("✅ Thank you! Your feedback has been successfully saved.", icon="💾")

# --- Feedback Display Function (NEW) ---

def display_feedback_data():
    """Loads and displays the user feedback data if the CSV file exists."""
    st.markdown("---")
    st.subheader("📚 Collected User Feedback")
    
    if os.path.exists(FEEDBACK_FILE):
        try:
            df = pd.read_csv(FEEDBACK_FILE)
            if not df.empty:
                # Use st.expander to hide the large table by default
                with st.expander(f"View {len(df)} Records of Feedback"):
                    st.dataframe(df, use_container_width=True)
            else:
                st.info("The feedback file exists, but it is currently empty.")
        except pd.errors.EmptyDataError:
            st.info("The feedback file exists, but it contains no data yet.")
        except Exception as e:
            st.error(f"Error loading feedback data: {e}")
    else:
        st.info(f"The feedback file ('{FEEDBACK_FILE}') has not been created yet. Submit a review above to create it.")


# --- UI Components ---

def show_sidebar():
    """Renders the Streamlit sidebar with documentation and download button."""
    st.sidebar.title("🧠 How This App Works")
    st.sidebar.markdown(
        """
        This is a simple proof-of-concept for a real-time fake news detector built 
        with Python and Streamlit.
        """
    )
    st.sidebar.subheader("Tech Stack")
    st.sidebar.markdown(
        f"""
        1. **Frontend:** [Streamlit](https://streamlit.io/).
        2. **Backend (ML):** [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) 
           using the pre-trained model: `{MODEL_ID}`.
        3. **Data Storage:** User feedback is saved locally to `{FEEDBACK_FILE}` 
           using [Pandas](https://pandas.python.org/).
        """
    )
    
    # Optional: Allow user to download the feedback file
    if os.path.exists(FEEDBACK_FILE):
        st.sidebar.markdown("---")
        with open(FEEDBACK_FILE, "rb") as file:
            st.sidebar.download_button(
                label="Download Feedback Data (CSV)",
                data=file,
                file_name=FEEDBACK_FILE,
                mime="text/csv"
            )


def display_result():
    """Displays the prediction and the feedback buttons based on session state."""
    
    if 'latest_prediction' not in st.session_state:
        return

    pred = st.session_state.latest_prediction
    label = pred['label']
    confidence = pred['confidence']

    st.subheader("Analysis Result")
    
    # The model uses LABEL_1 for FAKE, LABEL_0 for REAL.
    if label == 'LABEL_1':
        st.error(f"🚨 Predicted as: **FAKE NEWS**")
        st.markdown(f"Confidence (in FAKE): **{confidence:.2f}%**")
    else:
        st.success(f"✅ Predicted as: **REAL NEWS**")
        st.markdown(f"Confidence (in REAL): **{confidence:.2f}%**")

    # Bonus: Low Confidence Warning
    if confidence < LOW_CONFIDENCE_THRESHOLD * 100:
        st.warning(
            "⚠️ Low Confidence: The model is uncertain about this classification. "
            "This could be mixed, satirical, or unclear news."
        )
    
    st.markdown("---")
    st.subheader("Help Improve the Model")
    st.markdown("Was this prediction accurate?")

    # Feedback Button Layout (using columns for neatness)
    col1, col2, _ = st.columns([1, 1, 3])

    # Button logic uses on_click callback to trigger the file save (FIXED)
    is_disabled = st.session_state.get('feedback_saved', False)

    # Use on_click and arguments to ensure stability
    col1.button("✅ This result is correct", disabled=is_disabled, on_click=save_feedback, args=("Correct",))
    col2.button("❌ This result is wrong", disabled=is_disabled, on_click=save_feedback, args=("Wrong",))


# --- Main Application Logic ---

def main():
    """Main function to run the Streamlit app."""
    
    # Initialize necessary session state variables
    if 'headline_to_check' not in st.session_state:
        st.session_state['headline_to_check'] = ""
    # Reset feedback status to False ONLY if a new headline has been entered/checked
    if 'feedback_saved' not in st.session_state:
        st.session_state['feedback_saved'] = False
        
    # 1. Title and Description
    st.markdown("<h1 style='text-align: center; color: #4B0082;'>AI News Credibility Checker</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 20px;'>
        Paste any news headline or short article below. Our pre-trained 
        **DistilRoBERTa** model will attempt to classify it as 'Fake' or 'Real'.
        </div>
        """, unsafe_allow_html=True
    )
    
    # 2. Text Input
    user_input = st.text_area(
        "Enter news text here:", 
        value=st.session_state['headline_to_check'], 
        height=150
    )
    
    # 3. Check Button
    check_button = st.button("🔍 Check News Credibility", use_container_width=True, type="primary")

    # The prediction logic runs only when the button is pressed
    if check_button:
        # --- PREDICTION LOGIC ---
        
        # Reset feedback status, clear old prediction when new check starts
        st.session_state['feedback_saved'] = False
        st.session_state['headline_to_check'] = user_input
        st.markdown("---")
        
        if not user_input.strip():
            st.warning("Please paste some text into the box before checking.")
            return

        if classifier is None:
            return 
        
        with st.spinner('Running NLP model prediction...'):
            try:
                prediction = classifier(user_input, top_k=1)
                
                if prediction and isinstance(prediction, list):
                    result = prediction[0]
                    
                    # Store the result in session state (FIXED: This makes it persistent)
                    st.session_state['latest_prediction'] = {
                        'user_input': user_input,
                        'label': result['label'],
                        'confidence': result['score'] * 100 
                    }
                else:
                    st.error("Model returned an unexpected result format.")
                    # Clear session state if prediction failed
                    del st.session_state['latest_prediction']
                    return 

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                # Clear session state if prediction failed
                del st.session_state['latest_prediction']
                return 
    
    # 4. Display Result Section (FIXED: This runs on every reload if prediction exists)
    if 'latest_prediction' in st.session_state and st.session_state['headline_to_check'].strip():
        display_result()

    # 5. Display Feedback Data (NEW)
    display_feedback_data()

    # 6. Display Sidebar
    show_sidebar()

# Run the main function
if __name__ == "__main__":
    main()