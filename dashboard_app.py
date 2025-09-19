import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import numpy as np
from PIL import Image # Import Pillow for image handling

# Set a visually appealing style for matplotlib plots, using green shades
plt.style.use('seaborn-v0_8-whitegrid') # Start with a clean style
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', sns.color_palette("Greens_r", n_colors=6)) # Apply green color cycle

# --- Inject custom CSS for sticky title and green theme ---
st.markdown("""
    <style>
    /* Green theme */
    .stApp {
        background-color: #f0f4f0; /* Light green background */
        color: #1e3a2d; /* Dark green text */
    }
    .css-1d3z3hw { /* Sidebar background */
        background-color: #c8e6c9; /* Medium light green */
    }
    .st-emotion-cache-e3fckw { /* Common class for the block containing the title */
        position: sticky;
        top: 0;
        background-color: #f0f4f0; /* Match app background */
        z-index: 999; /* Ensure it stays on top of other content */
        padding-top: 1rem; /* Add some padding above the title */
        padding-bottom: 1rem; /* Add some padding below the title */
    }
    h1 {
        color: #004d40; /* Darker green for main headers */
    }
    h2, h3 {
        color: #2e7d32; /* Medium green for subheaders */
    }
    .st-emotion-cache-gh2jqd { /* Markdown text */
        color: #1e3a2d; /* Dark green */
    }
    .stButton>button {
        background-color: #4caf50; /* Green button background */
        color: white; /* White button text */
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #388e3c; /* Darker green on hover */
        color: white;
    }
    .stRadio > label > div { /* Radio button label color */
         color: #1e3a2d; /* Dark green */
    }
     .stRadio [data-baseweb=radio] [data-testid=stRadioLabel] { /* Radio button text */
         color: #1e3a2d; /* Dark green */
    }

    </style>
    """, unsafe_allow_html=True)

# --- Load Spotify Logo ---
try:
    # Assuming the logo is in the same directory as the script.
    # You might need to adjust the path if it's located elsewhere.
    spotify_logo = Image.open('spotify_logo.png') # Replace with the actual path to your logo file
except FileNotFoundError:
    st.warning("Spotify logo file not found. Please ensure 'spotify_logo.png' is in the correct directory.")
    spotify_logo = None # Set to None if not found


# --- App Title ---
col1, col2 = st.columns([1, 6]) # Create columns for logo and title
with col1:
    if spotify_logo:
        st.image(spotify_logo, width=50) # Display logo
with col2:
    st.title('Spotify Track Skips Analysis Dashboard') # Updated title
st.markdown("Exploring key factors influencing track skips and evaluating a predictive model.")

# --- Load Data ---
@st.cache_data
def load_data():
    # Load the dataset using a relative path (assuming spotify_history.csv is in the same directory as the script)
    df = pd.read_csv('spotify_history.csv')
    # Perform necessary data cleaning and preprocessing steps here as done in the notebook
    df.dropna(subset=['reason_start', 'reason_end'], inplace=True)
    df.drop_duplicates(inplace=True)
    df['ts'] = pd.to_datetime(df['ts'])
    df['hour'] = df['ts'].dt.hour
    df['dayofweek'] = df['ts'].dt.dayofweek
    def get_time_of_day(hour):
        if 5 <= hour < 12: return 'Morning'
        elif 12 <= hour < 17: return 'Afternoon'
        elif 17 <= hour < 22: return 'Evening'
        else: return 'Late Night'
    df['time_of_day'] = df['hour'].apply(get_time_of_day)
    df['played_less_than_30s'] = df['ms_played'] < 30000
    artist_stream_counts = df['artist_name'].value_counts()
    median_stream_count = artist_stream_counts.median()
    frequent_artists = artist_stream_counts[artist_stream_counts >= median_stream_count].index
    df['artist_frequency'] = df['artist_name'].apply(lambda x: 'Frequent' if x in frequent_artists else 'Infrequent')

    # One-hot encode categorical features (keep all for interaction features)
    categorical_cols = ['platform', 'reason_start', 'reason_end', 'shuffle', 'dayofweek', 'time_of_day', 'artist_frequency']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Create example interaction features (ensure base columns exist)
    if 'platform_android' in df_encoded.columns and 'reason_end_fwdbtn' in df_encoded.columns:
        df_encoded['android_fwdbtn_interaction'] = df_encoded['platform_android'] * df_encoded['reason_end_fwdbtn']
    if 'time_of_day_Late Night' in df_encoded.columns and 'platform_android' in df_encoded.columns:
         df_encoded['late_night_android_interaction'] = df_encoded['time_of_day_Late Night'] * df_encoded['platform_android']

    # Define features (X) and target (y) - drop original identifying cols and target
    target = 'skipped'
    identifying_columns = ['spotify_track_uri', 'ts', 'track_name', 'artist_name', 'album_name', 'hour'] # Exclude 'hour' as time_of_day and dayofweek are encoded
    columns_to_drop = [col for col in [target] + identifying_columns if col in df_encoded.columns]
    X = df_encoded.drop(columns=columns_to_drop)
    y = df_encoded[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, X.columns.tolist(), df # Also return feature names and the original (cleaned) df for metrics

X_train, X_test, y_train, y_test, feature_names, df_cleaned = load_data()

st.sidebar.header("Navigate Dashboard")
# Removed 'Key Performance Metrics' from the radio options
section = st.sidebar.radio("Go to", ['EDA Insights', 'Summary of Key Insights', 'Business Recommendations', 'Predictive Model Evaluation'])


# --- Calculate Variables for EDA Visualizations ---
# Need to recalculate these from the original df1 before encoding for simpler EDA
# Reload original df for EDA calculations
@st.cache_data
def load_original_df_for_eda():
    # Load the dataset using a relative path (assuming spotify_history.csv is in the same directory as the script)
    df = pd.read_csv('spotify_history.csv')
    df.dropna(subset=['reason_start', 'reason_end'], inplace=True)
    df.drop_duplicates(inplace=True)
    df['ts'] = pd.to_datetime(df['ts'])
    df['hour'] = df['ts'].dt.hour
    df['dayofweek'] = df['ts'].dt.dayofweek
    def get_time_of_day(hour):
        if 5 <= hour < 12: return 'Morning'
        elif 12 <= hour < 17: return 'Afternoon'
        elif 17 <= hour < 22: return 'Evening'
        else: return 'Late Night'
    df['time_of_day'] = df['hour'].apply(get_time_of_day)
    df['played_less_than_30s'] = df['ms_played'] < 30000
    artist_stream_counts = df['artist_name'].value_counts()
    median_stream_count = artist_stream_counts.median()
    frequent_artists = artist_stream_counts[artist_stream_counts >= median_stream_count].index
    df['artist_frequency'] = df['artist_name'].apply(lambda x: 'Frequent' if x in frequent_artists else 'Infrequent')
    return df

# Only load EDA df and calculate variables if needed for the selected section
if section in ['EDA Insights', 'Summary of Key Insights', 'Business Recommendations']: # Load if any of these sections are selected
    df_eda = load_original_df_for_eda()

    # 1. Variables for Skip Rate by ms_played Bin
    skipped_df_eda = df_eda[df_eda['skipped'] == True].copy()
    bins = [0, 30000, 60000, float('inf')]
    labels = ['<30s', '30s-60s', '>60s']
    # Check if skipped_df_eda is not empty before creating 'ms_played_bin'
    if not skipped_df_eda.empty:
        skipped_df_eda['ms_played_bin'] = pd.cut(skipped_df_eda['ms_played'], bins=bins, labels=labels, right=False)
        skipped_bin_counts = skipped_df_eda['ms_played_bin'].value_counts()
        total_skipped_streams_eda = len(skipped_df_eda)
        skipped_bin_proportions = (skipped_bin_counts / total_skipped_streams_eda) * 100 if total_skipped_streams_eda > 0 else pd.Series()
    else:
        skipped_bin_proportions = pd.Series()


    # 2. Variables for Skip Rate by Time of Day
    late_night_hours = [22, 23, 0, 1, 2]
    late_night_streams_eda = df_eda[df_eda['hour'].isin(late_night_hours)]
    daytime_streams_eda = df_eda[~df_eda['hour'].isin(late_night_hours)] # Use ~ for not in
    late_night_skip_rate = (late_night_streams_eda['skipped'].sum() / len(late_night_streams_eda)) * 100 if len(late_night_streams_eda) > 0 else 0
    daytime_skip_rate = (daytime_streams_eda['skipped'].sum() / len(daytime_streams_eda)) * 100 if len(daytime_streams_eda) > 0 else 0


    # 3. Variables for Skip Rate by Platform
    platform_skipped_counts = skipped_df_eda['platform'].value_counts() if not skipped_df_eda.empty else pd.Series()
    platform_skipped_proportions = (platform_skipped_counts / total_skipped_streams_eda) * 100 if total_skipped_streams_eda > 0 else pd.Series()


    # 4. Variables for Skip Rate by Artist Frequency
    skip_rate_by_artist_frequency = df_eda.groupby('artist_frequency')['skipped'].value_counts(normalize=True).unstack() * 100 if not df_eda.empty else pd.DataFrame()
    skip_rate_by_artist_frequency_plot_data = pd.DataFrame() # Initialize as empty
    if not skip_rate_by_artist_frequency.empty and True in skip_rate_by_artist_frequency.columns:
        skip_rate_by_artist_frequency_plot_data = pd.DataFrame({
            'Artist Frequency': skip_rate_by_artist_frequency.index,
            'Skip Rate (%)': skip_rate_by_artist_frequency[True]
        })

# --- Display Sections Based on Navigation ---

if section == 'Key Performance Metrics':
    # This section is removed as per user request to be included in EDA Insights
    pass # Keep this pass statement or remove the entire if block


elif section == 'EDA Insights':
    st.header('Exploratory Data Analysis (EDA) Insights')
    st.write("Exploring patterns and factors related to track skips.")

    # Calculate and display overall data metrics as scorecards within EDA Insights
    total_streams = len(df_cleaned)
    total_skipped = df_cleaned['skipped'].sum()
    overall_skip_rate = (total_skipped / total_streams) * 100 if total_streams > 0 else 0
    streaming_period_start = df_cleaned['ts'].min().strftime('%Y-%m-%d') if not df_cleaned.empty else 'N/A'
    streaming_period_end = df_cleaned['ts'].max().strftime('%Y-%m-%d') if not df_cleaned.empty else 'N/A'

    st.subheader('Overall Data Metrics')
    col1, col2, col3, col4 = st.columns(4) # Create 4 columns for scorecards
    with col1:
        st.metric("Total Streams", f"{total_streams:,}")
    with col2:
        st.metric("Total Skipped", f"{total_skipped:,}")
    with col3:
        st.metric("Overall Skip Rate", f"{overall_skip_rate:.2f}%")
    with col4:
        st.metric("Streaming Period", f"{streaming_period_start} to {streaming_period_end}")


    # Section: Skip Rate by Play Duration
    st.subheader('1. Skip Rate by Play Duration')
    st.write('Analysis of how the duration a track is played relates to skipping.')
    if 'skipped_bin_proportions' in locals() and not skipped_bin_proportions.empty:
        st.write("Proportion of Skipped Streams among Skipped Streams by ms_played Bin:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=skipped_bin_proportions.index, y=skipped_bin_proportions.values, palette='Greens_r', ax=ax) # Use Green palette
        ax.set_title('Proportion of Skipped Streams by ms_played Bin')
        ax.set_xlabel('Milliseconds Played Bin')
        ax.set_ylabel('Proportion of Skipped Streams (%)')
        # Highlight key insight: High proportion in <30s bin
        if '<30s' in skipped_bin_proportions.index:
            ax.annotate(f"{skipped_bin_proportions['<30s']:.1f}%",
                        xy=('<30s', skipped_bin_proportions['<30s']),
                        xytext=(10, 5), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black')) # Black arrow for contrast
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("Visualization for ms_played bin not available.")


    # Section: Skip Rate by Time of Day
    st.subheader('2. Skip Rate by Time of Day')
    st.write('Analysis of how the time of day relates to skipping (Late Night vs. Daytime).')
    # Check if both rates are available and not NaN before creating DataFrame and plotting
    if 'late_night_skip_rate' in locals() and pd.notna(late_night_skip_rate) and 'daytime_skip_rate' in locals() and pd.notna(daytime_skip_rate):
        skip_rates = pd.DataFrame({
            'Time Period': ['Late Night (22:00-02:00)', 'Daytime'],
            'Skip Rate (%)': [late_night_skip_rate, daytime_skip_rate]
        })
        st.write("Skip Rate by Time of Day:")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Time Period', y='Skip Rate (%)', data=skip_rates, palette='Greens_r', ax=ax) # Use Green palette
        ax.set_title('Skip Rate by Time of Day')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Skip Rate (%)')
        ax.set_ylim(0, 100)
        # Highlight key insight: Higher rate for Late Night
        if 'Late Night (22:00-02:00)' in skip_rates['Time Period'].values:
            late_night_val = skip_rates[skip_rates['Time Period'] == 'Late Night (22:00-02:00)']['Skip Rate (%)'].iloc[0]
            ax.annotate(f"{late_night_val:.1f}%",
                        xy=('Late Night (22:00-02:00)', late_night_val),
                        xytext=(10, 5), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black')) # Black arrow for contrast
        st.pyplot(fig)
        plt.close(fig)
    else:
         st.write("Skip rate data for time of day not available.")

    # Section: Skip Rate by Platform
    st.subheader('3. Skip Rate by Platform')
    st.write('Analysis of how the platform used relates to skipping.')
    if 'platform_skipped_proportions' in locals() and not platform_skipped_proportions.empty:
        st.write("Proportion of Skipped Streams by Platform:")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=platform_skipped_proportions.index, y=platform_skipped_proportions.values, palette='Greens_r', ax=ax) # Use Green palette
        ax.set_title('Proportion of Skipped Streams by Platform')
        ax.set_xlabel('Platform')
        ax.set_ylabel('Proportion of Skipped Streams (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Highlight key insight: Highest proportion for Android
        if 'android' in platform_skipped_proportions.index:
             android_val = platform_skipped_proportions['android']
             ax.annotate(f"{android_val:.1f}%",
                        xy=('android', android_val),
                        xytext=(10, 5), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black')) # Black arrow for contrast
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("Visualization for platform skipped proportions not available.")

    # Section: Skip Rate by Artist Frequency
    st.subheader('4. Skip Rate by Artist Frequency')
    st.write('Analysis of how artist frequency relates to skipping.')
    if 'skip_rate_by_artist_frequency_plot_data' in locals() and not skip_rate_by_artist_frequency_plot_data.empty:
        st.write("Skip Rate by Artist Frequency:")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Artist Frequency', y='Skip Rate (%)', data=skip_rate_by_artist_frequency_plot_data, palette='Greens_r', ax=ax) # Use Green palette
        ax.set_title('Skip Rate by Artist Frequency')
        ax.set_xlabel('Artist Frequency')
        ax.set_ylabel('Skip Rate (%)')
        ax.set_ylim(0, 100)
        # Highlight key insight: Higher rate for Infrequent artists
        if 'Infrequent' in skip_rate_by_artist_frequency_plot_data['Artist Frequency'].values:
             infrequent_val = skip_rate_by_artist_frequency_plot_data[skip_rate_by_artist_frequency_plot_data['Artist Frequency'] == 'Infrequent']['Skip Rate (%)'].iloc[0]
             ax.annotate(f"{infrequent_val:.1f}%",
                        xy=('Infrequent', infrequent_val),
                        xytext=(10, 5), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black')) # Black arrow for contrast

        st.pyplot(fig)
        plt.close(fig)
    else:
         st.write("Skip rate data for artist frequency not available.")


elif section == 'Summary of Key Insights':
    st.header('Summary of Key Insights')
    st.write("""
    Based on the analysis:
    - Tracks played for less than 30 seconds are highly likely to be skipped, indicating low initial engagement.
    - Skip rates are significantly higher during late-night hours compared to daytime, suggesting contextual factors influence skipping.
    - Streams by artists the user listens to less frequently have a substantially higher skip rate, pointing to potential issues with artist recommendation diversity.
    - The Android platform accounts for a very high proportion of all skipped streams, suggesting platform-specific factors might be contributing to skips for this user.
    - The reasons streams start and end also play a role, with user-initiated skips (fwdbtn, backbtn) and interruptions (appload) being notable factors.
    """)

elif section == 'Business Recommendations':
    st.header('Business Recommendations')
    st.write("""
    Based on these data-driven insights and the predictive model's findings, here are actionable strategies for improving recommendations and reducing disengagement:

    1.  **Improve Initial Track Engagement:** Focus on the first 30 seconds of recommended tracks. Experiment with different track introductions or ensure recommendations are highly relevant from the start, as skips predominantly happen very early and `ms_played` is a very important feature.
    2.  **Contextual Recommendations:** Implement time-aware recommendation strategies. Consider adjusting recommendations or playback experiences during late-night hours where skip rates are higher, reflecting its importance in the model.
    3.  **Enhance Artist Discovery:** Refine artist recommendation algorithms to better introduce new or less-frequent artists that align with the user's taste, addressing the higher skip rate for infrequent artists and their potential importance in the model. Ensure a balance between familiar and new content.
    4.  **Investigate Android Platform Experience:** Conduct a deeper analysis of the Android app's performance, UI/UX, and potential technical glitches that might be contributing to the high proportion of skipped streams originating from this platform. Address any identified frictions, as platform features are often important predictors.
    5.  **Analyze Skip Triggers:** Further investigate streams initiated by autoplay or ending with fwdbtn/backbtn to understand the specific contexts or track characteristics that trigger these user actions, as reason start/end features are also important.

    These recommendations aim to directly address the identified patterns of disengagement and leverage the insights from the data and model to potentially improve the user experience and reduce churn.
    """)


elif section == 'Predictive Model Evaluation':
    st.header('Predictive Model Evaluation')
    st.write("Evaluating the performance of the Tuned Weighted Random Forest model for predicting track skips.")

    @st.cache_resource # Cache the model training
    def train_tuned_rf_model(X_train, y_train):
        # Define the best hyperparameters found during tuning (replace with your actual best_params)
        # Example best_params - REPLACE WITH YOUR ACTUAL BEST_PARAMS FROM NOTEBOOK OUTPUT
        # Get your actual best_params from the output of the RandomizedSearchCV cell (c2730ce8)
        # Example: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 20, 'bootstrap': True}
        best_params = {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 30, 'bootstrap': True} # Replace with your actual best_params


        # Create and train the Tuned Weighted Random Forest model
        tuned_rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', **best_params)
        tuned_rf_model.fit(X_train, y_train)
        return tuned_rf_model

    # Train the tuned model
    tuned_rf_model = train_tuned_rf_model(X_train, y_train)

    # Make predictions and calculate evaluation metrics
    y_pred_tuned_rf = tuned_rf_model.predict(X_test)
    accuracy_tuned_rf = accuracy_score(y_test, y_pred_tuned_rf)
    precision_tuned_rf = precision_score(y_test, y_pred_tuned_rf)
    recall_tuned_rf = recall_score(y_test, y_pred_tuned_rf)
    f1_tuned_rf = f1_score(y_test, y_pred_tuned_rf)
    conf_matrix_tuned_rf = confusion_matrix(y_test, y_pred_tuned_rf)

    st.subheader('Model Performance Metrics (on Test Set)')
    st.write(f"**Accuracy:** {accuracy_tuned_rf:.4f}")
    st.write(f"**Precision (Skipped=True):** {precision_tuned_rf:.4f}")
    st.write(f"**Recall (Skipped=True):** {recall_tuned_rf:.4f}")
    st.write(f"**F1-score (Skipped=True):** {f1_tuned_rf:.4f}")

    st.subheader('Confusion Matrix')
    st.write("Visual representation of the model's predictions vs actual values.")

    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix_tuned_rf, annot=True, fmt='d', cmap='Greens', ax=ax_cm, # Use Green cmap
                xticklabels=['Not Skipped', 'Skipped'], yticklabels=['Not Skipped', 'Skipped'])
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    st.subheader('Feature Importances')
    st.write("Identifying which features were most influential in the model's predictions.")

    # Get feature importances from the tuned model
    importances = tuned_rf_model.feature_importances_
    # Create a DataFrame for better visualization
    feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    # Sort by importance
    feature_importances_df = feature_importances_df.sort_values('Importance', ascending=False).head(20) # Display top 20

    fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances_df, ax=ax_fi, palette='Greens_r') # Use Green palette
    ax_fi.set_title('Top 20 Feature Importances')
    ax_fi.set_xlabel('Importance')
    ax_fi.set_ylabel('Feature')
    plt.tight_layout()
    st.pyplot(fig_fi)
    plt.close(fig_fi)

    st.subheader('ROC Curve and AUC')
    st.write("Assessing the model's ability to distinguish between skipped and not-skipped tracks.")

    y_proba_tuned_rf = tuned_rf_model.predict_proba(X_test)[:, 1] # Probability of the positive class (skipped=True)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba_tuned_rf)
    auc_score = roc_auc_score(y_test, y_proba_tuned_rf)

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})', color='green') # Use green color
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random guess')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc='lower right')
    st.pyplot(fig_roc)
    plt.close(fig_roc)


# --- End of App ---
