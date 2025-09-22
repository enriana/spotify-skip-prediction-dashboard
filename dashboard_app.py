import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import numpy as np
from PIL import Image # Import Pillow for image handling

# Set Streamlit page configuration to wide mode and light theme
st.set_page_config(layout="wide", initial_sidebar_state="collapsed") # Removed theme argument

# Set a visually appealing style for matplotlib plots, using green shades
plt.style.use('seaborn-v0_8-whitegrid') # Start with a clean style
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', sns.color_palette("Greens_r", n_colors=6)) # Apply green color cycle

# --- Inject custom CSS for sticky title, green theme, and font/size adjustments ---
st.markdown("""
    <style>
    /* Green theme */
    .stApp {
        background-color: #f0f4f0; /* Light green background */
        color: #1e3a2d; /* Dark green text */
        font-size: 14px; /* Adjust base font size */
    }
    .css-1d3z3hw { /* Sidebar background */
        background-color: #c8e6c9; /* Medium light green */
    }
    .st-emotion-cache-e3fckw { /* Common class for the block containing the title */
        position: sticky;
        top: 0;
        background-color: #f0f4f0; /* Match app background */
        z-index: 999; /* Ensure it stays on top of other content */
        padding-top: 0.5rem; /* Adjusted padding above the title */
        padding-bottom: 0.5rem; /* Adjusted padding below the title */
    }
    h1 {
        color: #004d40; /* Darker green for main headers */
        font-size: 28px; /* Adjusted H1 font size */
    }
    h2 {
        color: #2e7d32; /* Medium green for subheaders */
        font-size: 22px; /* Adjusted H2 font size */
        margin-top: 1.5rem; /* Add space above H2 */
        margin-bottom: 0.8rem; /* Add space below H2 */
    }
    h3 {
        color: #2e7d32; /* Medium green for subheaders */
        font-size: 18px; /* Adjusted H3 font size */
        margin-top: 1rem; /* Add space above H3 */
        margin-bottom: 0.5rem; /* Add space below H3 */
    }
    .st-emotion-cache-gh2jqd { /* Markdown text */
        color: #1e3a2d; /* Dark green */
        font-size: 14px; /* Ensure markdown text uses base font size */
    }
    .stButton>button {
        background-color: #4caf50; /* Green button background */
        color: white; /* White button text */
        border-radius: 5px;
        font-size: 14px; /* Adjusted button font size */
        padding: 0.5rem 1rem; /* Adjusted button padding */
    }
    .stButton>button:hover {
        background-color: #388e3c; /* Darker green on hover */
        color: white;
    }
    .stRadio > label > div { /* Radio button label color */
         color: #1e3a2d; /* Dark green */
         font-size: 14px; /* Adjusted radio button font size */
    }
     .stRadio [data-baseweb=radio] [data-testid=stRadioLabel] { /* Radio button text */
         color: #1e3a2d; /* Dark green */
         font-size: 14px; /* Adjusted radio button text font size */
    }

    /* Adjust metric font sizes */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem; /* Adjust value font size */
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem; /* Adjust label font size */
        color: #1e3a2d; /* Ensure metric label color is dark green */
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem; /* Adjust delta font size */
    }

    /* Adjust main content area for wider layout (overridden by set_page_config) */
    /* .css-ysnqb2 {
        max-width: 1400px;
        padding-left: 2rem;
        padding-right: 2rem;
    } */

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
        st.image(spotify_logo, width=40) # Adjusted logo size
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

df_eda = load_original_df_for_eda()

# --- Overall Data Metrics ---
st.header('Overall Streaming Metrics') # Changed header text
col_period, col_metrics = st.columns([1, 3]) # Create columns for period filter and metrics

with col_period:
    st.subheader('Streaming Period')
    if not df_cleaned.empty:
        min_date = df_cleaned['ts'].min().date()
        max_date = df_cleaned['ts'].max().date()
        start_date = st.date_input("Start Date", min_date)
        end_date = st.date_input("End Date", max_date)

        # Convert date inputs to datetime objects for filtering
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)

        # Apply date filter to the cleaned dataframe
        df_filtered = df_cleaned[(df_cleaned['ts'] >= start_datetime) & (df_cleaned['ts'] <= end_datetime)]

    else:
        st.write("No data available to filter by date.")
        df_filtered = df_cleaned.copy() # Use original df if no date filter

with col_metrics:
    # Calculate metrics based on the filtered data
    total_streams = len(df_filtered)
    total_skipped = df_filtered['skipped'].sum()
    overall_skip_rate = (total_skipped / total_streams) * 100 if total_streams > 0 else 0
    total_minutes_streamed = df_filtered['ms_played'].sum() / (1000 * 60) # Convert ms to minutes
    total_unique_songs = df_filtered['track_name'].nunique() # Count unique track names
    total_unique_artists = df_filtered['artist_name'].nunique() # Count unique artist names

    st.subheader('Key Metrics') # Added subheader for metrics
    col1, col2, col3, col4, col5 = st.columns(5) # Create 5 columns for the new metrics

    with col1:
        st.metric("Total Streams", f"{total_streams:,}")
    with col2:
        st.metric("Total Skipped", f"{total_skipped:,}")
    with col3:
        st.metric("Overall Skip Rate", f"{overall_skip_rate:.2f}%")
    with col4:
        st.metric("Total Minutes Streamed", f"{total_minutes_streamed:,.2f}") # Format with comma and 2 decimal places
    with col5:
        st.metric("Total Unique Artists", f"{total_unique_artists:,}")

# --- Top Lists ---
st.header('Top Lists (within selected period)')
col_top_songs, col_top_artists, col_top_genres = st.columns(3)

with col_top_songs:
    st.subheader('Top Songs')
    if not df_filtered.empty:
        # Assuming 'track_name' is the column for song titles
        top_songs = df_filtered['track_name'].value_counts().reset_index()
        top_songs.columns = ['Song Title', 'Stream Count']
        st.dataframe(top_songs.head(10), hide_index=True) # Display top 10 as a dataframe
    else:
        st.write("No data available to show top songs.")

with col_top_artists:
    st.subheader('Top Artists')
    if not df_filtered.empty:
        # Assuming 'artist_name' is the column for artist names
        top_artists = df_filtered['artist_name'].value_counts().reset_index()
        top_artists.columns = ['Artist Name', 'Stream Count']
        st.dataframe(top_artists.head(10), hide_index=True) # Display top 10 as a dataframe
    else:
        st.write("No data available to show top artists.")

with col_top_genres:
    st.subheader('Top Genres (Requires Genre Data)')
    st.write("Genre data is not available in this dataset.")
    st.write("Analysis of top genres would require an external data source mapping tracks/artists to genres.")
    # Placeholder for genre data if it were available
    # if 'genre' in df_filtered.columns:
    #     top_genres = df_filtered['genre'].value_counts().reset_index()
    #     top_genres.columns = ['Genre', 'Stream Count']
    #     st.dataframe(top_genres.head(10))
    # else:
    #     st.write("Genre data is not available in this dataset.")


# --- Streams by Time of Day Chart ---
st.header('Streams by Time of Day (within selected period)')
if not df_filtered.empty:
    # Calculate streams per hour for the filtered data
    streams_by_hour_filtered = df_filtered['hour'].value_counts().sort_index()
    streams_by_hour_df = streams_by_hour_filtered.reset_index()
    streams_by_hour_df.columns = ['Hour', 'Stream Count']

    # Map hours to time of day categories for the filtered data
    streams_by_hour_df['Time of Day'] = streams_by_hour_df['Hour'].apply(get_time_of_day)

    # Calculate total streams per time of day category for the filtered data
    streams_by_time_of_day = streams_by_hour_df.groupby('Time of Day')['Stream Count'].sum().reindex(['Morning', 'Afternoon', 'Evening', 'Late Night']) # Ensure consistent order

    # Create the bar chart
    fig_time, ax_time = plt.subplots(figsize=(10, 6)) # Adjusted figure size
    sns.barplot(x=streams_by_time_of_day.index, y=streams_by_time_of_day.values, palette='Greens_r', ax=ax_time)
    ax_time.set_title('Total Streams by Time of Day')
    ax_time.set_xlabel('Time of Day')
    ax_time.set_ylabel('Total Stream Count')
    st.pyplot(fig_time)
    plt.close(fig_time)
else:
    st.write("No data available to show streams by time of day for the selected period.")


# --- EDA Insights Section ---
st.header('Exploratory Data Analysis (EDA) Insights') # Kept header for this section
st.write("Exploring patterns and factors related to track skips.")

# Recalculate EDA variables based on the filtered data
if not df_filtered.empty:
    skipped_df_filtered = df_filtered[df_filtered['skipped'] == True].copy()
    # Check if skipped_df_filtered is not empty before creating 'ms_played_bin'
    if not skipped_df_filtered.empty:
        skipped_df_filtered['ms_played_bin'] = pd.cut(skipped_df_filtered['ms_played'], bins=bins, labels=labels, right=False)
        skipped_bin_counts = skipped_df_filtered['ms_played_bin'].value_counts()
        total_skipped_streams_filtered = len(skipped_df_filtered)
        skipped_bin_proportions = (skipped_bin_counts / total_skipped_streams_filtered) * 100 if total_skipped_streams_filtered > 0 else pd.Series()
    else:
         skipped_bin_proportions = pd.Series()

    late_night_streams_filtered = df_filtered[df_filtered['hour'].isin(late_night_hours)]
    daytime_streams_filtered = df_filtered[~df_filtered['hour'].isin(late_night_hours)]
    late_night_skip_rate = (late_night_streams_filtered['skipped'].sum() / len(late_night_streams_filtered)) * 100 if len(late_night_streams_filtered) > 0 else 0
    daytime_skip_rate = (daytime_streams_filtered['skipped'].sum() / len(daytime_streams_filtered)) * 100 if len(daytime_streams_filtered) > 0 else 0

    platform_skipped_counts = skipped_df_filtered['platform'].value_counts() if not skipped_df_filtered.empty else pd.Series()
    platform_skipped_proportions = (platform_skipped_counts / total_skipped_streams_filtered) * 100 if total_skipped_streams_filtered > 0 else pd.Series()

    skip_rate_by_artist_frequency = df_filtered.groupby('artist_frequency')['skipped'].value_counts(normalize=True).unstack() * 100 if not df_filtered.empty else pd.DataFrame()
    skip_rate_by_artist_frequency_plot_data = pd.DataFrame()
    if not skip_rate_by_artist_frequency.empty and True in skip_rate_by_artist_frequency.columns:
        skip_rate_by_artist_frequency_plot_data = pd.DataFrame({
            'Artist Frequency': skip_rate_by_artist_frequency.index,
            'Skip Rate (%)': skip_rate_by_artist_frequency[True]
        })
else:
    # If df_filtered is empty, set all EDA variables to empty or 0
    skipped_bin_proportions = pd.Series()
    late_night_skip_rate = 0
    daytime_skip_rate = 0
    platform_skipped_proportions = pd.Series()
    skip_rate_by_artist_frequency_plot_data = pd.DataFrame()


# Create sub-columns for EDA plots
eda_plot_col1, eda_plot_col2 = st.columns(2)

with eda_plot_col1:
    # Section: Skip Rate by Play Duration
    st.subheader('1. Play Duration')
    st.write('Duration analysis.')
    if not skipped_bin_proportions.empty: # Check if data is available after filtering
        st.write("Skipped streams by duration bin:")
        fig, ax = plt.subplots(figsize=(6, 4)) # Further adjusted figure size
        sns.barplot(x=skipped_bin_proportions.index, y=skipped_bin_proportions.values, palette='Greens_r', ax=ax) # Use Green palette
        ax.set_title('Skipped Streams by ms_played Bin')
        ax.set_xlabel('Milliseconds Played Bin')
        ax.set_ylabel('Proportion (%)')
        # Highlight key insight: High proportion in <30s bin
        if '<30s' in skipped_bin_proportions.index:
            ax.annotate(f"{skipped_bin_proportions['<30s']:.1f}%",
                        xy=('<30s', skipped_bin_proportions['<30s']),
                        xytext=(5, 5), textcoords='offset points', # Adjusted text position
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black')) # Black arrow for contrast
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("Visualization for ms_played bin not available for the selected period.")


    # Section: Skip Rate by Platform
    st.subheader('3. Platform Skips')
    st.write('Platform analysis.')
    if not platform_skipped_proportions.empty: # Check if data is available after filtering
        st.write("Skipped streams by Platform:")
        fig, ax = plt.subplots(figsize=(6, 4)) # Further adjusted figure size
        sns.barplot(x=platform_skipped_proportions.index, y=platform_skipped_proportions.values, palette='Greens_r', ax=ax) # Use Green palette
        ax.set_title('Skipped Streams by Platform')
        ax.set_xlabel('Platform')
        ax.set_ylabel('Proportion (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Highlight key insight: Highest proportion for Android
        if 'android' in platform_skipped_proportions.index:
             android_val = platform_skipped_proportions['android']
             ax.annotate(f"{android_val:.1f}%",
                        xy=('android', android_val),
                        xytext=(5, 5), textcoords='offset points', # Adjusted text position
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black')) # Black arrow for contrast
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("Visualization for platform skipped proportions not available for the selected period.")


with eda_plot_col2:
    # Section: Skip Rate by Time of Day
    st.subheader('2. Time of Day')
    st.write('Time analysis.')
    # Check if both rates are available and not NaN before creating DataFrame and plotting
    if pd.notna(late_night_skip_rate) and pd.notna(daytime_skip_rate):
        skip_rates = pd.DataFrame({
            'Time Period': ['Late Night (22:00-02:00)', 'Daytime'],
            'Skip Rate (%)': [late_night_skip_rate, daytime_skip_rate]
        })
        st.write("Skip Rate by Time of Day:")
        fig, ax = plt.subplots(figsize=(6, 4)) # Further adjusted figure size
        sns.barplot(x='Time Period', y='Skip Rate (%)', data=skip_rates, palette='Greens_r', ax=ax) # Use Green palette
        ax.set_title('Skip Rate by Time of Day')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Skip Rate (%)')
        ax.set_ylim(0, 100) # Ensure y-axis is from 0 to 100 for rates
        # Highlight key insight: Higher rate for Late Night
        if 'Late Night (22:00-02:00)' in skip_rates['Time Period'].values:
            late_night_val = skip_rates[skip_rates['Time Period'] == 'Late Night (22:00-02:00)']['Skip Rate (%)'].iloc[0]
            ax.annotate(f"{late_night_val:.1f}%",
                        xy=('Late Night (22:00-02:00)', late_night_val),
                        xytext=(5, 5), textcoords='offset points', # Adjusted text position
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black')) # Black arrow for contrast
        st.pyplot(fig)
        plt.close(fig)
    else:
         st.write("Skip rate data for time of day not available for the selected period.")

    # Section: Skip Rate by Artist Frequency
    st.subheader('4. Artist Frequency')
    st.write('Artist frequency analysis.')
    if not skip_rate_by_artist_frequency_plot_data.empty: # Check if data is available after filtering
        st.write("Skip Rate by Artist Frequency:")
        fig, ax = plt.subplots(figsize=(6, 4)) # Further adjusted figure size
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
                        xytext=(5, 5), textcoords='offset points', # Adjusted text position
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black')) # Black arrow for contrast

        st.pyplot(fig)
        plt.close(fig)
    else:
         st.write("Skip rate data for artist frequency not available for the selected period.")

# Create main columns for the layout - Adjusted columns for insights and model evaluation
col_insights_recs, col_model_eval = st.columns([1, 1]) # Adjusted ratios as needed

with col_insights_recs:
    # Summary of Key Insights Section
    st.header('Summary of Key Insights')
    st.write("""
    - Skips highly likely < 30s play.
    - Higher skips during late-night hours.
    - Infrequent artists have higher skip rate.
    - Android platform accounts for high skipped proportion.
    - Reasons (start/end) and user actions (fwdbtn, backbtn) influence skips.
    """)

    # Business Recommendations Section
    st.header('Business Recommendations')
    st.write("""
    1.  **Improve Initial Engagement:** Focus on first 30s of recommendations.
    2.  **Contextual Recommendations:** Implement time-aware strategies.
    3.  **Enhance Artist Discovery:** Better introduce new/less-frequent artists.
    4.  **Investigate Android:** Address platform-specific issues contributing to skips.
    5.  **Analyze Skip Triggers:** Understand contexts of fwdbtn/backbtn/appload.
    """)

with col_model_eval:
    # Predictive Model Evaluation Section
    st.header('Predictive Model Evaluation')
    st.write("Evaluating Tuned Weighted Random Forest model performance.")

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

    st.subheader('Model Metrics')
    st.write(f"**Accuracy:** {accuracy_tuned_rf:.4f}")
    st.write(f"**Precision (Skipped):** {precision_tuned_rf:.4f}")
    st.write(f"**Recall (Skipped):** {recall_tuned_rf:.4f}")
    st.write(f"**F1-score (Skipped):** {f1_tuned_rf:.4f}")

    # Create sub-columns for model evaluation visuals
    model_viz_col1, model_viz_col2 = st.columns(2)

    with model_viz_col1:
        st.subheader('Confusion Matrix')
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5)) # Further adjusted figure size
        sns.heatmap(conf_matrix_tuned_rf, annot=True, fmt='d', cmap='Greens', ax=ax_cm, # Use Green cmap
                    xticklabels=['Not Skipped', 'Skipped'], yticklabels=['Not Skipped', 'Skipped'])
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        ax_cm.set_title('Confusion Matrix')
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        st.subheader('ROC Curve')
        y_proba_tuned_rf = tuned_rf_model.predict_proba(X_test)[:, 1] # Probability of the positive class (skipped=True)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba_tuned_rf)
        auc_score = roc_auc_score(y_test, y_proba_tuned_rf)

        fig_roc, ax_roc = plt.subplots(figsize=(4.5, 3.5)) # Further adjusted figure size
        ax_roc.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', color='green') # Use green color
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random guess')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend(loc='lower right')
        st.pyplot(fig_roc)
        plt.close(fig_roc)


    with model_viz_col2:
        st.subheader('Feature Importances')
        st.write("Top 10 influential features.") # Shortened description

        # Get feature importances from the tuned model
        importances = tuned_rf_model.feature_importances_
        # Create a DataFrame for better visualization
        feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        # Sort by importance
        feature_importances_df = feature_importances_df.sort_values('Importance', ascending=False).head(10) # Display top 10 for brevity in column

        fig_fi, ax_fi = plt.subplots(figsize=(6, 5)) # Adjusted figure size for single column
        sns.barplot(x='Importance', y='Feature', data=feature_importances_df, ax=ax_fi, palette='Greens_r') # Use Green palette
        ax_fi.set_title('Top 10 Feature Importances')
        ax_fi.set_xlabel('Importance')
        ax_fi.set_ylabel('Feature')
        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close(fig_fi)


# --- End of App ---
