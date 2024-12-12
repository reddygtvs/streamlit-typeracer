import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 6)
st.title("Race Data Analysis Dashboard")
st.markdown("""
Welcome to my Race Data Analysis Dashboard! This application allows me to upload my `race_data.csv` file and automatically generates insightful visualizations to help me understand my racing performance.
""")

st.sidebar.header("Upload Your Race Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        st.stop()

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Data Cleaning and Preprocessing")
    try:
        df.rename(columns={
            'Race #': 'Race_Num',
            'WPM': 'WPM',
            'Accuracy': 'Accuracy',
            'Rank': 'Rank',
            '# Racers': 'Num_Racers',
            'Text ID': 'Text_ID',
            'Date/Time (UTC)': 'DateTime_UTC'
        }, inplace=True)

        # Parse datetime and sort the data
        df['DateTime_UTC'] = pd.to_datetime(df['DateTime_UTC'], dayfirst=True, infer_datetime_format=True, errors='coerce')
        df.dropna(subset=['DateTime_UTC'], inplace=True)
        df = df.sort_values('DateTime_UTC').reset_index(drop=True)
        df['YearMonth'] = df['DateTime_UTC'].dt.to_period('M').dt.to_timestamp()
        df['Date'] = df['DateTime_UTC'].dt.date
        df['Hour'] = df['DateTime_UTC'].dt.hour
        df['DayOfWeek'] = df['DateTime_UTC'].dt.day_name()

        df['Win'] = (df['Rank'] == 1).astype(int)

        st.success("Data successfully cleaned and preprocessed.")
    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        st.stop()

    def generate_plots(df):
        # Sorted by Race_Num
        df_sorted = df.sort_values('Race_Num').reset_index(drop=True)

        st.header("Part 1: Introduction and Data Overview")
        st.markdown("""
        ## Data Summary

        - **Data Source:** [typeracer.com](https://typeracer.com)
        - **All-Time Average WPM:** **97.3**
        - **Best Race WPM:** **186**
        - **Total Races:** **34,686**
        - **Total Wins:** **13,797**
        - **First Race Date:** **24/05/2020**
        - **Performance Comparison:** **99.7% WPM** compared to all racers on Typeracer
        """)

        # Plot #1: Distribution of WPM
        st.subheader("Distribution of Words Per Minute (WPM)")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['WPM'], kde=True, bins=30, color='skyblue', ax=ax1)
        ax1.set_title("Distribution of WPM")
        ax1.set_xlabel("Words Per Minute (WPM)")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

        # Plot #2: Distribution of Accuracy
        st.subheader("Distribution of Accuracy")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['Accuracy'], kde=True, bins=30, color='coral', ax=ax2)
        ax2.set_title("Distribution of Accuracy")
        ax2.set_xlabel("Accuracy")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

        # Plot #9: Rank Distribution
        st.subheader("Rank Distribution (Percentage of Races)")
        rank_counts = df['Rank'].value_counts(normalize=True).reset_index()
        rank_counts.columns = ['Rank', 'Percentage']
        fig9, ax9 = plt.subplots()
        sns.barplot(x='Rank', y='Percentage', data=rank_counts, palette="muted", ax=ax9)
        ax9.set_title("Rank Distribution (Percentage of Races)")
        ax9.set_xlabel("Rank")
        ax9.set_ylabel("Percentage (%)")
        st.pyplot(fig9)

        # Plot #10: Accuracy vs. Rank
        st.subheader("Average Accuracy by Rank")
        rank_accuracy = df.groupby('Rank')['Accuracy'].mean().reset_index()
        fig10, ax10 = plt.subplots()
        sns.barplot(x='Rank', y='Accuracy', data=rank_accuracy, palette="Set2", ax=ax10)
        ax10.set_title("Average Accuracy by Rank")
        ax10.set_xlabel("Rank")
        ax10.set_ylabel("Average Accuracy")
        st.pyplot(fig10)

        
        st.header("Part 2: Temporal Trends in Performance")

        # Plot #3: Monthly Average WPM Over Time
        st.subheader("Monthly Average WPM Over Time")
        monthly_avg = df.groupby('YearMonth')['WPM'].mean().reset_index()
        fig3, ax3 = plt.subplots()
        sns.lineplot(x='YearMonth', y='WPM', data=monthly_avg, marker='o', color='blue', ax=ax3)
        ax3.set_title("Monthly Average WPM Over Time")
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Average WPM")
        plt.xticks(rotation=45)
        st.pyplot(fig3)

        # Plot #4: Daily Average WPM Over Time
        st.subheader("Daily Average WPM Over Time")
        daily_avg = df.groupby('Date')['WPM'].mean().reset_index()
        fig4, ax4 = plt.subplots()
        sns.lineplot(x='Date', y='WPM', data=daily_avg, color='orange', ax=ax4)
        ax4.set_title("Daily Average WPM Over Time")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Average WPM")
        plt.xticks(rotation=45)
        st.pyplot(fig4)

        # Plot #5: Rolling Average WPM (Every 100 Races)
        st.subheader("Rolling Average WPM (Every 100 Races)")
        rolling_avg = df_sorted[['Race_Num', 'WPM']].copy()
        rolling_avg['Rolling_Avg_WPM'] = rolling_avg['WPM'].rolling(window=100).mean()
        fig5, ax5 = plt.subplots()
        sns.lineplot(x='Race_Num', y='Rolling_Avg_WPM', data=rolling_avg, color='green', ax=ax5)
        ax5.set_title("Rolling Average WPM (Every 100 Races)")
        ax5.set_xlabel("Race Number")
        ax5.set_ylabel("Rolling Average WPM")
        st.pyplot(fig5)

        # Plot #22: Cumulative Accuracy Over Time
        st.subheader("Cumulative Average Accuracy Over Time")
        df_sorted_cum_acc = df.sort_values('DateTime_UTC').reset_index(drop=True)
        df_sorted_cum_acc['Cumulative_Accuracy'] = df_sorted_cum_acc['Accuracy'].expanding().mean()

        fig22, ax22 = plt.subplots()
        sns.lineplot(x='Race_Num', y='Cumulative_Accuracy', data=df_sorted_cum_acc, color='purple', ax=ax22)
        ax22.set_title("Cumulative Average Accuracy Over All Races")
        ax22.set_xlabel("Race Number")
        ax22.set_ylabel("Cumulative Average Accuracy")
        plt.tight_layout()
        st.pyplot(fig22)

 
        st.header("Part 3: Performance by Context")

        # Plot #6: WPM by Time of Day
        st.subheader("Average WPM by Time of Day")
        hourly_avg = df.groupby('Hour')['WPM'].mean().reset_index()
        fig6, ax6 = plt.subplots()
        sns.barplot(x='Hour', y='WPM', data=hourly_avg, palette="Blues_d", ax=ax6)
        ax6.set_title("Average WPM by Time of Day")
        ax6.set_xlabel("Hour of Day")
        ax6.set_ylabel("Average WPM")
        st.pyplot(fig6)

        # Plot #7: WPM Improvement Over Time for Frequent Texts
        st.subheader("WPM Improvement Over Time for Frequent Texts")
        top_texts = df['Text_ID'].value_counts().head(10).index
        top_texts_df = df[df['Text_ID'].isin(top_texts)]
        fig7, ax7 = plt.subplots()
        sns.lineplot(data=top_texts_df, x='DateTime_UTC', y='WPM', hue='Text_ID', palette='tab10', ax=ax7)
        ax7.set_title("WPM Improvement Over Time for Frequent Texts")
        ax7.set_xlabel("Date")
        ax7.set_ylabel("WPM")
        ax7.legend(title="Text ID", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig7)

        # Plot #8: Top 10 vs Bottom 10 Texts by Average WPM
        st.subheader("Top 10 vs Bottom 10 Texts by Average WPM")
        text_wpm = df.groupby('Text_ID')['WPM'].mean()
        top_texts = text_wpm.nlargest(10)
        bottom_texts = text_wpm.nsmallest(10)
        top_bottom_texts = pd.concat([top_texts, bottom_texts])

        fig8, ax8 = plt.subplots(figsize=(12,8))
        sns.barplot(x=top_bottom_texts.index.astype(str), y=top_bottom_texts.values, palette="pastel", ax=ax8)
        ax8.set_title("Top 10 vs Bottom 10 Texts by WPM")
        ax8.set_xlabel("Text ID")
        ax8.set_ylabel("Average WPM")
        ax8.set_xticklabels(ax8.get_xticklabels(), rotation=45)
        plt.tight_layout()
        st.pyplot(fig8)

        # Plot #21: WPM vs Accuracy
        st.subheader("WPM vs Accuracy")
        fig21, ax21 = plt.subplots()
        sns.scatterplot(x='WPM', y='Accuracy', data=df, alpha=0.6, ax=ax21)
        ax21.set_title("WPM vs Accuracy")
        ax21.set_xlabel("Words Per Minute")
        ax21.set_ylabel("Accuracy")
        plt.tight_layout()
        st.pyplot(fig21)

        # Plot #23: WPM Distribution for Top 10 Texts
        st.subheader("WPM Distribution for Top 10 Texts")
        text_wpm_avg = df.groupby('Text_ID')['WPM'].mean()
        top_10_texts = df['Text_ID'].value_counts().head(10).index

        fig23, ax23 = plt.subplots(figsize=(14, 8))
        for text_id in top_10_texts:
            subset = df[df['Text_ID'] == text_id]
            sns.kdeplot(subset['WPM'], label=f'Text ID {text_id}', shade=True, ax=ax23)
        ax23.set_title("WPM Distribution for Top 10 Texts")
        ax23.set_xlabel("WPM")
        ax23.set_ylabel("Density")
        ax23.legend(title="Text ID", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig23)

        
        st.header("Part 4: Ranking and Win Analysis")

        # Plot #11: Monthly Win Rate Over Time
        st.subheader("Monthly Win Rate Over Time")
        win_rate_monthly = df.groupby('YearMonth')['Win'].mean().reset_index()
        win_rate_monthly.rename(columns={'Win': 'Win_Rate'}, inplace=True)
        fig11, ax11 = plt.subplots()
        sns.lineplot(x='YearMonth', y='Win_Rate', data=win_rate_monthly, marker='o', color='gold', ax=ax11)
        ax11.set_title("Monthly Win Rate Over Time")
        ax11.set_xlabel("Month")
        ax11.set_ylabel("Win Rate (%)")
        plt.xticks(rotation=45)
        st.pyplot(fig11)

        # Plot #17: Win Rate After a Previous Win
        st.subheader("Win Rate After a Previous Win")
        if 'Win' in df_sorted.columns:
            df_sorted['Prev_Win'] = df_sorted['Win'].shift(1).fillna(0).astype(int)
            win_rate_after_prev = df_sorted.groupby('Prev_Win')['Win'].mean().reset_index()
            win_rate_after_prev['Condition'] = win_rate_after_prev['Prev_Win'].map({1: 'After Win', 0: 'After Not Win'})
            fig17, ax17 = plt.subplots(figsize=(8,6))
            sns.barplot(x='Condition', y='Win', data=win_rate_after_prev, palette="Set1", ax=ax17)
            ax17.set_title("Win Rate After Previous Race")
            ax17.set_xlabel("Condition")
            ax17.set_ylabel("Win Rate (%)")
            st.pyplot(fig17)
        else:
            st.warning("Win column not found. Ensure that the 'Rank' column is correctly named and contains numerical values.")

        # Plot #25: Top 5 Fastest and Slowest Races
        st.subheader("Top 5 Fastest and Slowest Races")
        top_fastest = df.nlargest(5, 'WPM')
        top_slowest = df.nsmallest(5, 'WPM')

        fig25, ax25 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Race_Num', y='WPM', data=top_fastest, color='red', label='Top 5 Fastest', ax=ax25)
        sns.scatterplot(x='Race_Num', y='WPM', data=top_slowest, color='blue', label='Top 5 Slowest', ax=ax25)
        ax25.set_title("Top 5 Fastest and Slowest Races")
        ax25.set_xlabel("Race Number")
        ax25.set_ylabel("WPM")
        ax25.legend()
        plt.tight_layout()
        st.pyplot(fig25)

        
        st.header("Part 5: Performance Consistency and Outliers")

        # Plot #18: Consistency Score Over Time
        st.subheader("Consistency Score Over Time")
        rolling_window = 30  # races
        df_sorted['Rolling_Std_WPM'] = df_sorted['WPM'].rolling(window=rolling_window).std()
        fig18, ax18 = plt.subplots()
        sns.lineplot(x='Race_Num', y='Rolling_Std_WPM', data=df_sorted, color='orange', ax=ax18)
        ax18.set_title(f"Rolling {rolling_window}-Race Standard Deviation of WPM (Consistency Score)")
        ax18.set_xlabel("Race Number")
        ax18.set_ylabel("WPM Standard Deviation")
        st.pyplot(fig18)

        # Plot #24: Outlier Analysis: WPM by Rank
        
        st.subheader("Outlier Analysis: WPM by Rank")
        fig24, ax24 = plt.subplots()
        sns.boxplot(x='Rank', y='WPM', data=df, palette="Set3", ax=ax24)
        ax24.set_title("Outlier Analysis: WPM by Rank")
        ax24.set_xlabel("Rank")
        ax24.set_ylabel("WPM")
        plt.tight_layout()
        st.pyplot(fig24)

        st.header("Part 6: Impact of External Factors")

        # Plot #14: Impact of Number of Racers on Average WPM
        st.subheader("Impact of Number of Racers on Average WPM")
        avg_wpm_racers = df.groupby('Num_Racers')['WPM'].mean().reset_index()
        fig14, ax14 = plt.subplots()
        sns.lineplot(x='Num_Racers', y='WPM', data=avg_wpm_racers, marker='o', color='teal', ax=ax14)
        ax14.set_title("Impact of Number of Racers on Average WPM")
        ax14.set_xlabel("Number of Racers")
        ax14.set_ylabel("Average WPM")
        st.pyplot(fig14)

        # Plot #33: Time Between Races and Performance
        st.subheader("Time Between Races and Performance")
        df_sorted_time = df.sort_values('DateTime_UTC').reset_index(drop=True)
        df_sorted_time['Time_Diff'] = df_sorted_time['DateTime_UTC'].diff().dt.total_seconds().div(3600).fillna(0)
        bins = [0, 1, 3, 6, 12, 24, 48, 168, np.inf]
        labels = ['0-1h', '1-3h', '3-6h', '6-12h', '12-24h', '24-48h', '48h-1wk', '1wk+']
        df_sorted_time['Time_Diff_Bin'] = pd.cut(df_sorted_time['Time_Diff'], bins=bins, labels=labels)

        avg_wpm_time_diff = df_sorted_time.groupby('Time_Diff_Bin')['WPM'].mean().reset_index()

        fig33, ax33 = plt.subplots(figsize=(12,6))
        sns.barplot(x='Time_Diff_Bin', y='WPM', data=avg_wpm_time_diff, palette="coolwarm", ax=ax33)
        ax33.set_title("Average WPM by Time Between Races")
        ax33.set_xlabel("Time Between Races")
        ax33.set_ylabel("Average WPM")
        ax33.set_xticklabels(ax33.get_xticklabels(), rotation=45)
        plt.tight_layout()
        st.pyplot(fig33)

        st.header("Part 7: Forecasting Future Performance")

        st.subheader("Forecast: Rolling Average WPM Projection to 2030")

        df_sorted_plot26 = df.sort_values('Race_Num').reset_index(drop=True)
        df_sorted_plot26['Rolling_Avg_WPM'] = df_sorted_plot26['WPM'].rolling(window=100).mean()

        df_rolling = df_sorted_plot26[['Race_Num', 'Rolling_Avg_WPM']].dropna()
        X = df_rolling[['Race_Num']]
        y = df_rolling['Rolling_Avg_WPM']

        model = LinearRegression()
        model.fit(X, y)
        last_race_num = df_sorted_plot26['Race_Num'].max()
        current_date = df_sorted_plot26['DateTime_UTC'].max()
        end_date = pd.to_datetime('2030-12-31')
        total_days = (current_date - df_sorted_plot26['DateTime_UTC'].min()).days
        total_races = df_sorted_plot26['Race_Num'].max()
        avg_time_between_races = total_days / total_races if total_races > 0 else 1  

        future_days = (end_date - current_date).days
        future_races = int(future_days / avg_time_between_races)
        future_race_nums = np.arange(last_race_num + 1, last_race_num + future_races + 1).reshape(-1, 1)
        predicted_wpm = model.predict(future_race_nums)

        future_df = pd.DataFrame({
            'Race_Num': future_race_nums.flatten(),
            'Rolling_Avg_WPM': predicted_wpm
        })

        fig26, ax26 = plt.subplots(figsize=(12,6))
        sns.lineplot(x='Race_Num', y='Rolling_Avg_WPM', data=df_rolling, label='Historical', ax=ax26)
        sns.lineplot(x='Race_Num', y='Rolling_Avg_WPM', data=future_df, label='Predicted', ax=ax26)
        ax26.set_title("Rolling Average WPM Forecast to 2030")
        ax26.set_xlabel("Race Number")
        ax26.set_ylabel("Rolling Average WPM")
        plt.tight_layout()
        st.pyplot(fig26)

    generate_plots(df)

    st.sidebar.header("Download Data")
    processed_data = df.copy()
    csv = processed_data.to_csv(index=False)
    st.sidebar.download_button(
        label="Download Processed Data as CSV",
        data=csv,
        file_name='processed_race_data.csv',
        mime='text/csv',
    )

else:
    st.info("Please upload a `race_data.csv` file to proceed.")
