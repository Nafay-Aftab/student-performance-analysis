import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from numerize.numerize import numerize
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

@st.cache_data
def load_data():
    # Load the CSV file from the local directory
    df = pd.read_csv('StudentsPerformance.csv')

    # Calculate the average score
    df['Average Score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

    # Define performance labels
    def performance_label(avg):
        if avg > 80:
            return 'Excellent'
        elif 60 <= avg <= 80:
            return 'Good'
        else:
            return 'Average'

    # Apply the labels
    df['Performance Label'] = df['Average Score'].apply(performance_label)

    return df

# Load the dataset
df = load_data()

DARK_BG = '#121212'
TEXT_COLOR = 'white'
GRID_COLOR = '#444444'

def apply_dark_theme(ax, fig):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.title.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.tick_params(axis='x', colors=TEXT_COLOR)
    ax.tick_params(axis='y', colors=TEXT_COLOR)
    ax.grid(color=GRID_COLOR)
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
    # Also set bar label colors in caller functions

def plot_gender_distribution(df):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.set_style("darkgrid")
    sns.countplot(x='gender', data=df, palette='Set2', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, color=TEXT_COLOR)
    apply_dark_theme(ax, fig)
    ax.set_title('Gender Distribution of Students', fontsize=18)
    ax.set_xlabel('Gender', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    plt.tight_layout()
    return fig

def plot_lunch_distribution(df):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.set_style("darkgrid")
    sns.countplot(x='lunch', data=df, palette='Pastel1', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, color=TEXT_COLOR)
    apply_dark_theme(ax, fig)
    ax.set_title('Standard vs Free/Reduced Lunch', fontsize=17)
    ax.set_xlabel('Lunch Type', fontsize=14)
    ax.set_ylabel('Student Count', fontsize=14)
    plt.tight_layout()
    return fig

def plot_test_preparation_status(df):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.set_style("darkgrid")
    sns.countplot(x='test preparation course', data=df, palette='Accent', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, color=TEXT_COLOR)
    apply_dark_theme(ax, fig)
    ax.set_title('Test Preparation Status', fontsize=17)
    ax.set_xlabel('Preparation Course', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    plt.tight_layout()
    return fig

def plot_education_distribution(df):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.set_style("darkgrid")
    sns.countplot(x='parental level of education', data=df, palette='Set3', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, color=TEXT_COLOR)
    apply_dark_theme(ax, fig)
    ax.set_title('Parental Education Level Distribution', fontsize=17)
    ax.set_xlabel('Education Type', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    plt.tight_layout()
    return fig

def plot_scores_distribution(df):
    scores = ['math score', 'reading score', 'writing score']
    colors = ['#FF6F61', '#6B5B95', '#88B04B']
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.patch.set_facecolor(DARK_BG)
    for i, score in enumerate(scores):
        mean = df[score].mean()
        median = df[score].median()
        skewness = df[score].skew()
        axs[i].set_facecolor(DARK_BG)
        sns.histplot(data=df, x=score, ax=axs[i], color=colors[i], bins=20)
        axs[i].axvline(mean, color='red', linewidth=2, label=f'Mean: {mean:.2f}')
        axs[i].axvline(median, color='blue', linewidth=2, label=f'Median: {median:.2f}')
        legend = axs[i].legend(title=f"Skewness: {skewness:.2f}")
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
        for text in legend.get_title().get_children():
            text.set_color(TEXT_COLOR)
        axs[i].title.set_color(TEXT_COLOR)
        axs[i].xaxis.label.set_color(TEXT_COLOR)
        axs[i].yaxis.label.set_color(TEXT_COLOR)
        axs[i].tick_params(axis='x', colors=TEXT_COLOR)
        axs[i].tick_params(axis='y', colors=TEXT_COLOR)
        axs[i].grid(color=GRID_COLOR)
    fig.suptitle('Score Distributions', fontsize=16, y=0.98, color=TEXT_COLOR)
    fig.subplots_adjust(top=0.90, bottom=0.12, hspace=0.4)
    plt.tight_layout()
    return fig

def plot_avg_scores_by_gender(df):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.set_style('darkgrid')
    group = df.groupby('gender')[['math score','reading score','writing score']].mean().reset_index()
    group_long = group.melt(id_vars='gender', var_name='Subject', value_name='Average Score')
    sns.barplot(data=group_long, x='Subject', y='Average Score', hue='gender', palette='coolwarm', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', color=TEXT_COLOR)
    apply_dark_theme(ax, fig)
    ax.set_title('Average Scores by Gender and Subject', fontsize=17)
    ax.set_xlabel('Subject', fontsize=14)
    ax.set_ylabel('Average Score', fontsize=14)
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    plt.tight_layout()
    return fig

def plot_avg_scores_by_test_prep(df):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.set_style('darkgrid')
    group = df.groupby('test preparation course')[['math score','reading score','writing score']].mean().reset_index()
    group_long = group.melt(id_vars='test preparation course', value_name='Average Score', var_name='Subject')
    sns.barplot(data=group_long, x='Subject', y='Average Score', hue='test preparation course', palette='Spectral', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', color=TEXT_COLOR)
    apply_dark_theme(ax, fig)
    ax.set_title('Scores by Test Preparation Status', fontsize=17)
    ax.set_xlabel('Subject', fontsize=14)
    ax.set_ylabel('Average Score', fontsize=14)
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    plt.tight_layout()
    return fig

def plot_avg_scores_by_lunch(df):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.set_style('darkgrid')
    group = df.groupby('lunch')[['math score','reading score','writing score']].mean().reset_index()
    group_long = group.melt(id_vars='lunch', value_name='Average Score', var_name='Subject')
    sns.barplot(data=group_long, x='Subject', y='Average Score', hue='lunch', palette='Set1', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', color=TEXT_COLOR)
    apply_dark_theme(ax, fig)
    ax.set_title('Lunch Type vs Scores', fontsize=17)
    ax.set_xlabel('Subject', fontsize=14)
    ax.set_ylabel('Average Score', fontsize=14)
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    plt.tight_layout()
    return fig

def plot_avg_scores_by_parent_education(df):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.set_style('darkgrid')
    group = df.groupby('parental level of education')[['math score','reading score','writing score']].mean().reset_index()
    group_long = group.melt(id_vars='parental level of education', value_name='Average Score', var_name='Subject')
    sns.barplot(data=group_long, x='Subject', y='Average Score', hue='parental level of education', palette='tab20', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', color=TEXT_COLOR)
    apply_dark_theme(ax, fig)
    ax.set_title('Parent Education vs Scores', fontsize=17)
    ax.set_xlabel('Subject', fontsize=14)
    ax.set_ylabel('Average Score', fontsize=14)
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig

def plot_math_vs_other_subjects(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor(DARK_BG)
    for ax, color, title, ylabel in zip(
        axes,
        ['#FFA07A', '#9370DB'],
        ['Math vs Reading', 'Math vs Writing'],
        ['Reading Score', 'Writing Score']
    ):
        ax.set_facecolor(DARK_BG)
        sns.scatterplot(data=df, x='math score', y=ylabel.lower(), ax=ax, color=color, alpha=0.6)
        ax.set_title(title, color=TEXT_COLOR)
        ax.set_xlabel('Math Score', color=TEXT_COLOR)
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
        ax.tick_params(axis='x', colors=TEXT_COLOR)
        ax.tick_params(axis='y', colors=TEXT_COLOR)
        ax.grid(color=GRID_COLOR)
    plt.suptitle('Math Performance Comparison', fontsize=17, color=TEXT_COLOR)
    plt.tight_layout()
    return fig

def plot_math_score_by_gender_and_prep(df):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.set_style('darkgrid')
    sns.barplot(x='gender', y='math score', hue='test preparation course', data=df, ci=None, palette='Paired', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', color=TEXT_COLOR)
    apply_dark_theme(ax, fig)
    ax.set_title('Math Scores by Gender & Prep', fontsize=17)
    ax.set_xlabel('Gender', fontsize=14)
    ax.set_ylabel('Math Score', fontsize=14)
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    plt.tight_layout()
    return fig

def plot_lunch_and_prep_impact_on_writing(df):
    group = df.groupby(['lunch', 'test preparation course'])['writing score'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_style('darkgrid')
    sns.barplot(data=group, x='lunch', y='writing score', hue='test preparation course', palette='cool', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', color=TEXT_COLOR)
    apply_dark_theme(ax, fig)
    ax.set_title('Lunch & Prep vs Writing Scores', fontsize=17)
    ax.set_xlabel('Lunch Type', fontsize=14)
    ax.set_ylabel('Avg Writing Score', fontsize=14)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    return fig

def plot_performance_by_gender(df):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.set_style('darkgrid')
    sns.countplot(data=df, x='Performance Label', hue='gender', palette='Set1', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, color=TEXT_COLOR)
    apply_dark_theme(ax, fig)
    ax.set_title('Performance by Gender', fontsize=17)
    ax.set_xlabel('Performance Label', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    plt.tight_layout()
    return fig

def plot_gender_in_parent_education(df):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.set_style('darkgrid')
    sns.countplot(data=df, x='parental level of education', hue='gender', palette='Dark2', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, color=TEXT_COLOR)
    apply_dark_theme(ax, fig)
    ax.set_title('Gender vs Parent Education', fontsize=17)
    ax.set_xlabel('Parental Education', fontsize=14)
    ax.set_ylabel('Student Count', fontsize=14)
    plt.tight_layout()
    return fig



st.title("📊 Student Performance Dashboard")
st.markdown("<br>", unsafe_allow_html=True)
st.sidebar.title("🧮 Select and Filter Data")

option = st.sidebar.selectbox(
    label="📊 Choose a graph to display:",
    options=[
        "Gender Distribution",
        "Performace By Gender",
        "Gender in Parent Education",
        "Lunch Distribution",
        "Test Preparation Status",
        "Education Distribution",
        "Scores Distribution",
        "Average Scores By Genders",
        "Average Scores By Test Preparation",
        "Average Scores By Lunch",
        "Average Scores By Parent Education",
        "Maths Vs Other Subjects",
        "Maths Score By Gender and Preparation",
        "Lunch and Preparation Impact on Writing"
    ],
    index=0 
)

with st.sidebar.expander(label='🔍 Select the filters', expanded=True):
    gender = st.multiselect('Select Gender:', options=df['gender'].unique(), default=df['gender'].unique())
    race = st.multiselect('Select Race/Etnicity:', options=df['race/ethnicity'].unique(), default=df['race/ethnicity'].unique())
    parent = st.multiselect('Select Parental Level of Educatuon:', options=df['parental level of education'].unique(), default=df['parental level of education'].unique())
    lunch = st.multiselect('Select Type of Lunch:', options=df['lunch'].unique(), default=df['lunch'].unique())
    test = st.multiselect('Select Test Preparation Status:', options=df['test preparation course'].unique(), default=df['test preparation course'].unique())

filtered_df = df.query(
    'gender in @gender and '
    '`race/ethnicity` in @race and '
    '`parental level of education` in @parent and '
    'lunch in @lunch and '
    '`test preparation course` in @test'
)


if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filters. Please adjust your filter selections.")
else:
    col1, col2, col3, col4 = st.columns(4, gap='medium')
    with col1:
        st.metric(label='📊 Math Avg', value=f"{filtered_df['math score'].mean():.1f}")
    with col2:
        st.metric(label='📚 Reading Avg', value=f"{filtered_df['reading score'].mean():.1f}")
    with col3:
        st.metric(label='✍️ Writing Avg', value=f"{filtered_df['writing score'].mean():.1f}")
    with col4:
        total_students = filtered_df.shape[0]
        st.metric(label='🎓 Total Students', value=numerize(total_students))

    col5, col6, col7, col8 = st.columns(4, gap='medium')
    with col5:
        male_count = filtered_df[filtered_df['gender'] == 'male'].shape[0]
        st.metric(label='👦 Male Students', value=numerize(male_count))
    with col6:
        female_count = filtered_df[filtered_df['gender'] == 'female'].shape[0]
        st.metric(label='👧 Female Students', value=numerize(female_count))
    with col7:
        excellent = filtered_df[filtered_df['Performance Label'] == 'Excellent'].shape[0]
        st.metric(label='🏆 Excellent Performer', value=numerize(excellent))
    with col8:
        good = filtered_df[filtered_df['Performance Label'] == 'Good'].shape[0]
        st.metric(label='👍 Good Performer', value=numerize(good))

    col9, _, _ = st.columns(3, gap='medium')
    with col9:
        avg_count = filtered_df[filtered_df['Performance Label'] == 'Average'].shape[0]
        st.metric(label='🙂 Average Performer', value=numerize(avg_count))

    st.header("📈 Data Visualization")

    if option == 'Gender Distribution':
        fig = plot_gender_distribution(filtered_df)
        st.pyplot(fig)
    elif option == 'Performace By Gender':
        fig = plot_performance_by_gender(filtered_df)
        st.pyplot(fig)
    elif option == 'Gender in Parent Education':
        fig = plot_gender_in_parent_education(filtered_df)
        st.pyplot(fig)
    elif option == 'Lunch Distribution':
        fig = plot_lunch_distribution(filtered_df)
        st.pyplot(fig)
    elif option == 'Test Preparation Status':
        fig = plot_test_preparation_status(filtered_df)
        st.pyplot(fig)
    elif option == 'Education Distribution':
        fig = plot_education_distribution(filtered_df)
        st.pyplot(fig)
    elif option == 'Scores Distribution':
        fig = plot_scores_distribution(filtered_df)
        st.pyplot(fig)
    elif option == 'Average Scores By Genders':
        fig = plot_avg_scores_by_gender(filtered_df)
        st.pyplot(fig)
    elif option == 'Average Scores By Test Preparation':
        fig = plot_avg_scores_by_test_prep(filtered_df)
        st.pyplot(fig)
    elif option == 'Average Scores By Lunch':
        fig = plot_avg_scores_by_lunch(filtered_df)
        st.pyplot(fig)
    elif option == 'Average Scores By Parent Education':
        fig = plot_avg_scores_by_parent_education(filtered_df)
        st.pyplot(fig)
    elif option == 'Maths Vs Other Subjects':
        fig = plot_math_vs_other_subjects(filtered_df)
        st.pyplot(fig)
    elif option == 'Maths Score By Gender and Preparation':
        fig = plot_math_score_by_gender_and_prep(filtered_df)
        st.pyplot(fig)
    elif option == 'Lunch and Preparation Impact on Writing':
        fig = plot_lunch_and_prep_impact_on_writing(filtered_df)
        st.pyplot(fig)
