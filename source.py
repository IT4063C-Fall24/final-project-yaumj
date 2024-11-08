#!/usr/bin/env python
# coding: utf-8

# # {Breaking the Glass Firewall: Why Women Leave Tech Careers and Why Those Who Stay Don’t Advance}📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# This project explores why women in technology are more likely to experience limited career progression and leave the industry at higher rates than their male counterparts. Understanding these differences in career trajectories between men and women is essential for promoting fairness in the workplace, reducing costs associated with turnover, and improving overall organizational success by retaining diverse talent.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# How do promotion and retention rates for women compare to those for men at similar career stages in the tech industry?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# **Hypothesis:**  
# Women in technology experience lower promotion rates and leave the industry at higher rates compared to their male counterparts, even when they have similar qualifications and experience. This disparity is driven by factors that disproportionately affect women, including a higher likelihood of layoffs and gender-based discrimination. As a result, women are less likely to reach senior positions or remain in the tech industry long-term.
# 

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# **Data Sources**  
# *HackerRank 2018 Developer Survey (Kaggle):* Published in 2018, based on 2017 survey responses.  
# *Pew Research Center 2017 STEM Survey (zip file):* Based on 2017 responses.  
# *NSF's National Center for Science and Engineering Statistics (Web-Scraped Tables):* Spans several years, ending in 2019, with specific data points from 2017 for comparison.
# *From College to Jobs American Community Survey 2019 (U.S. Census Bureau xls files): Detailed information about jobs and salaries broken down by gender and level of education. 
# 
# **Relating the Data**  
# - The datasets can be linked based on the shared timeframe (2017-2019) and gender as a common variable. 
# - Gender will serve as a primary key or part of a composite key for linking. (Non-binary or unknown will be excluded from analysis.)
# - Ages will be limited to the 25 - 64 range to match the American Community Survey data.
# 

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# 
# After joining the data from the various sources, I will generate specific visualizations to confirm or reject my hypothesis.
# 
# **Planned Visualizations to Support the Hypothesis:**
# 1. *Line Chart: Gender and Age Distribution in Technology*  
# A line chart will display the number of employees in the technology sector, separated by gender and age groups. The x-axis will represent the age groups (e.g., 20-25, 26-30, etc.), and the y-axis will show the number of employees. This chart will highlight the drop-off point where women begin to leave the industry earlier than men, helping visualize retention issues.
# 
# 2. *100% Stacked Column Chart: Men vs Women in Different Tech Roles*  
# A 100% stacked column chart will show the proportional representation of men and women across different tech roles (e.g., Junior Developers, Senior Developers, Managers, Executives). Each column will represent a different role, and the stacked columns will show the gender distribution within that role as a percentage. This will provide a clear visual of how underrepresented women are in higher-level positions.
# 
# 3. *Side-by-Side Column Chart: Workplace Concerns for Men vs Women*  
# A side-by-side column chart will compare the key workplace concerns between men and women, such as issues with career progression, work-life balance, pay disparity, and workplace discrimination. Each concern will have two columns—one representing men and one representing women. This will make it easy to see where concerns overlap and where significant differences exist between the genders.

# #### Package Imports

# In[104]:


#import packages
import os # to create subfolder for data organization
from dotenv import load_dotenv
load_dotenv(override=True)

import opendatasets as od
import pandas as pd
import numpy as np
import pyreadstat
import requests
import re # for string manipulation

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from zipfile import ZipFile
from urllib.request import urlretrieve
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from docx import Document


# #### Import Dataset 1: HackerRank Developer Survey Published in 2018 that covered 2017 Questionnaire Responses

# In[105]:


# import dataset from Kaggle using URL 
dataset_url = "https://www.kaggle.com/datasets/hackerrank/developer-survey-2018/data"
od.download(dataset_url, data_dir="./data")


# #### Convert dataset to a pandas dataframe and inspect data for Exploratory Data Analysis (EDA)

# In[106]:


# Define the data directory
data_dir = './data/developer-survey-2018'

# Read each CSV file into its own DataFrame with meaningful names
country_code_df = pd.read_csv(os.path.join(data_dir, 'Country-Code-Mapping.csv'))
dev_survey_codebook_df = pd.read_csv(os.path.join(data_dir, 'HackerRank-Developer-Survey-2018-Codebook.csv'))
dev_survey_numeric_mapping_df = pd.read_csv(os.path.join(data_dir, 'HackerRank-Developer-Survey-2018-Numeric-Mapping.csv'))
dev_survey_numeric_df = pd.read_csv(os.path.join(data_dir, 'HackerRank-Developer-Survey-2018-Numeric.csv'))
dev_survey_values_df = pd.read_csv(os.path.join(data_dir, 'HackerRank-Developer-Survey-2018-Values.csv'))

# Display the first 5 records from each DataFrame
display(country_code_df.head(5))
display(dev_survey_codebook_df.head(5))
display(dev_survey_numeric_mapping_df.head(5))
display(dev_survey_numeric_df.head(5))
display(dev_survey_values_df.head(5))


# #### Exploratory Data Analysis (EDA) Insights:
# - The dataset is global, not local, meaning it does not precisely align with my other datasets.
#     - Information from the country_code_df can be used to identify the country code and country name format so I can filter the other dataframes to only include the United States
# - The datset includes NaN values in several fields including the country. 
#     - Data will need to be explored further to determine the handling of NaN or #NULL! values
#         - For example, if the NaN or NULL value is in gender or country, it cannot be determined using other methods and needs to be excluded from analysis. However,
#         if the NaN or NULL is in the Job Level or Current Rank field, I may be able to populate those fields based on data from similar records.
# - The Numeric and Values datasets have identical infomation, just in different formats.
#     - In the Numeric dataset, all of the values are numerically coded.
#     - In the Values dataset, all of the responses are in plain English.
# - There are outliers in this data as far as what can be directly related to my other datasets and my hypothesis for the purpose of analysis.
#     - Age ranges will need to be limited to match what is available in the datasets from other sources to make "apples to apples" comparisons.

# In[107]:


# Find United States in the country_code_df to use for filtering purposes

# Rename columns
country_code_df.rename(columns={'Value': 'country_code', 'Label': 'country'}, inplace=True)

# Filter for the United States
us_country_code_df = country_code_df[country_code_df['country'] == 'United States']

# Display the resulting DataFrame
display(us_country_code_df)


# #### Filtered dataframe to include only respondents in the United States

# In[108]:


# Filter the DataFrame for United States questionnaire responses
us_dev_survey_numeric_df = dev_survey_numeric_df[dev_survey_numeric_df['CountryNumeric2'] == 167]

# Display the number of records
num_records_numeric = us_dev_survey_numeric_df.shape[0]
display(f"Number of records: {num_records_numeric}")

# Display the resulting DataFrame
display(us_dev_survey_numeric_df.head(5))


# #### Reduce dataframe columns to only those relevant to supporting or disproving my hypothesis
# - fields such as date survey was completed and questions about the HackerRank survey were removed from dataframe for simplification

# In[109]:


# List of relevant columns to keep
columns_to_keep = [
    'RespondentID', 'q2Age', 'q3Gender', 'q10Industry',
    'q8JobLevel', 'q9CurrentRole',
    'q12JobCritPrefTechStack', 'q12JobCritCompMission',
    'q12JobCritCompCulture', 'q12JobCritWorkLifeBal',
    'q12JobCritCompensation', 'q12JobCritProximity',
    'q12JobCritPerks', 'q12JobCritSmartPeopleTeam',
    'q12JobCritImpactwithProduct', 'q12JobCritInterestProblems',
    'q12JobCritFundingandValuation', 'q12JobCritStability',
    'q12JobCritProfGrowth', 'q16HiringManager',
    'q17HirChaNoDiversCandidates', 'q20CandYearExp',
    'q20CandCompScienceDegree', 'q20CandCodingBootcamp',
    'q20CandSkillCert', 'q20CandHackerRankActivity',
    'q20CandOtherCodingCommAct', 'q20CandGithubPersProj',
    'q20CandOpenSourceContrib', 'q20CandHackathonPart',
    'q20CandPrevWorkExp', 'q20CandPrestigeDegree',
    'q20CandLinkInSkills', 'q20CandGithubPersProj2'
]

# Create a new DataFrame with only the selected columns
filtered_us_dev_survey_numeric_df = us_dev_survey_numeric_df[columns_to_keep]

# Display information about the DataFrame 
filtered_us_dev_survey_numeric_df.info()

# Display the resulting DataFrame
display(filtered_us_dev_survey_numeric_df.head(5))


# #### Check for Duplicate Records
# - Some records will be similar, but all records should have a unique RespondentID

# In[110]:


# Check for duplicates in the RespondentID field
duplicates = filtered_us_dev_survey_numeric_df[filtered_us_dev_survey_numeric_df.duplicated('RespondentID', keep=False)]

# Count the number of duplicate RespondentIDs
num_duplicates = duplicates.shape[0]

# Display the duplicates if any
if num_duplicates > 0:
    print(f"Number of duplicate records found: {num_duplicates}")
    display(duplicates[['RespondentID']])
else:
    print("No duplicate records found in RespondentID.")


# #### Filter DataFrame to be Consistent with American Community Survey Parameters for Data Matching
# - Remove ages under 25 years old (coded as q2Age: 1, 2, or 3)
# - Remove ages over 64 years old (coded as q2Age: 8 or 9)
# - Remove non-binary respondents (coded as q3Gender: 3)

# In[111]:


# Summary of counts for each value in q2Age
age_summary = filtered_us_dev_survey_numeric_df['q2Age'].value_counts(dropna=False)
print("Summary of counts for each value in q2Age:")
print(age_summary)

# Summary of counts for each value in q3Gender
gender_summary = filtered_us_dev_survey_numeric_df['q3Gender'].value_counts(dropna=False)
print("\nSummary of counts for each value in q3Gender:")
print(gender_summary)


# In[112]:


# Remove records where q2Age is #NULL!, 1, 2, 3, 8, or 9
filtered_us_dev_survey_numeric_df = filtered_us_dev_survey_numeric_df[
    ~filtered_us_dev_survey_numeric_df['q2Age'].isin(['#NULL!', '1', '2', '3', '8', '9'])
]

# Remove records where q3Gender is #NULL! or 3
filtered_us_dev_survey_numeric_df = filtered_us_dev_survey_numeric_df[
    ~filtered_us_dev_survey_numeric_df['q3Gender'].isin(['#NULL!', '3'])
]

# Display the row count of the refined DataFrame
row_count_after_filtering = filtered_us_dev_survey_numeric_df.shape[0]
print(f"Row count after filtering: {row_count_after_filtering}")

# New summary of counts for each value in q2Age
age_summary = filtered_us_dev_survey_numeric_df['q2Age'].value_counts(dropna=False)
print("New summary of counts for each value in q2Age:")
print(age_summary)

# Summary of counts for each value in q3Gender
gender_summary = filtered_us_dev_survey_numeric_df['q3Gender'].value_counts(dropna=False)
print("\nNew summary of counts for each value in q3Gender:")
print(gender_summary)


# #### Exploratory Data Analysis (EDA)
# - Begin renaming columns
# - The numeric dataframe should consist entirely of int64 data types, yet the majority have an "object" data type instead.
#     - These datatypes will need to be converted for certain types of analysis like a correlation matrix.

# In[113]:


# Rename columns
filtered_us_dev_survey_numeric_df.rename(columns={
    'q2Age': 'Age',
    'q3Gender': 'Gender',
    'q10Industry': 'Industry',
    'q8JobLevel': 'Job Level',
    'q9CurrentRole': 'Current Role'
}, inplace=True)

# Convert all columns to Int64 
filtered_us_dev_survey_numeric_df = filtered_us_dev_survey_numeric_df.apply(pd.to_numeric, errors='coerce').astype('Int64')

# Display the resulting Numeric DataFrame and info
print("Revised Numeric DataFrame (No Splitting, Renamed Columns):")
display(filtered_us_dev_survey_numeric_df.info())
display(filtered_us_dev_survey_numeric_df.head(5))


# #### Data Visualization using Seaborn Correlation Matrix
# **Purpose:** Examine the relationships between key demographic and job criteria variables in the Worker Responses Numeric DataFrame.
# 
# **Insights:**
# - **Age and Job Level Correlation:** The matrix reveals a moderate correlation between *Age* and *Job Level*, indicating that older workers are more likely to hold higher job levels, reflecting common career progression trends.
# - **Weak Correlations:** Most other variables show minimal correlation, suggesting that individual factors like *job preferences* or *industry* alone do not strongly predict other attributes, making them less effective for simple statistical analysis--at least in this specific dataset.
# 
# 
# **INSIGHTS UPDATE:**
# - *After further filtering to remove ages under 25 and over 64, the moderate correlation between age and job level that was seen in the earlier version is no longer apparent.*

# In[114]:


# Select only the desired columns for the correlation matrix
selected_columns = ['Age', 'Gender', 'Industry', 'Job Level', 'Current Role']
filtered_responses = filtered_us_dev_survey_numeric_df[selected_columns]

# Filter out rows with any missing values in the selected columns
filtered_responses = filtered_responses.dropna()

# Compute the correlation matrix on the filtered data with selected columns
correlation_matrix = filtered_responses.corr()

# Plot the correlation heatmap for Worker Responses
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title("Correlation Matrix for HackerRank Survey Responses")
plt.show()


# ## Repeat filtering logic from numeric version of data on the values version of the data
# - Both datasets contain the same records, but one has numeric codes for all responses and the other has plain language values for all responses 
# so the same logic can be used for both dataframes.

# In[115]:


# Filter the DataFrame for CountryNumeric2 = "United States"
us_dev_survey_values_df = dev_survey_values_df[dev_survey_values_df['CountryNumeric2'] == "United States"]

# Display the number of records
num_records_values = us_dev_survey_values_df.shape[0]
display(f"Number of records: {num_records_values}")

# Display the resulting DataFrame
display(us_dev_survey_values_df.head(5))


# In[116]:


# Reduce dataframe columns to only those relevant to supporting or disproving my hypothesis

# List of relevant columns to keep
columns_to_keep = [
    'RespondentID', 'q2Age', 'q3Gender', 'q10Industry',
    'q8JobLevel', 'q9CurrentRole',
    'q12JobCritPrefTechStack', 'q12JobCritCompMission',
    'q12JobCritCompCulture', 'q12JobCritWorkLifeBal',
    'q12JobCritCompensation', 'q12JobCritProximity',
    'q12JobCritPerks', 'q12JobCritSmartPeopleTeam',
    'q12JobCritImpactwithProduct', 'q12JobCritInterestProblems',
    'q12JobCritFundingandValuation', 'q12JobCritStability',
    'q12JobCritProfGrowth', 'q16HiringManager',
    'q17HirChaNoDiversCandidates', 'q20CandYearExp',
    'q20CandCompScienceDegree', 'q20CandCodingBootcamp',
    'q20CandSkillCert', 'q20CandHackerRankActivity',
    'q20CandOtherCodingCommAct', 'q20CandGithubPersProj',
    'q20CandOpenSourceContrib', 'q20CandHackathonPart',
    'q20CandPrevWorkExp', 'q20CandPrestigeDegree',
    'q20CandLinkInSkills', 'q20CandGithubPersProj2'
]

# Create a new DataFrame with only the selected columns
filtered_us_dev_survey_values_df = us_dev_survey_values_df[columns_to_keep]

# Display information about the DataFrame 
filtered_us_dev_survey_values_df.info()

# Display the resulting DataFrame
display(filtered_us_dev_survey_values_df.head(5))


# #### Clean the values dataframe
# - Removed records where gender is missing because they cannot be used to prove/disprove the hypothesis.
# - Filtered out records where gender is non-binary to match the parameters of the US Census Burea's American Community Survey.
# - Filtered out records where respondents were under 25 or over 64 to match the parameters of the US Census Bureau's American Community Survey.
# - Filtered out records where the age is null because both age and gender are necessary to determine job level comparisons.
# 

# In[117]:


# Create a copy to work on and avoid SettingWithCopyWarning
filtered_us_dev_survey_values_df = filtered_us_dev_survey_values_df.copy()

# Drop records where Gender is null
filtered_us_dev_survey_values_df.dropna(subset=['q3Gender'], inplace=True)

# Drop records where Gender is '#NULL!' or 'Non-Binary'
filtered_us_dev_survey_values_df.drop(
    filtered_us_dev_survey_values_df[filtered_us_dev_survey_values_df['q3Gender'].isin(['#NULL!', 'Non-Binary'])].index,
    inplace=True
)

# Filter out respondents who are under 25 or over 64 
filtered_us_dev_survey_values_df = filtered_us_dev_survey_values_df[
    ~filtered_us_dev_survey_values_df['q2Age'].isin(["Under 12 years old", "12 - 18 years old", "18 - 24 years old", "65 - 74 years old", "75 years or older"])
]

# Filter out rows where Age is '#NULL!'
filtered_us_dev_survey_values_df = filtered_us_dev_survey_values_df[
    filtered_us_dev_survey_values_df['q2Age'] != '#NULL!'
]

# Display how many rows remain
print(f"Remaining records after cleaning (SHOULD BE 2942 IF CORRECT): {filtered_us_dev_survey_values_df.shape[0]}")

# New summary of counts for each value in q2Age
age_summary = filtered_us_dev_survey_values_df['q2Age'].value_counts(dropna=False)
print("New summary of counts for each value in Age:")
print(age_summary)

# Summary of counts for each value in q3Gender
gender_summary = filtered_us_dev_survey_values_df['q3Gender'].value_counts(dropna=False)
print("\nNew summary of counts for each value in Gender:")
print(gender_summary)


# #### Clean the values dataframe
# - Rename columns
# - Filtered out records where current role or job level is student because they are not relevant to my hypothesis.
# - Filtered out records where both the Job Level and Current Role were NaN because there is no way to determine values for the field if both are blank.
# 

# In[118]:


# Rename columns
filtered_us_dev_survey_values_df.rename(columns={
    'q2Age': 'Age',
    'q3Gender': 'Gender',
    'q10Industry': 'Industry',
    'q8JobLevel': 'Job Level',
    'q9CurrentRole': 'Current Role'
}, inplace=True)

# Filter out rows where Current Role or Job Level is "Student"
filtered_us_dev_survey_values_df = filtered_us_dev_survey_values_df[
    (filtered_us_dev_survey_values_df['Current Role'] != 'Student') & 
    (filtered_us_dev_survey_values_df['Job Level'] != 'Student')
]

# Filter out rows where both Job Level and Current Role are NaN
filtered_us_dev_survey_values_df = filtered_us_dev_survey_values_df[
    ~ (filtered_us_dev_survey_values_df['Job Level'].isna() & filtered_us_dev_survey_values_df['Current Role'].isna())
]

# Display how many rows remain
print(f"Remaining responses after cleaning: {filtered_us_dev_survey_values_df.shape[0]}")


# #### Data Visualization using Matplotlib Bar Graph
# **Purpose:** Perform a preliminary analysis of the HackerRank Dataset breakdown by age and gender
# **Insights:**
# - **Age Group Representation**: The graph reveals how many workers fall into each age group, highlighting the most common age demographics within the dataset.
# - **Gender Distribution**: By comparing the heights of the bars for different genders, we can see which age groups have higher counts of male and female workers. This highlights trends in workforce composition.
# - **Demographic Changes:** The visualization aims to assess whether there is a change in demographics proportionally from age group to age group, providing insights into how representation shifts across different stages of workforce experience.

# In[119]:


# Group the data by Age Group and Gender, and count occurrences
age_gender_counts = filtered_us_dev_survey_values_df.groupby(['Age', 'Gender']).size().unstack(fill_value=0)

# Plotting
plt.figure(figsize=(10, 6))

# Plot each gender as a separate bar
age_gender_counts.plot(kind='bar', width=0.8, ax=plt.gca())

# Customize the plot
plt.title('Count of Workers by Age Group and Gender')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Gender', loc='upper right')
plt.grid(axis='y')  # Add horizontal grid lines for better readability

plt.show()


# #### Additional Exploratory Data Analyis (EDA) for Data Relevance
# - Determine if records where Job Level = New grad have current roles other than "unemployed" to decide if they should be included in analysis.
# - Look for records where the Job Level is NaN but the Current Role is not NaN.
#     - These records can be used with machine learning classification to populate missing values.

# In[120]:


# Review dataset to determine what data is relevant

# Check Current Role for Job Level "New grad"
new_grad_roles = filtered_us_dev_survey_values_df[filtered_us_dev_survey_values_df['Job Level'] == 'New grad'] \
    .groupby('Current Role').size().reset_index(name='Counts')

# Filter for rows where Job Level is NaN and Current Role is not NaN
nan_job_level_current_role_df = filtered_us_dev_survey_values_df[filtered_us_dev_survey_values_df['Job Level'].isna() & 
    filtered_us_dev_survey_values_df['Current Role'].notna()]

# Display results
print("Current Roles for Job Level 'New grad':")
display(new_grad_roles)

print("Rows where Job Level is NaN and Current Role is not NaN:")
display(nan_job_level_current_role_df[['Job Level', 'Current Role']])


# #### More Exploratory Data Analysis (EDA)
# - Check if there is a dominant job level associated with the Current Role field that could be used to populate empty fields.

# In[121]:


# Group by Current Role and Job Level, and count occurrences
role_level_counts = filtered_us_dev_survey_values_df.groupby(['Current Role', 'Job Level']).size().reset_index(name='Counts')

# Set the option to display all rows for this code cell only
with pd.option_context('display.max_rows', None):
    display(role_level_counts)


# #### Data Visualization using Seaborn Box Plot
# **Purpose:** Demonstrate how gender may impact job levels within specific career roles, in this case, "Development Operations Engineer."
# 
# **Insights:**
# - **Gender-based Distribution:** 
#     - The box plot illustrates the distribution of job levels across genders within the "Development Operations Engineer" role. 
#     - Males occupy a wide range of job levels, predominantly at the Senior Developer level, with a few scattered across roles such as Principal Engineer and Engineering Manager. This broader range suggests more opportunities or representation in higher-level roles for males within this career path. 
#     - In contrast, females are mostly represented at the Level 1 Developer (junior) level, with a smaller range and limited presence in higher job levels. The presence of only a few females, and their clustering at the junior level, may reflect a restricted career progression or underrepresentation at advanced job levels in this role.
# 
# - **Rationale for Machine Learning Approach:** 
#     - Given the clear differences in distribution between genders, a machine learning model like K-Nearest Neighbors (KNN) could be beneficial for predicting job levels. KNN can account for the complex relationships between demographic features (such as gender) and job levels, offering a more refined prediction than simple averages or medians.
#     - This approach would allow the model to better understand and respond to the demographic factors that impact job levels, potentially providing insights into patterns of representation and advancement within this role and the other roles within the dataset.
# 
# This visualization highlights the relevance of employing a machine learning approach that can adapt to demographic differences, rather than relying on single-point estimates like the mean or median for classification.

# In[122]:


# Filter for a specific current role 
specific_role = "Development Operations Engineer"
role_data = filtered_us_dev_survey_values_df[filtered_us_dev_survey_values_df['Current Role'] == specific_role].copy()

# Create a horizontal box plot with gaps between the boxes
plt.figure(figsize=(6, 6))
sns.boxplot(data=role_data, y='Job Level', x='Gender', hue='Gender', palette='Set2', width=0.9, dodge=True)
plt.title(f'Distribution of Job Levels for {specific_role} by Gender')
plt.ylabel('Job Level')
plt.xlabel('Gender')
plt.xticks(rotation=0)
plt.grid(axis='y')  # Add horizontal grid lines for better readability
plt.show()


# #### Use Machine Learning to Populate NaN Job Level Records

# In[123]:


# Use KNearestNeighbors to determine most likely Job Level based on age, gender, and current role

age_mapping = {
    '25 - 34 years old': 4,
    '35 - 44 years old': 5,
    '45 - 54 years old': 6,
    '55 - 64 years old': 7,
}

gender_mapping = {
    'Male': 1,
    'Female': 2,
}

# Prepare the DataFrame for KNN training
train_df = filtered_us_dev_survey_values_df[filtered_us_dev_survey_values_df['Job Level'].notna()].copy()

# Create numeric mappings for Age and Gender for training
train_df.loc[:, 'Age Numeric'] = train_df['Age'].map(age_mapping)
train_df.loc[:, 'Gender Numeric'] = train_df['Gender'].map(gender_mapping)

# Encode Current Role using Label Encoding
le_role = LabelEncoder()
train_df.loc[:, 'Current Role Encoded'] = le_role.fit_transform(train_df['Current Role'])

# Prepare the feature set and target variable for training
X_train = train_df[['Age Numeric', 'Gender Numeric', 'Current Role Encoded']]
y_train = train_df['Job Level']

# Filter the rows with NaN Job Level for predictions
predict_df = filtered_us_dev_survey_values_df[filtered_us_dev_survey_values_df['Job Level'].isna()].copy()
predict_df.loc[:, 'Age Numeric'] = predict_df['Age'].map(age_mapping)
predict_df.loc[:, 'Gender Numeric'] = predict_df['Gender'].map(gender_mapping)
predict_df.loc[:, 'Current Role Encoded'] = le_role.transform(predict_df['Current Role'])

# Create the feature set for predictions
X_predict = predict_df[['Age Numeric', 'Gender Numeric', 'Current Role Encoded']]

# Initialize and fit KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict NaN Job Levels
predicted_levels = knn.predict(X_predict)

# Display the predictions before overwriting the original DataFrame
predict_df['Predicted Job Level'] = predicted_levels
print("Predicted Job Levels for records with NaN Job Level:")
display(predict_df[['Age', 'Gender', 'Current Role', 'Predicted Job Level']])


# In[124]:


#Update Job Level field with predictions and verify changes

# Update the Job Level in the original DataFrame where it was NaN
filtered_us_dev_survey_values_df.loc[filtered_us_dev_survey_values_df['Job Level'].isna(), 'Job Level'] = predicted_levels

# Verify that no records remain with NaN Job Level
remaining_nan_job_levels = filtered_us_dev_survey_values_df['Job Level'].isna().sum()
print(f"Number of records with NaN Job Level after updating: {remaining_nan_job_levels}")

# Display the cleaned DataFrame
display(filtered_us_dev_survey_values_df.head(5))


# #### Import Dataset 2: 2017 Pew Research Center STEM Survey

# In[125]:


# import zip file from Pew Research
file_handle, _ = urlretrieve("https://www.pewresearch.org/wp-content/uploads/sites/20/2019/04/2017-Pew-Research-Center-STEM-survey.zip")
zipfile = ZipFile(file_handle, "r")
zipfile.extractall("./data")
zipfile.close()


# #### Examine contents of .sav file

# In[126]:


file_path = 'data/materials for public release/2017 Pew Research Center STEM survey.sav'

# Read the .sav file into a DataFrame
df, meta = pyreadstat.read_sav(file_path)

# Display basic information about the DataFrame
print(df.info())

# Display the first few rows of the DataFrame
print(df.head())

print(df.tail())


# #### Read the .docx file
# - Read the Pew Research Center files associated with the .sav file and convert them into .txt files to understand the codes used.

# In[127]:


# Load the Questionnaire document
questionnaire_path = './data/materials for public release/Questionnaire - 2017 Pew Research Center STEM Survey.docx'
questionnaire_doc = Document(questionnaire_path)

# Extract text from the Questionnaire document
prc_questions = []  
for paragraph in questionnaire_doc.paragraphs:
    prc_questions.append(paragraph.text)

# Join all paragraphs into a single string for the questionnaire
prc_questions_text = "\n".join(prc_questions)

# Define the path for saving the questionnaire .txt file
questionnaire_txt_file_path = './data/materials for public release/prc_questionnaire.txt'

# Save the full text to a .txt file
with open(questionnaire_txt_file_path, 'w', encoding='utf-8') as f:
    f.write(prc_questions_text)

print(f"The questionnaire has been extracted and saved to '{questionnaire_txt_file_path}'.")

# Preview the first few lines of the extracted questionnaire
print("Preview of the extracted questionnaire:")
print(prc_questions_text[:400])

# Load the Codebook document
codebook_path = './data/materials for public release/Codebook - 2017 Pew Research Center STEM Survey.docx'
codebook_doc = Document(codebook_path)

# Extract text from the Codebook document
prc_codebook = []  
for paragraph in codebook_doc.paragraphs:
    prc_codebook.append(paragraph.text)

# Join all paragraphs into a single string for the codebook
prc_codebook_text = "\n".join(prc_codebook)

# Define the path for saving the codebook .txt file
codebook_txt_file_path = './data/materials for public release/prc_codebook.txt'

# Save the full text to a .txt file
with open(codebook_txt_file_path, 'w', encoding='utf-8') as f:
    f.write(prc_codebook_text)

print(f"The codebook has been extracted and saved to '{codebook_txt_file_path}'.")

# Preview the first few lines of the extracted codebook
print("Preview of the extracted codebook:")
print(prc_codebook_text[:400])


# #### Display All Column Names in Dataframe
# - Reading the column names with the new context of the Questionnaire and Codebook file will help to determine which columns are needed for analysis

# In[128]:


# Display all column names in the DataFrame
print("Column names in the Pew Research dataset:")
for col in df.columns:
    print(col)


# #### Convert CaseID columns from float to int64

# In[129]:


# Convert 'CaseID' to int64
df['CaseID'] = df['CaseID'].astype('int64')

# Confirm the change
print("\nData types after conversion:")
print(df.dtypes)


# #### Exploratory Data Analysis (EDA)
# - Search for null values or refused responses in fields required for analysis.

# In[130]:


# Summary of counts for each value in WORK_1
employed_full_time = df['WORK_1'].value_counts(dropna=False)
print("Summary of counts for each value in WORK_1:")
print(employed_full_time)

# Summary of counts for each value in WORK_2
employed_part_time = df['WORK_2'].value_counts(dropna=False)
print("\nSummary of counts for each value in WORK_2:")
print(employed_part_time)

# Summary of counts for each value in WORK_3
self_employed_full_time = df['WORK_3'].value_counts(dropna=False)
print("\nSummary of counts for each value in WORK_3:")
print(self_employed_full_time)

# Summary of counts for each value in WORK_4
self_employed_part_time = df['WORK_4'].value_counts(dropna=False)
print("\nSummary of counts for each value in WORK_4:")
print(self_employed_part_time)

# Summary of counts for each value in EMPLOYED
employment_status = df['EMPLOYED'].value_counts(dropna=False)
print("\nSummary of counts for each value in EMPLOYED:")
print(employment_status)

# Summary of counts for each value in FULLPART
employment_full_part = df['FULLPART'].value_counts(dropna=False)
print("\nSummary of counts for each value in FULLPART:")
print(employment_full_part)

# Summary of counts for each value in SELFEMPLOYED
self_employment = df['SELFEMPLOYED'].value_counts(dropna=False)
print("\nSummary of counts for each value in SELFEMPLOYED:")
print(self_employment)

# Summary of counts for each value in ppagecat
prc_age = df['ppagecat'].value_counts(dropna=False)
print("\nSummary of counts for each value in ppagecat:")
print(prc_age)

# Summary of counts for each value in PPGENDER
prc_gender = df['PPGENDER'].value_counts(dropna=False)
print("\nSummary of counts for each value in PPGENDER:")
print(prc_gender)


# In[131]:


# Convert specified columns to int64
columns_to_convert = [
    'WORK_1', 'WORK_2', 'WORK_3', 'WORK_4', 
    'EMPLOYED', 'FULLPART', 'SELFEMPLOYED'
]

# Convert to int64
for column in columns_to_convert:
    df[column] = df[column].astype('int64')

# Drop records with 9 for WORK_1, WORK_2, WORK_3, WORK_4, EMPLOYED, FULLPART, SELFEMPLOYED
df = df[~df[columns_to_convert].isin([9]).any(axis=1)]

# For ppagecat, drop 1, 6, and 7
df = df[~df['ppagecat'].isin([1, 6, 7])]

# Confirming the changes
print(f"Remaining records after filtering: {df.shape[0]}")
print("\nUpdated DataFrame info:")
print(df.info())


# #### Refine DataFrame for Pew Research Center STEM Survey
# - Eliminate fields not needed for hypothesis
# - Begin renaming fields

# In[132]:


# List of columns to exclude
columns_to_exclude = [
    'SCH2a', 'SCH2b', 'SCH3a', 'SCH3b', 'SCH3c', 'SCH3d',
    'SCH4', 'SCH5a', 'SCH5b', 'SCH5c', 'SCH6a', 'SCH6b',
    'SCH6c', 'SCH6d', 'SCH6e', 'SCH6f', 'SCH6g', 'SCH6h',
    'SCH7', 'SCH8a', 'SCH8b', 'SCH9a', 'SCH9b', 'SCH10_flag',
    'SCH10A_1', 'SCH10A_2', 'SCH10A_3', 'SCH10A_4', 'SCH10A_5',
    'SCH10A_6', 'SCH10A_Refused', 'SCH10B_1', 'SCH10B_2',
    'SCH10B_3', 'SCH10B_4', 'SCH10B_5', 'SCH10B_6', 'SCH10B_Refused',
    'STEMJOBa', 'STEMJOBb', 'STEMJOBc', 'STEMJOBd', 'STEMJOBe',
    'STEMJOBf', 'STEMJOBg', 'STEMJOBh', 'REASON2a', 'REASON2b',
    'REASON2c', 'REASON2d', 'REASON2e', 'REASON2f', 'REASON2g',
    'ETHNJOB2', 'ETHNJOB2_OE1_col', 'ETHNJOB2_OE2_col', 
    'ETHNJOB2_OE3_col', 'STEM_DEGREE', 'RACE_col', 'SCICOUR2_t',
    'MATHCOUR2_t', 'PPT017_t', 'PPT18OV_t', 'PPHHSIZE_t', 
    'ETHN1', 'ETHN2a', 'ETHN2b', 'ETHN2c', 'ETHN2d', 
    'ETHN3a', 'ETHN3b', 'ETHN3c', 'ETHN3d', 'ETHN4', 
    'ETHN5', 'ETHN6_a', 'ETHN6_b', 'ETHN6_c', 'ETHN6_d',
    'ETHN6_Refused', 'ETHNDISC_a', 'ETHNDISC_b', 'ETHNDISC_c',
    'ETHNDISC_d', 'ETHNDISC_e', 'ETHNDISC_f', 'ETHNDISC_g',
    'ETHNDISC_h', 'ETHNDISC_i', 'ETHNDISC_Refused', 'FAMSTEM1',
    'FAMSTEM2_1', 'FAMSTEM2_2', 'FAMSTEM2_Refused', 'INTEREST1',
    'TECH4', 'TECH5', 'TECH6', 'ETHNJOB1', 'PARTY', 'PARTYLN',
    'IDEO', 'PUBLIC', 'DOV_FORM', 'PPHHHEAD', 'HH_INCOME_col',
    'PPMARIT', 'PPMSACAT', 'SCH1_OE1_col', 'SCH1_OE2_col',
    'SCH1_OE3_col', 'SCH7_OE1_col', 'INTEREST2_OE', 'INTEREST3_OE1_col',
    'INTEREST3_OE2_col', 'INTEREST3_OE3_col'
]

# Create a new DataFrame excluding the specified columns
pew_research_numeric = df.loc[:, ~df.columns.isin(columns_to_exclude)].copy()

# Create a dictionary for renaming specified columns
columns_to_rename = {
    'WORK_1': 'Employed Full-Time by Company',
    'WORK_2': 'Employed Part-Time by Company',
    'WORK_3': 'Self-Employed Full-Time',
    'WORK_4': 'Self-Employed Part-Time',
    'EMPLOYED': 'Employment Status',
    'FULLPART': 'Employee Type',
    'SELFEMPLOYED': 'Self-Employment Status',
    'OCCUPATION_col': 'Occupation',
    'INDUSTRY_col': 'Industry',
    'TEACHSTEM': 'STEM Teacher Y_N',
    'WORKTYPE_FINAL': 'STEM Worker Y_N',
    'EDUC4CAT': 'Education Level Categorical',
    'RECONA_col': 'Computer Work Y_N',
    'RECONB_col': 'Engineer Y_N',
    'RECONC_col': 'Science Worker Type',
    'STEM_DEGREE': 'STEM Degree Y_N',
    'JOBVALU1_1': 'Job Choice - High Pay',
    'JOBVALU1_2': 'Job Choice - Work-Life Balance',
    'JOBVALU1_3': 'Job Choice - Advancement Opportunities',
    'JOBVALU1_4': 'Job Choice - Contribution to Society',
    'JOBVALU1_5': 'Job Choice - Respect of Others',
    'JOBVALU1_6': 'Job Choice - Helping Others',
    'JOBVALU1_7': 'Job Choice - Welcoming Environment',
    'JOBVALU1_8': 'Job Choice - None of the Above',
    'JOBVALU2': 'Job Choice - Most Important',
    'AHEADa': 'Employment Advancement - Assertiveness',
    'AHEADb': 'Employment Advancement - Socializing with Co-Workers',
    'AHEADc': 'Employment Advancement - Point Out Workplace Problems',
    'AHEADd': 'Employment Advancement - Workplace Mentor',
    'AHEADe': 'Employment Advancement - Sharing Personal Life',
    'AHEADf': 'Employment Advancement - Working Harder',
    'AHEADg': 'Employment Advancement - Point Out Personal Accomplishments',
    'TALENT': 'Employment Advancement - Natural Ability',
    'PROVE': 'Workplace Respect - Need to Prove Oneself',
    'RESPECTA': 'Workplace Respect - Valued by Supervisor',
    'RESPECTB': 'Workplace Respect - Valued by Co-Workers'
}

# Rename the columns as specified
pew_research_numeric.rename(columns=columns_to_rename, inplace=True)

# Display the new DataFrame info to confirm changes
print("Column names in the Pew Research Numeric dataset:")
for col in pew_research_numeric.columns:
    print(col)


# ## Data after this point is still in the process of being analyzed, cleaned, and transformed.
# - Due to time constraints for the submission of Checkpoint 2, the work on these datasets is not complete. However, they will continue to be worked with for the final project.
# - The work prior to this point should include all requirements for Checkpoint 2: Exploratory Data Analysis & Visualization aside from one of the visualizations, which is included after Dataset 4 was imported and partially processed.

# #### Import Dataset 3: National Center for Science and Engineering Statistics (NCSES)
# - Includes demographic breakdown of STEM participation in the workforce from 1993 - 2019
# - Data was compiled by the NCSES from the U.S. Census Bureau, American Community Survey, National Center for Science and Engineering Statistics, and more
# - For the full list of compiled sources: https://ncses.nsf.gov/pubs/nsb20212/data#source-block 

# In[133]:


# scrape HTML file to extract tables
os.makedirs("data/ncses", exist_ok=True) # create new subfolder for tables scraped from website

url = "https://ncses.nsf.gov/pubs/nsb20212/participation-of-demographic-groups-in-stem"
page = requests.get(url)
soup = BeautifulSoup(page.content, "html.parser") # parse the html as a string

tables = soup.find_all("table")


# loop through each table to extract the title and data
for i in range(len(tables)):
    table = tables[i]

    # Extract the table's title attribute and clean it up for a filename
    title = table.get('title')  # Find the title attribute in the table

    if title:
        title = title.strip()  # Use the title if it exists
    else:
        title = f'table_{i}'  # Generic name if the table does not have a title
    
    # Clean title: replace spaces with hyphens and remove special characters except apostrophes
    title = title.replace("'", "")  # Remove apostrophes
    title = re.sub(r"[^\w\s]", "-", title)  # Replace special characters with hyphens
    title = title.replace(" ", "-")  # Replace spaces with hyphens

    # Truncate the title if it exceeds 50 characters for filename compatibility
    if len(title) > 50:
        title = title[:50]

    file_name = f"data/ncses/{title}.csv" # generate an empty csv for each table

    # convert each table to a pandas df
    df = pd.read_html(str(table))[0]

    # save the pandas df as a csv
    df.to_csv(file_name, index=False)


# **Import Additional Resources From National Center for Science and Engineering Statistics (NCSES)**
# - The Report titled *"The STEM Labor Force of Today: Scientists, Engineers, and Skilled Technical Workers"* spans several pages and has supplemental tables that are not included on any of the pages. 

# In[134]:


# import data-tables zip file from NCSES
file_handle, _ = urlretrieve("https://ncses.nsf.gov/pubs/nsb20212/assets/nsb20212-report-tables-excels.zip")
zipfile = ZipFile(file_handle, "r")
zipfile.extractall("./data/ncses")
zipfile.close()

# import data-figures zip file from NCSES
file_handle, _ = urlretrieve("https://ncses.nsf.gov/pubs/nsb20212/assets/nsb20212-report-figures-excels.zip")
zipfile = ZipFile(file_handle, "r")
zipfile.extractall("./data/ncses")
zipfile.close()

# import supplemental tables zip file from NCSES
file_handle, _ = urlretrieve("https://ncses.nsf.gov/pubs/nsb20212/assets/supplemental-tables/nsb20212-supplemental-tables-figures-tables-excels.zip")
zipfile = ZipFile(file_handle, "r")
zipfile.extractall("./data/ncses")
zipfile.close()


# **Convert relevant xlsx files into pandas dataframes**

# In[135]:


# Define the path to the file: Table LBR-7 - Women with a bachelor's degree or above, by broad occupational group and highest degree: 1993, 2003, 2019
file_path = './data/ncses/nsb20212-tablbr-007.xlsx'

# Define start rows for each section based on the Excel file structure
start_row_degrees = 9  # Adjust based on where Degree Focus data starts
start_row_occupations = 6  # Adjust based on where Occupational Group data starts
num_rows_degrees = 2  # Number of rows for the degree section
num_rows_occupations = 2  # Number of rows for the occupation section

# Load the Degree Focus data
degree_focus_df = pd.read_excel(
    file_path,
    skiprows=start_row_degrees,
    nrows=num_rows_degrees,
    names=["Category", "1993 - Count", "2003 - Count", "2019 - Count", "1993 - Percent", "2003 - Percent", "2019 - Percent"]
)

# Load the Occupational Group data
occupational_group_df = pd.read_excel(
    file_path,
    skiprows=start_row_occupations,
    nrows=num_rows_occupations,
    names=["Category", "1993 - Count", "2003 - Count", "2019 - Count", "1993 - Percent", "2003 - Percent", "2019 - Percent"]
)

# Add a column to indicate whether the data is for degrees or occupations
degree_focus_df['Type'] = 'Degree'
occupational_group_df['Type'] = 'Occupation'

# Convert "Thousands" columns to actual numbers by multiplying by 1,000
for col in ["1993 - Count", "2003 - Count", "2019 - Count"]:
    degree_focus_df[col] = degree_focus_df[col] * 1000
    occupational_group_df[col] = occupational_group_df[col] * 1000

# Concatenate both DataFrames to have one unified DataFrame for comparison
ncses_women_science_and_engineering_ed_vs_employment_df = pd.concat([degree_focus_df, occupational_group_df], ignore_index=True)

# Display the resulting DataFrame
print("Combined DataFrame for Degrees and Occupations:")
display(ncses_women_science_and_engineering_ed_vs_employment_df)


# In[136]:


# Define the path to the file: Figure LBR-21 - Women with a bachelor's degree or higher in S&E and S&E-related occupations: Selected years, 1993–2019
file_path = './data/ncses/nsb20212-figlbr-021.xlsx'

# Load the data, skipping rows based on the Excel file structure
start_row = 3  

# Load the data with descriptive column names
women_s_e_degree_trends_df = pd.read_excel(
    file_path,
    skiprows=start_row,
    names=[
        "Year", "Computer and Mathematical Scientists (%)", "Biological, Agricultural, and Environmental Life Scientists (%)",
        "Physical Scientists (%)", "Social Scientists (%)", "Engineers (%)", "All S&E-related Workers (%)"
    ]
)

# Display the resulting DataFrame to ensure it loaded correctly
print("S&E Degree Trends for Women DataFrame (with % values):")
display(women_s_e_degree_trends_df)


# In[137]:


# Define the path to the file: Figure LBR-27 - Median annual salaries of full-time workers with highest degrees in S&E or S&E-related fields, by sex: Selected years, 1995, 2003, and 2019
file_path = './data/ncses/nsb20212-figlbr-027.xlsx'

# Load the data, skipping rows based on the Excel file structure
start_row = 3  

# Load the data into a DataFrame
median_salary_by_gender_df = pd.read_excel(
    file_path,
    skiprows=start_row,
    names=["Degree Field", "Gender", "1995 Salary", "2003 Salary", "2019 Salary"]
)

# Forward-fill the "Degree Field" column to handle merged cells properly
median_salary_by_gender_df["Degree Field"] = median_salary_by_gender_df["Degree Field"].ffill()

# Display the final structured DataFrame
print("Restructured Salary DataFrame:")
display(median_salary_by_gender_df)


# #### Interactive Line Graph using Plotly Graph Objects: Median Salary by Gender with Percentage Increase Over Time
# 
# **Purpose:** To analyze and demonstrate the disparity in median salaries between genders over time for workers with S&E degrees, highlighting how salary increases have influenced the wage gap.
# 
# **Insights:**
# - **Initial Salary Disparity:**  
#     - In 1995, the median salary for males was $69,000, while females earned $47,000, creating an initial wage gap of 31.88%.
# - **Unequal Salary Increases:**  
#     - Over the next 8 years (first measured interval), the male median salary increased by 20.3%, whereas the female median salary increased by only 17.0%. This disproportionate increase further widened the gap between male and female earnings.
# - **Persistent Disparity by 2019:**  
#     - At the next interval, 16 years later, both genders saw an identical salary increase of 3.6%. However, the male median salary reached $86,000, while the female median salary only rose to $57,000. This resulted in an even larger wage gap, with a final disparity of 33.72%.
# 
# This interactive visualization highlights the long-standing and widening disparity in median salaries between genders. The unequal salary increases at crucial intervals have exacerbated the wage gap, emphasizing how systemic disparities in salary growth prevent women from closing the gap in career fields that require science and engineering degrees.

# In[138]:


# Filter only rows where Degree Field is "S&E"
se_salary_df = median_salary_by_gender_df[median_salary_by_gender_df["Degree Field"] == "S&E"].copy()

# Reshape the DataFrame from wide to long format for plotting
salary_melted_df = se_salary_df.melt(
    id_vars=["Gender"],
    value_vars=["1995 Salary", "2003 Salary", "2019 Salary"],
    var_name="Year",
    value_name="Median Salary"
)

# Convert 'Year' to numeric format
salary_melted_df["Year"] = salary_melted_df["Year"].str.extract('(\d{4})').astype(int)

# Sort values to ensure correct plotting order
salary_melted_df = salary_melted_df.sort_values(["Gender", "Year"])

# Calculate the percentage increase for each gender
salary_melted_df["Percentage Increase"] = salary_melted_df.groupby("Gender")["Median Salary"].pct_change() * 100

# Create the line chart with percentage increase annotations
fig = go.Figure()

# Add Male Salary Line
fig.add_trace(go.Scatter(
    x=salary_melted_df[salary_melted_df["Gender"] == "Male"]["Year"],
    y=salary_melted_df[salary_melted_df["Gender"] == "Male"]["Median Salary"],
    mode="lines+markers+text",
    name="Male Salary",
    line=dict(color="red"),
    marker=dict(size=8),
    text=[f"{perc:.1f}% Increase" if not pd.isna(perc) else ""
          for perc in salary_melted_df[salary_melted_df["Gender"] == "Male"]["Percentage Increase"]],
    textposition="top left",
    hovertemplate="<b>Year:</b> %{x}<br><b>Salary:</b> %{y:$,}<br>%{text}<extra></extra>"
))

# Add Female Salary Line
fig.add_trace(go.Scatter(
    x=salary_melted_df[salary_melted_df["Gender"] == "Female"]["Year"],
    y=salary_melted_df[salary_melted_df["Gender"] == "Female"]["Median Salary"],
    mode="lines+markers+text",
    name="Female Salary",
    line=dict(color="blue"),
    marker=dict(size=8),
    text=[f"{perc:.1f}% Increase" if not pd.isna(perc) else ""
          for perc in salary_melted_df[salary_melted_df["Gender"] == "Female"]["Percentage Increase"]],
    textposition="top left",
    hovertemplate="<b>Year:</b> %{x}<br><b>Salary:</b> %{y:$,}<br>%{text}<extra></extra>"
))

# Customize layout
fig.update_layout(
    title="Median Salary by Gender with Percentage Increase Over Time (S&E Degrees)",
    xaxis_title="Year",
    yaxis_title="Median Salary ($)",
    yaxis_tickprefix="$",
    yaxis_range=[40000, 105000],  # Adjusted y-axis range to give more room for top labels
    legend=dict(x=0.1, y=1.1, orientation="h"),
)

# Show the figure
fig.show()


# In[139]:


# Define the path to the file: Table SLBR-30 - Number and median salary of full-time workers with highest degree in S&E field, by sex and occupation: 2019
file_path = './data/ncses/nsb20212-tabslbr-030.xlsx'

# Define specific row indices for the occupations 
selected_rows = [7, 28, 40, 53, 66, 85, 112] 

# Load the entire file first, then filter for the selected rows
data = pd.read_excel(file_path, header=None)

# Select the specified rows and reset the index for male and female data
slbr30_male_df = data.iloc[selected_rows, [0, 5, 6]].copy()
slbr30_female_df = data.iloc[selected_rows, [0, 3, 4]].copy()

# Rename columns for both DataFrames
slbr30_male_df.columns = ["Occupation", "Total Workers", "Median Salary"]
slbr30_female_df.columns = ["Occupation", "Total Workers", "Median Salary"]

# Convert Thousands to actual counts
slbr30_male_df["Total Workers"] = slbr30_male_df["Total Workers"] * 1000
slbr30_female_df["Total Workers"] = slbr30_female_df["Total Workers"] * 1000

# Add Gender columns
slbr30_male_df["Gender"] = "Male"
slbr30_female_df["Gender"] = "Female"

# Combine the DataFrames
employment_count_and_salary_by_occupation_and_gender_df = pd.concat([slbr30_male_df, slbr30_female_df], ignore_index=True)

# Display the resulting DataFrame
print("Combined DataFrame for Selected Occupations and Salaries:")
display(employment_count_and_salary_by_occupation_and_gender_df)


# In[140]:


# Define the path to the file: Table SLBR-32 - Employed S&E highest degree holders, by sex, race or ethnicity, field of highest degree, and broad occupational category: 2019
file_path = './data/ncses/nsb20212-tabslbr-032.xlsx'

# Define start rows and number of rows for each gender based on the Excel file structure
start_row_female = 7  
start_row_male = 14   
num_rows_female = 5   
num_rows_male = 5     

# Load the Female data
female_df = pd.read_excel(
    file_path,
    skiprows=start_row_female,
    nrows=num_rows_female,
    names=[
        "Degree Type", "Total S&E Occupations (%)", 
        "S&E Occupations - Degree Related (%)", "S&E Occupations - Not Related to Degree (%)",
        "S&E-Related Occupations (%)", "Non-S&E Occupations (%)"
    ]
)

# Add Gender column for Female
female_df['Gender'] = 'Female'

# Load the Male data
male_df = pd.read_excel(
    file_path,
    skiprows=start_row_male,
    nrows=num_rows_male,
    names=[
        "Degree Type", "Total S&E Occupations (%)", 
        "S&E Occupations - Degree Related (%)", "S&E Occupations - Not Related to Degree (%)",
        "S&E-Related Occupations (%)", "Non-S&E Occupations (%)"
    ]
)

# Add Gender column for Male
male_df['Gender'] = 'Male'

# Combine the two DataFrames
se_degree_vs_occupation_by_gender_df = pd.concat([female_df, male_df], ignore_index=True)

# Display the final structured DataFrame
print("Occupation of S&E Degree Holders DataFrame with Gender:")
display(se_degree_vs_occupation_by_gender_df)


# #### Import Dataset 4: United States Census Bureau
# - From College to Jobs: American Community Survey 2019

# In[141]:


# Define the directory to store the downloaded files
download_dir = './data/from-college-to-jobs-acs-2019'
os.makedirs(download_dir, exist_ok=True)

# Dictionary mapping descriptive names to URLs
urls = {
    "job-by-bach-degree-field-all-ed-levels.xlsx": "https://www2.census.gov/programs-surveys/demo/tables/industry-occupation/2019/table1.xlsx",
    "job-by-bach-degree-field-bach-degree.xlsx": "https://www2.census.gov/programs-surveys/demo/tables/industry-occupation/2019/table2.xlsx",
    "job-by-bach-degree-field-grad-degree.xlsx": "https://www2.census.gov/programs-surveys/demo/tables/industry-occupation/2019/table3.xlsx",
    "med-earn-by-degree-level-field-and-occupation.xlsx": "https://www2.census.gov/programs-surveys/demo/tables/industry-occupation/2019/table4.xlsx"
}

# Download each file with the specified name
for file_name, url in urls.items():
    file_path = os.path.join(download_dir, file_name)
    response = requests.get(url)
    
    # Save the file
    with open(file_path, 'wb') as file:
        file.write(response.content)
    
    # Print confirmation message
    print(f"Downloaded {file_name} to {file_path}")


# #### Exploratory Data Analyis on Census Tables
# - The excel spreadsheet includes explanations of the data that spans several rows at the top and bottom of the data
#     - This means that rows need to be specifically extracted or excluded 
# - The census tables have headings and subheadings in the columns
#     - This means the columns need to be renamed to reflect the discriptive header rather than the repetitive subheading
# - The table contents are divided by row with a heading indicating the categorization in the rows following the row's subheading (e.g. male, female, white, hispanic)
#     - This means specific rows need to be extracted to ensure the data is only what is needed for analysis

# **Extract the data for the men from the first Excel file to test processing**

# In[142]:


# Define the path to the file
file_path = './data/from-college-to-jobs-acs-2019/job-by-bach-degree-field-all-ed-levels.xlsx'

# Specify only the required columns by their indices
selected_columns = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

# Define the starting row and the number of rows to read
start_row = 27
num_rows = 20  # Number of rows to read after the starting row

# Load the data with only the selected columns and limit the rows
df_men_all_ed_levels = pd.read_excel(file_path, skiprows=start_row, nrows=num_rows, usecols=selected_columns)

# Define descriptive column names for the selected columns
columns = [
    "Occupation", "FoD-Computers_Math_Stats", "FoD-Engineering", "FoD-Physical_Sciences",
    "FoD-Biological_Environmental_Agricultural_Sciences", "FoD-Psychology",
    "FoD-Social_Sciences", "FoD-Multidiscipline", "FoD-Science_and_Engineering_Related",
    "FoD-Business", "FoD-Education", "FoD-Literature_and_Languages",
    "FoD-Liberal_Arts_and_History", "FoD-Visual_and_Performing_Arts",
    "FoD-Communications", "FoD-Other_EG_Criminal_Justice_or_Social_Work"
]

# Apply the custom column names
df_men_all_ed_levels.columns = columns

# Remove any leading dots or whitespace from the 'Occupation' column
df_men_all_ed_levels['Occupation'] = df_men_all_ed_levels['Occupation'].str.lstrip(". ")

# Add a gender column
df_men_all_ed_levels['Gender'] = 'Male'

# Preview to check the data
print(df_men_all_ed_levels.head())
print(df_men_all_ed_levels.tail())


# #### Exploratory Data Analyis (EDA): Check the Data Types

# In[143]:


df_men_all_ed_levels.info()


# **Convert columns to correct data types (float64 to int64)**

# In[144]:


# Select numeric columns that need conversion to int64
numeric_columns = [
    "FoD-Computers_Math_Stats", "FoD-Engineering", "FoD-Physical_Sciences", 
    "FoD-Biological_Environmental_Agricultural_Sciences", "FoD-Psychology",
    "FoD-Social_Sciences", "FoD-Multidiscipline", "FoD-Science_and_Engineering_Related",
    "FoD-Business", "FoD-Education", "FoD-Liberal_Arts_and_History", 
    "FoD-Visual_and_Performing_Arts", "FoD-Communications", 
    "FoD-Other_EG_Criminal_Justice_or_Social_Work"
]

# Convert selected numeric columns to int64
df_men_all_ed_levels[numeric_columns] = df_men_all_ed_levels[numeric_columns].astype('int64')

# Display updated data types to confirm changes
print("\nData types after conversion:")
print(df_men_all_ed_levels.info())


# **Repeat the process for the women's data in the same file**

# In[145]:


# Define the path to the file
file_path = './data/from-college-to-jobs-acs-2019/job-by-bach-degree-field-all-ed-levels.xlsx'

# Specify the selected columns by their indices (same as for men)
selected_columns = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

# Define the header row and the number of rows for women
header_row = 49
num_rows = 20  # The same as for men

# Load the data for women with the specified columns, skipping initial rows and using the right number of rows
df_women_all_ed_levels = pd.read_excel(
    file_path, skiprows=header_row, nrows=num_rows, usecols=selected_columns
)

# Define descriptive column names
columns = [
    "Occupation", "FoD-Computers_Math_Stats", "FoD-Engineering", "FoD-Physical_Sciences",
    "FoD-Biological_Environmental_Agricultural_Sciences", "FoD-Psychology",
    "FoD-Social_Sciences", "FoD-Multidiscipline", "FoD-Science_and_Engineering_Related",
    "FoD-Business", "FoD-Education", "FoD-Literature_and_Languages",
    "FoD-Liberal_Arts_and_History", "FoD-Visual_and_Performing_Arts",
    "FoD-Communications", "FoD-Other_EG_Criminal_Justice_or_Social_Work"
]

# Apply the column names
df_women_all_ed_levels.columns = columns

# Remove any leading dots or whitespace from the 'Occupation' column
df_women_all_ed_levels['Occupation'] = df_women_all_ed_levels['Occupation'].str.lstrip(". ")

# Add a gender column with "Female" as the value
df_women_all_ed_levels['Gender'] = 'Female'

# Convert numeric columns to int64 to ensure consistency with the men’s DataFrame
numeric_columns = [
    "FoD-Computers_Math_Stats", "FoD-Engineering", "FoD-Physical_Sciences",
    "FoD-Biological_Environmental_Agricultural_Sciences", "FoD-Psychology",
    "FoD-Social_Sciences", "FoD-Multidiscipline", "FoD-Science_and_Engineering_Related",
    "FoD-Business", "FoD-Education", "FoD-Literature_and_Languages",
    "FoD-Liberal_Arts_and_History", "FoD-Visual_and_Performing_Arts",
    "FoD-Communications", "FoD-Other_EG_Criminal_Justice_or_Social_Work"
]
df_women_all_ed_levels[numeric_columns] = df_women_all_ed_levels[numeric_columns].astype('int64')

# Preview to confirm correct data import and types
print(df_women_all_ed_levels.head())
print(df_women_all_ed_levels.tail())
print(df_women_all_ed_levels.info())


# #### Generate a Stacked Bar Chart using Matplotlib to compare the dataframes
# **Purpose:** Analyze the distribution of fields of degree by occupation across genders, focusing on the highest concentrations of careers and how degree backgrounds vary between men and women.
# 
# **Insights:**
# 
# - **Career Concentration by Gender:**
#   - For **men**, the occupation with the highest representation is **Managers (Non-STEM)**, highlighting a strong presence in managerial roles outside of science and engineering fields.
#   - For **women**, **Education** has the highest concentration, with a significant representation also seen in **Healthcare** roles, underscoring these fields as primary career paths for women in the dataset.
# 
# - **Gender Disparities in STEM Roles:** 
#   - **Men** are much more likely to be employed as **Computer Workers** and **Engineers**, even when they hold degrees seemingly unrelated to these fields, such as **Liberal Arts and History**. This suggests a broader acceptance or hiring trend for men in technical roles, regardless of their field of study.
#   - **Women** have comparatively lower representation in technical fields like Computer Work and Engineering, which may indicate potential barriers to entry or differing career choices despite educational background.
# 
# - **Healthcare Sector Dominance by Women:** Women overwhelmingly dominate roles in **Healthcare**, aligning with the degree distributions in fields like **Biological, Environmental, and Agricultural Sciences**. This suggests a continued trend of women pursuing healthcare-related careers.
# 
# - **Cross-Disciplinary Employment Trends:** The visualizations reveal that while men frequently cross into technical roles with non-STEM degrees, women tend to stay within fields closely aligned with their degree, such as **Education** and **Social Services**.
# 

# In[146]:


# Define fields of degree columns
fields = df_men_all_ed_levels.columns[1:-1]  # Excluding 'Occupation' and 'Gender'

# Set up the figure with two subplots
fig, (ax_men, ax_women) = plt.subplots(1, 2, figsize=(18, 10), sharey=True)

# Plot for Men
bottom = None
for field in fields:
    ax_men.bar(
        df_men_all_ed_levels['Occupation'], df_men_all_ed_levels[field],
        label=field, bottom=bottom
    )
    bottom = df_men_all_ed_levels[fields[:list(fields).index(field)+1]].sum(axis=1)
ax_men.set_title("Field of Degree Distribution by Occupation (Men)")
ax_men.set_xlabel("Occupation")
ax_men.set_ylabel("Count")
ax_men.legend(title="Field of Degree", bbox_to_anchor=(1.05, 1), loc='upper left')
ax_men.set_xticks(range(len(df_men_all_ed_levels['Occupation'])))
ax_men.set_xticklabels(df_men_all_ed_levels['Occupation'], rotation=45, ha='right')

# Plot for Women
bottom = None
for field in fields:
    ax_women.bar(
        df_women_all_ed_levels['Occupation'], df_women_all_ed_levels[field],
        label=field, bottom=bottom
    )
    bottom = df_women_all_ed_levels[fields[:list(fields).index(field)+1]].sum(axis=1)
ax_women.set_title("Field of Degree Distribution by Occupation (Women)")
ax_women.set_xlabel("Occupation")
ax_women.set_xticks(range(len(df_women_all_ed_levels['Occupation'])))
ax_women.set_xticklabels(df_women_all_ed_levels['Occupation'], rotation=45, ha='right')

# Adjust layout and show plot
plt.tight_layout()
plt.show()


# #### Process the second xlsx file from the American Community Survey
# - Recreate the steps used on the first file from the dataset

# In[147]:


# Define the path to the file
file_path = './data/from-college-to-jobs-acs-2019/job-by-bach-degree-field-bach-degree.xlsx'

# Specify only the required columns by their indices
selected_columns = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

# Define the starting row and the number of rows to read
start_row = 27
num_rows = 20  # Number of rows to read after the starting row

# Load the data with only the selected columns and limit the rows
df_men_bach_degree = pd.read_excel(file_path, skiprows=start_row, nrows=num_rows, usecols=selected_columns)

# Define descriptive column names for the selected columns
columns = [
    "Occupation", "FoD-Computers_Math_Stats", "FoD-Engineering", "FoD-Physical_Sciences",
    "FoD-Biological_Environmental_Agricultural_Sciences", "FoD-Psychology",
    "FoD-Social_Sciences", "FoD-Multidiscipline", "FoD-Science_and_Engineering_Related",
    "FoD-Business", "FoD-Education", "FoD-Literature_and_Languages",
    "FoD-Liberal_Arts_and_History", "FoD-Visual_and_Performing_Arts",
    "FoD-Communications", "FoD-Other_EG_Criminal_Justice_or_Social_Work"
]

# Apply the custom column names
df_men_bach_degree.columns = columns

# Remove any leading dots or whitespace from the 'Occupation' column
df_men_bach_degree['Occupation'] = df_men_bach_degree['Occupation'].str.lstrip(". ")

# Add a gender column
df_men_bach_degree['Gender'] = 'Male'

# Select numeric columns that need conversion to int64
numeric_columns = [
    "FoD-Computers_Math_Stats", "FoD-Engineering", "FoD-Physical_Sciences",
    "FoD-Biological_Environmental_Agricultural_Sciences", "FoD-Psychology",
    "FoD-Social_Sciences", "FoD-Multidiscipline", "FoD-Science_and_Engineering_Related",
    "FoD-Business", "FoD-Education", "FoD-Literature_and_Languages",
    "FoD-Liberal_Arts_and_History", "FoD-Visual_and_Performing_Arts",
    "FoD-Communications", "FoD-Other_EG_Criminal_Justice_or_Social_Work"
]

# Convert selected numeric columns to int64
df_men_bach_degree[numeric_columns] = df_men_bach_degree[numeric_columns].astype('int64')

# Preview to confirm correct data import and types
print(df_men_bach_degree.head())
print(df_men_bach_degree.tail())
print(df_men_bach_degree.info())


# In[148]:


# Define the path to the file
file_path = './data/from-college-to-jobs-acs-2019/job-by-bach-degree-field-bach-degree.xlsx'

# Specify the selected columns by their indices (same as for men)
selected_columns = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

# Define the header row and the number of rows for women
header_row = 49
num_rows = 20  # The same as for men

# Load the data for women with the specified columns, skipping initial rows and using the right number of rows
df_women_bach_degree = pd.read_excel(
    file_path, skiprows=header_row, nrows=num_rows, usecols=selected_columns
)

# Define descriptive column names
columns = [
    "Occupation", "FoD-Computers_Math_Stats", "FoD-Engineering", "FoD-Physical_Sciences",
    "FoD-Biological_Environmental_Agricultural_Sciences", "FoD-Psychology",
    "FoD-Social_Sciences", "FoD-Multidiscipline", "FoD-Science_and_Engineering_Related",
    "FoD-Business", "FoD-Education", "FoD-Literature_and_Languages",
    "FoD-Liberal_Arts_and_History", "FoD-Visual_and_Performing_Arts",
    "FoD-Communications", "FoD-Other_EG_Criminal_Justice_or_Social_Work"
]

# Apply the column names
df_women_bach_degree.columns = columns

# Remove any leading dots or whitespace from the 'Occupation' column
df_women_bach_degree['Occupation'] = df_women_bach_degree['Occupation'].str.lstrip(". ")

# Add a gender column with "Female" as the value
df_women_bach_degree['Gender'] = 'Female'

# Convert numeric columns to int64 to ensure consistency with the men’s DataFrame
numeric_columns = [
    "FoD-Computers_Math_Stats", "FoD-Engineering", "FoD-Physical_Sciences",
    "FoD-Biological_Environmental_Agricultural_Sciences", "FoD-Psychology",
    "FoD-Social_Sciences", "FoD-Multidiscipline", "FoD-Science_and_Engineering_Related",
    "FoD-Business", "FoD-Education", "FoD-Literature_and_Languages",
    "FoD-Liberal_Arts_and_History", "FoD-Visual_and_Performing_Arts",
    "FoD-Communications", "FoD-Other_EG_Criminal_Justice_or_Social_Work"
]
df_women_bach_degree[numeric_columns] = df_women_bach_degree[numeric_columns].astype('int64')

# Preview to confirm correct data import and types
print(df_women_bach_degree.head())
print(df_women_bach_degree.tail())
print(df_women_bach_degree.info())


# #### Process the third xlsx file from the American Community Survey
# - Recreate the steps used on the first and second files from the dataset

# In[149]:


# Define the path to the file
file_path = './data/from-college-to-jobs-acs-2019/job-by-bach-degree-field-grad-degree.xlsx'

# Specify only the required columns by their indices
selected_columns = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

# Define the starting row and the number of rows to read
start_row = 27
num_rows = 20  # Number of rows to read after the starting row

# Load the data with only the selected columns and limit the rows
df_men_grad_degree = pd.read_excel(file_path, skiprows=start_row, nrows=num_rows, usecols=selected_columns)

# Define descriptive column names for the selected columns
columns = [
    "Occupation", "FoD-Computers_Math_Stats", "FoD-Engineering", "FoD-Physical_Sciences",
    "FoD-Biological_Environmental_Agricultural_Sciences", "FoD-Psychology",
    "FoD-Social_Sciences", "FoD-Multidiscipline", "FoD-Science_and_Engineering_Related",
    "FoD-Business", "FoD-Education", "FoD-Literature_and_Languages",
    "FoD-Liberal_Arts_and_History", "FoD-Visual_and_Performing_Arts",
    "FoD-Communications", "FoD-Other_EG_Criminal_Justice_or_Social_Work"
]

# Apply the custom column names
df_men_grad_degree.columns = columns

# Remove any leading dots or whitespace from the 'Occupation' column
df_men_grad_degree['Occupation'] = df_men_grad_degree['Occupation'].str.lstrip(". ")

# Add a gender column
df_men_grad_degree['Gender'] = 'Male'

# Select numeric columns that need conversion to int64
numeric_columns = [
    "FoD-Computers_Math_Stats", "FoD-Engineering", "FoD-Physical_Sciences",
    "FoD-Biological_Environmental_Agricultural_Sciences", "FoD-Psychology",
    "FoD-Social_Sciences", "FoD-Multidiscipline", "FoD-Science_and_Engineering_Related",
    "FoD-Business", "FoD-Education", "FoD-Literature_and_Languages",
    "FoD-Liberal_Arts_and_History", "FoD-Visual_and_Performing_Arts",
    "FoD-Communications", "FoD-Other_EG_Criminal_Justice_or_Social_Work"
]

# Convert selected numeric columns to int64
df_men_grad_degree[numeric_columns] = df_men_grad_degree[numeric_columns].astype('int64')

# Preview to confirm correct data import and types
print(df_men_grad_degree.head())
print(df_men_grad_degree.tail())
print(df_men_grad_degree.info())


# In[150]:


# Define the path to the file
file_path = './data/from-college-to-jobs-acs-2019/job-by-bach-degree-field-grad-degree.xlsx'

# Specify the selected columns by their indices (same as for men)
selected_columns = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

# Define the header row and the number of rows for women
header_row = 49
num_rows = 20  # The same as for men

# Load the data for women with the specified columns, skipping initial rows and using the right number of rows
df_women_grad_degree = pd.read_excel(
    file_path, skiprows=header_row, nrows=num_rows, usecols=selected_columns
)

# Define descriptive column names
columns = [
    "Occupation", "FoD-Computers_Math_Stats", "FoD-Engineering", "FoD-Physical_Sciences",
    "FoD-Biological_Environmental_Agricultural_Sciences", "FoD-Psychology",
    "FoD-Social_Sciences", "FoD-Multidiscipline", "FoD-Science_and_Engineering_Related",
    "FoD-Business", "FoD-Education", "FoD-Literature_and_Languages",
    "FoD-Liberal_Arts_and_History", "FoD-Visual_and_Performing_Arts",
    "FoD-Communications", "FoD-Other_EG_Criminal_Justice_or_Social_Work"
]

# Apply the column names
df_women_grad_degree.columns = columns

# Remove any leading dots or whitespace from the 'Occupation' column
df_women_grad_degree['Occupation'] = df_women_grad_degree['Occupation'].str.lstrip(". ")

# Add a gender column with "Female" as the value
df_women_grad_degree['Gender'] = 'Female'

# Convert numeric columns to int64 to ensure consistency with the men’s DataFrame
numeric_columns = [
    "FoD-Computers_Math_Stats", "FoD-Engineering", "FoD-Physical_Sciences",
    "FoD-Biological_Environmental_Agricultural_Sciences", "FoD-Psychology",
    "FoD-Social_Sciences", "FoD-Multidiscipline", "FoD-Science_and_Engineering_Related",
    "FoD-Business", "FoD-Education", "FoD-Literature_and_Languages",
    "FoD-Liberal_Arts_and_History", "FoD-Visual_and_Performing_Arts",
    "FoD-Communications", "FoD-Other_EG_Criminal_Justice_or_Social_Work"
]
df_women_grad_degree[numeric_columns] = df_women_grad_degree[numeric_columns].astype('int64')

# Preview to confirm correct data import and types
print(df_women_grad_degree.head())
print(df_women_grad_degree.tail())
print(df_women_grad_degree.info())


# #### Process the 4th xlsx file from the American Community Survey

# In[151]:


# Define the path to the file
file_path = './data/from-college-to-jobs-acs-2019/med-earn-by-degree-level-field-and-occupation.xlsx'

# Define the columns to read (based on the specified indices)
selected_columns = [0, 1, 3, 5, 7, 9, 11]

# Define column names for the data
columns = [
    "Occupation", "STEM Major All Degrees", "non-STEM Major All Degrees",
    "STEM Major Bachelors", "non-STEM Major Bachelors",
    "STEM Major Graduate Degree", "non-STEM Major Graduate Degree"
]

# Load the data for men
start_row_men = 16  
num_rows_men = 7  # Number of rows for the men's data section
male_median_earnings = pd.read_excel(file_path, skiprows=start_row_men, nrows=num_rows_men, usecols=selected_columns)
male_median_earnings.columns = columns

# Remove any leading dots or whitespace from the 'Occupation' column
male_median_earnings['Occupation'] = male_median_earnings['Occupation'].str.lstrip(". ")

male_median_earnings['Gender'] = 'Male'

# Load the data for women
start_row_women = start_row_men + num_rows_men + 2  
num_rows_women = 7  # Number of rows for the women's data section
female_median_earnings = pd.read_excel(file_path, skiprows=start_row_women, nrows=num_rows_women, usecols=selected_columns)
female_median_earnings.columns = columns

# Remove any leading dots or whitespace from the 'Occupation' column
female_median_earnings['Occupation'] = female_median_earnings['Occupation'].str.lstrip(". ")

female_median_earnings['Gender'] = 'Female'

# Preview the results to confirm
print("Men's Median Earnings DataFrame:")
display(male_median_earnings.head(6))
print("Women's Median Earnings DataFrame:")
display(female_median_earnings.head(6))


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# 
# - https://it4063c.github.io/course-notes/working-with-data/data-sources for methods to import the various data types  
# - https://www.kaggle.com/datasets/hackerrank/developer-survey-2018/data for the Kaggle dataset 
# - https://www.pewresearch.org/social-trends/2018/01/09/women-and-men-in-stem-often-at-odds-over-workplace-equity/  for the link to the Pew Research Survey
# - https://ncses.nsf.gov/pubs/nsb20212/participation-of-demographic-groups-in-stem for the html scraped dataset
# - https://ncses.nsf.gov/pubs/nsb20212/downloads for additional tables that were not part of the html scrape
# - https://www.census.gov/library/stories/2021/06/does-majoring-in-stem-lead-to-stem-job-after-graduation.html for the links to the American Community Survey 2019
# - IT4075 Applied Machine Learning zyBooks for data classification logic and code
# - https://medium.com/@acceldia/python-101-reading-excel-and-spss-files-with-pandas-eed6d0441c0b to learn how to work with .sav files
# - https://python-docx.readthedocs.io/en/latest/user/documents.html to learn how to work with .docx files inside Python

# In[152]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

