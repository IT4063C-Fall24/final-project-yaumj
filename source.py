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
# 
# **Relating the Data**  
# The datasets can be linked based on the shared year (2017) and gender as a common variable. Gender will serve as a primary key or part of a composite key for linking.

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

# In[2]:


#import packages
import os # to create subfolder for data organization
from dotenv import load_dotenv
load_dotenv(override=True)

import opendatasets as od
import pandas as pd
import pyreadstat
import requests
import re # for string manipulation

import matplotlib.pyplot as plt
import seaborn as sns

from zipfile import ZipFile
from urllib.request import urlretrieve
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from docx import Document


# #### Import Dataset 1: HackerRank Developer Survey Published in 2018 that covered 2017 Questionnaire Responses

# In[81]:


# import dataset from Kaggle using URL 
dataset_url = "https://www.kaggle.com/datasets/hackerrank/developer-survey-2018/data"
od.download(dataset_url, data_dir="./data")


# #### Convert dataset to a pandas dataframe and inspect data for Exploratory Data Analysis (EDA)

# In[82]:


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

# In[83]:


# Find United States in the country_code_df to use for filtering purposes

# Rename columns
country_code_df.rename(columns={'Value': 'country_code', 'Label': 'country'}, inplace=True)

# Filter for the United States
us_country_code_df = country_code_df[country_code_df['country'] == 'United States']

# Display the resulting DataFrame
display(us_country_code_df)


# #### Filtered dataframe to include only respondents in the United States

# In[84]:


# Filter the DataFrame for United States questionnaire responses
us_dev_survey_numeric_df = dev_survey_numeric_df[dev_survey_numeric_df['CountryNumeric2'] == 167]

# Display the number of records
num_records_numeric = us_dev_survey_numeric_df.shape[0]
display(f"Number of records: {num_records_numeric}")

# Display the resulting DataFrame
display(us_dev_survey_numeric_df.head(5))


# #### Reduce dataframe columns to only those relevant to supporting or disproving my hypothesis
# - fields such as date survey was completed and questions about the HackerRank survey were removed from dataframe for simplification

# In[85]:


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

# In[108]:


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


# #### Separate Responses by Worker Responses and Hiring Manager Responses
# - Used q16HiringManager response to determine role
# - Chose specific response fields relevant to hypothesis such as age, gender, industry, job level, current role, and job qualities for worker dataframe
# - Survey questions were ordered by if they applied to worker responses or hiring manager responses. For example, everything after q16HiringManager were
# questions only shown/relevant to people who responded "Yes"

# In[86]:


# Define columns for Worker Responses Numeric DataFrame
worker_columns_numeric = [
    'RespondentID', 'q2Age', 'q3Gender', 'q10Industry',
    'q8JobLevel', 'q9CurrentRole',
    'q12JobCritPrefTechStack', 'q12JobCritCompMission',
    'q12JobCritCompCulture', 'q12JobCritWorkLifeBal',
    'q12JobCritCompensation', 'q12JobCritProximity',
    'q12JobCritPerks', 'q12JobCritSmartPeopleTeam',
    'q12JobCritImpactwithProduct', 'q12JobCritInterestProblems',
    'q12JobCritFundingandValuation', 'q12JobCritStability',
    'q12JobCritProfGrowth', 'q16HiringManager'
]

# Create Worker Responses Numeric DataFrame
worker_responses_numeric_df = filtered_us_dev_survey_numeric_df[
    filtered_us_dev_survey_numeric_df['q16HiringManager'] == 2  # Not a hiring manager
][worker_columns_numeric]

# Define columns for Hiring Manager Responses Numeric DataFrame
hiring_manager_columns_numeric = [
    'RespondentID', 'q2Age', 'q3Gender', 'q10Industry',
    'q8JobLevel', 'q9CurrentRole', 'q16HiringManager',
    'q17HirChaNoDiversCandidates', 'q20CandYearExp',
    'q20CandCompScienceDegree', 'q20CandCodingBootcamp',
    'q20CandSkillCert', 'q20CandHackerRankActivity',
    'q20CandOtherCodingCommAct', 'q20CandGithubPersProj',
    'q20CandOpenSourceContrib', 'q20CandHackathonPart',
    'q20CandPrevWorkExp', 'q20CandPrestigeDegree',
    'q20CandLinkInSkills', 'q20CandGithubPersProj2'
]

# Create Hiring Manager Responses Numeric DataFrame
hiring_manager_responses_numeric_df = filtered_us_dev_survey_numeric_df[
    filtered_us_dev_survey_numeric_df['q16HiringManager'] == 1  # Is a hiring manager
][hiring_manager_columns_numeric]

# Display the resulting Worker Responses Numeric DataFrame and info about the dataframe
print("Worker Responses Numeric DataFrame:")
display(worker_responses_numeric_df.info())
display(worker_responses_numeric_df.head(5))

# Display the resulting Hiring Manager Responses Numeric DataFrame and info about the dataframe
print("Hiring Manager Responses Numeric DataFrame:")
display(hiring_manager_responses_numeric_df.info())
display(hiring_manager_responses_numeric_df.head(5))


# #### Exploratory Data Analysis (EDA)
# - The numeric dataframe should consist entirely of int64 data types, yet the majority have an "object" data type instead.
#  - These datatypes will need to be converted for certain types of analysis like a correlation matrix.

# In[156]:


# Convert all columns in Worker Responses Numeric DataFrame to int64
worker_responses_numeric_df = worker_responses_numeric_df.apply(pd.to_numeric, errors='coerce').astype('Int64')

# Convert all columns in Hiring Manager Responses Numeric DataFrame to int64
hiring_manager_responses_numeric_df = hiring_manager_responses_numeric_df.apply(pd.to_numeric, errors='coerce').astype('Int64')

# Display the resulting Worker Responses Numeric DataFrame and info about the dataframe
print("Worker Responses Numeric DataFrame:")
display(worker_responses_numeric_df.info())
display(worker_responses_numeric_df.head(5))

# Display the resulting Hiring Manager Responses Numeric DataFrame and info about the dataframe
print("Hiring Manager Responses Numeric DataFrame:")
display(hiring_manager_responses_numeric_df.info())
display(hiring_manager_responses_numeric_df.head(5))


# #### Data Visualization using Seaborn Correlation Matrix
# **Purpose:** Examine the relationships between key demographic and job criteria variables in the Worker Responses Numeric DataFrame.
# 
# **Insights:**
# - **Age and Job Level Correlation:** The matrix reveals a moderate correlation between *Age* and *Job Level*, indicating that older workers are more likely to hold higher job levels, reflecting common career progression trends.
# - **Weak Correlations:** Most other variables show minimal correlation, suggesting that individual factors like *job preferences* or *industry* alone do not strongly predict other attributes, making them less effective for simple statistical analysis--at least in this specific dataset.

# In[158]:


# Filter out rows with any missing values for the correlation calculation
worker_responses_filtered = worker_responses_numeric_df.dropna()

# Compute the correlation matrix on the filtered data
worker_correlation_matrix = worker_responses_filtered.corr()

# Plot the correlation heatmap for Worker Responses
plt.figure(figsize=(10, 8))
sns.heatmap(worker_correlation_matrix, annot=False, cmap='coolwarm')
plt.title("Correlation Matrix for Worker Responses Numeric DataFrame")
plt.show()


# ## Repeat filtering logic from numeric version of data on the values version of the data
# - Both datasets contain the same records, but one has numeric codes for all responses and the other has plain language values for all responses 
# so the same logic can be used for both dataframes.

# In[87]:


# Filter the DataFrame for CountryNumeric2 = "United States"
us_dev_survey_values_df = dev_survey_values_df[dev_survey_values_df['CountryNumeric2'] == "United States"]

# Display the number of records
num_records_values = us_dev_survey_values_df.shape[0]
display(f"Number of records: {num_records_values}")

# Display the resulting DataFrame
display(us_dev_survey_values_df.head(5))


# In[88]:


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


# In[89]:


# Define columns for Worker Responses Values DataFrame
worker_columns_values = [
    'RespondentID', 'q2Age', 'q3Gender', 'q10Industry',
    'q8JobLevel', 'q9CurrentRole', 'q16HiringManager',
    'q12JobCritPrefTechStack', 'q12JobCritCompMission',
    'q12JobCritCompCulture', 'q12JobCritWorkLifeBal',
    'q12JobCritCompensation', 'q12JobCritProximity',
    'q12JobCritPerks', 'q12JobCritSmartPeopleTeam',
    'q12JobCritImpactwithProduct', 'q12JobCritInterestProblems',
    'q12JobCritFundingandValuation', 'q12JobCritStability',
    'q12JobCritProfGrowth'
]

# Create Worker Responses Values DataFrame
worker_responses_values_df = filtered_us_dev_survey_values_df[
    filtered_us_dev_survey_values_df['q16HiringManager'] == 'No'  # Not a hiring manager
][worker_columns_values]

# Define columns for Hiring Manager Responses Values DataFrame
hiring_manager_columns_values = [
    'RespondentID', 'q2Age', 'q3Gender', 'q10Industry',
    'q8JobLevel', 'q9CurrentRole', 'q16HiringManager',
    'q17HirChaNoDiversCandidates', 'q20CandYearExp',
    'q20CandCompScienceDegree', 'q20CandCodingBootcamp',
    'q20CandSkillCert', 'q20CandHackerRankActivity',
    'q20CandOtherCodingCommAct', 'q20CandGithubPersProj',
    'q20CandOpenSourceContrib', 'q20CandHackathonPart',
    'q20CandPrevWorkExp', 'q20CandPrestigeDegree',
    'q20CandLinkInSkills', 'q20CandGithubPersProj2'
]

# Create Hiring Manager Responses Values DataFrame
hiring_manager_responses_values_df = filtered_us_dev_survey_values_df[
    filtered_us_dev_survey_values_df['q16HiringManager'] == 'Yes'  # Is a hiring manager
][hiring_manager_columns_values]

# Display the resulting Worker Responses Values DataFrame and info about the dataframe
print("Worker Responses Values DataFrame:")
display(worker_responses_values_df.info())
display(worker_responses_values_df.head(5))

# Display the resulting Hiring Manager Responses Values DataFrame and info about the dataframe
print("Hiring Manager Responses Values DataFrame:")
display(hiring_manager_responses_values_df.info())
display(hiring_manager_responses_values_df.head(5))


# #### Clean the workers dataframe
# - Removed records where gender is missing because they cannot be used to prove/disprove the hypothesis.
# - Filtered out records where respondents were 18 or younger since they were unlikely to be relevant to investigating the professional levels in the technology field. 
# Anyone under 18 who shows a job, is essentially an outlier for the purpose of my dataset.
# - Filtered out records where the age is null because both age and gender are necessary to determine job level comparisons.
# - Filtered out records where current role or job level is student because they are not relevant to my hypothesis.
# - Filtered out records where both the Job Level and Current Role were NaN because there is no way to determine values for the field if both are blank.
# 

# In[109]:


# Rename columns for clarity
worker_responses_values_df.rename(columns={
    'q2Age': 'Age',
    'q3Gender': 'Gender',
    'q10Industry': 'Industry',
    'q8JobLevel': 'Job Level',
    'q9CurrentRole': 'Current Role'
}, inplace=True)

# Drop records where Gender is null
worker_responses_values_df.dropna(subset=['Gender'], inplace=True)

# Drop records where Gender is '#NULL!'
worker_responses_values_df.drop(
    worker_responses_values_df[worker_responses_values_df['Gender'] == '#NULL!'].index,
    inplace=True
)

# Filter out respondents who are categorized as "Under 12 years old" or "12 - 18 years old"
worker_responses_values_df = worker_responses_values_df[
    ~worker_responses_values_df['Age'].isin(["Under 12 years old", "12 - 18 years old"])
]

# Filter out rows where Age is '#NULL!'
worker_responses_values_df = worker_responses_values_df[
    worker_responses_values_df['Age'] != '#NULL!'
]

# Filter out rows where Current Role or Job Level is "Student"
worker_responses_values_df = worker_responses_values_df[
    (worker_responses_values_df['Current Role'] != 'Student') & 
    (worker_responses_values_df['Job Level'] != 'Student')
]

# Filter out rows where both Job Level and Current Role are NaN
worker_responses_values_df = worker_responses_values_df[
    ~ (worker_responses_values_df['Job Level'].isna() & worker_responses_values_df['Current Role'].isna())
]

# Display how many rows remain
print(f"Remaining records after cleaning: {worker_responses_values_df.shape[0]}")

# Display the cleaned DataFrame
display(worker_responses_values_df.head(5))


# #### Data Visualization using Matplotlib Bar Graph
# **Purpose:** Perform a preliminary analysis of the HackerRank Dataset breakdown by age and gender
# **Insights:**
# - **Age Group Representation**: The graph reveals how many workers fall into each age group, highlighting the most common age demographics within the dataset.
# - **Gender Distribution**: By comparing the heights of the bars for different genders, we can see which age groups have higher counts of male, female, and non-binary workers. This highlights trends in workforce composition.
# - **Demographic Changes:** The visualization aims to assess whether there is a change in demographics proportionally from age group to age group, providing insights into how representation shifts across different stages of workforce experience.

# In[154]:


# Group the data by Age Group and Gender, and count occurrences
age_gender_counts = worker_responses_values_df.groupby(['Age', 'Gender']).size().unstack(fill_value=0)

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

# In[102]:


# Review dataset to determine what data is relevant

# Check Current Role for Job Level "New grad"
new_grad_roles = worker_responses_values_df[worker_responses_values_df['Job Level'] == 'New grad'] \
    .groupby('Current Role').size().reset_index(name='Counts')

# Filter for rows where Job Level is NaN and Current Role is not NaN
nan_job_level_current_role_df = worker_responses_values_df[worker_responses_values_df['Job Level'].isna() & 
    worker_responses_values_df['Current Role'].notna()]

# Display results
print("Current Roles for Job Level 'New grad':")
display(new_grad_roles)

print("Rows where Job Level is NaN and Current Role is not NaN:")
display(nan_job_level_current_role_df[['Job Level', 'Current Role']])


# #### More Exploratory Data Analysis (EDA)
# - Check if there is a dominant job level associated with the Current Role field that could be used to populate empty fields.

# In[129]:


# Group by Current Role and Job Level, and count occurrences
role_level_counts = worker_responses_values_df.groupby(['Current Role', 'Job Level']).size().reset_index(name='Counts')

# Set the option to display all rows for this code cell only
with pd.option_context('display.max_rows', None):
    display(role_level_counts)


# #### Data Visualization using Seaborn Box Plot
# **Purpose:** Demonstrate how gender may impact job levels within specific career roles, in this case, "Software Test Engineer."
# 
# **Insights:**
# - **Gender-based Distribution:** The box plot illustrates how job levels are distributed across genders within the "Software Test Engineer" role. For instance, males are generally concentrated in junior and mid-level positions, while non-binary individuals occupy a broader range of senior levels.
# - **Outliers and Range:** The visualization shows that job levels for male and non-binary workers in this role vary significantly, with outliers extending to senior positions, indicating possible disparities in advancement.
# - **Rationale for Machine Learning Approach:** The diverse distribution across gender groups suggests that simple statistical measures like the mean or median would be insufficient for accurately predicting job level. Instead, using a KNN classifier can better capture these nuances by considering demographic variables like gender and role.
# 
# This visualization highlights the relevance of employing a machine learning approach that can adapt to demographic differences, rather than relying on single-point estimates like the mean or median for classification.

# In[152]:


# Filter for a specific current role 
specific_role = "Software Test Engineer"
role_data = worker_responses_values_df[worker_responses_values_df['Current Role'] == specific_role].copy()

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

# In[105]:


# Use KNearestNeighbors to determine most likely Job Level based on age, gender, and current role

# Prepare the DataFrame for KNN training
train_df = worker_responses_values_df[worker_responses_values_df['Job Level'].notna()].copy()

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
predict_df = worker_responses_values_df[worker_responses_values_df['Job Level'].isna()].copy()
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
display(predict_df[['Current Role', 'Predicted Job Level']])


# In[107]:


#Update Job Level field with predictions and verify changes

# Update the Job Level in the original DataFrame where it was NaN
worker_responses_values_df.loc[worker_responses_values_df['Job Level'].isna(), 'Job Level'] = predicted_levels

# Verify that no records remain with NaN Job Level
remaining_nan_job_levels = worker_responses_values_df['Job Level'].isna().sum()
print(f"Number of records with NaN Job Level after updating: {remaining_nan_job_levels}")

# Display the cleaned DataFrame
display(worker_responses_values_df.head(5))


# #### Import Dataset 2: 2017 Pew Research Center STEM Survey

# In[111]:


# import zip file from Pew Research
file_handle, _ = urlretrieve("https://www.pewresearch.org/wp-content/uploads/sites/20/2019/04/2017-Pew-Research-Center-STEM-survey.zip")
zipfile = ZipFile(file_handle, "r")
zipfile.extractall("./data")
zipfile.close()


# #### Examine contents of .sav file

# In[122]:


file_path = 'data/materials for public release/2017 Pew Research Center STEM survey.sav'

# Read the .sav file into a DataFrame
df, meta = pyreadstat.read_sav(file_path)

# Display basic information about the DataFrame
print(df.info())

# Display the first few rows of the DataFrame
print(df.head())


# #### Read the .docx file
# - Read the Pew Research Center files associated with the .sav file and convert them into .txt files to understand the codes used.

# In[124]:


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

# In[123]:


# Display all column names in the DataFrame
print("Column names in the Pew Research dataset:")
for col in df.columns:
    print(col)


# ## Data after this point is still in the process of being analyzed, cleaned, and transformed.
# - Due to time constraints for the submission of Checkpoint 2, the work on these datasets is not complete. However, they will continue to be worked with for the final project.
# - The work prior to this point should include all requirements for Checkpoint 2: Exploratory Data Analysis & Visualization aside from one of the visualizations, which is included after Dataset 4 was imported and partially processed.

# #### Import Dataset 3: National Center for Science and Engineering Statistics (NCSES)
# - Includes demographic breakdown of STEM participation in the workforce from 1993 - 2019
# - Data was compiled by the NCSES from the U.S. Census Bureau, American Community Survey, National Center for Science and Engineering Statistics, and more
# - For the full list of compiled sources: https://ncses.nsf.gov/pubs/nsb20212/data#source-block 

# In[6]:


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


# #### Import Dataset 4: United States Census Bureau
# - From College to Jobs: American Community Survey 2019

# In[3]:


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

# In[15]:


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

# In[16]:


df_men_all_ed_levels.info()


# **Convert columns to correct data types (float64 to int64)**

# In[17]:


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

# In[24]:


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

# In[27]:


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


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# 
# - https://it4063c.github.io/course-notes/working-with-data/data-sources for methods to import the various data types  
# - https://www.kaggle.com/datasets/hackerrank/developer-survey-2018/data for the Kaggle dataset 
# - https://www.pewresearch.org/social-trends/2018/01/09/women-and-men-in-stem-often-at-odds-over-workplace-equity/  for the link to the Pew Research Survey
# - https://ncses.nsf.gov/pubs/nsb20212/participation-of-demographic-groups-in-stem for the html scraped dataset
# - https://www.census.gov/library/stories/2021/06/does-majoring-in-stem-lead-to-stem-job-after-graduation.html for the links to the American Community Survey 2019
# - IT4075 Applied Machine Learning zyBooks for data classification logic and code
# - https://medium.com/@acceldia/python-101-reading-excel-and-spss-files-with-pandas-eed6d0441c0b to learn how to work with .sav files
# - https://python-docx.readthedocs.io/en/latest/user/documents.html to learn how to work with .docx files inside Python

# In[1]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

