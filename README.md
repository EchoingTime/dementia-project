URL: https://github.com/EchoingTime/dementia-project.git

# Dementia Project

Programmed By: Dante Anzalone, Dustin McDonnell, and Leidy Rojas Huisa

Course: CSCI 4105: Knowledge Discovery and Data Mining

Professor: Duo Wei

Date: 2024-10-21

### **Description**:

Machine learning in medicine: Classification and prediction of dementia by support vector machines (SVM) is a study
exploring the use of support vector machines for predicting dementia from a longitudinal group of 150 subjects aged 60
to 96, including 373 subjects with MRI data. This study focuses on using machine learning algorithms to ease the
medical industry with the collection of medical data, which in this case uses machine learning algorithms to predict
dementia and check on possible dementia progress.

### **Objective**:

To apply data mining techniques to the dementia data set and draw connections, e.g., age related to dementia,
by extracting useful and relevant information.

### Preliminary Research Questions:

1) Conduct data cleaning of the Dementia dataset by removing duplicates, handling missing values, and correcting
inconsistencies.

2) Understanding risk factors with data clustering by revealing risk factors among patients in similar clusters and
helping to identify the potential progress of dementia.

3) Using association rules for patient segmentation can encounter existing relationships and patterns among patients with
related conditions.

4) Construct a decision tree to categorize data by predicting the possibility of a patient developing dementia based on
clinical and demographic data.

### Oasis Longitudinal Demographics Dataset Terminology

This section was used alongside AI to help with finding out aberrations.

1) Subject ID --> ID of Patient
2) Group --> Represents whether a patient is classified as Nondemented (2), demented (1), or converted (0). Healthy 
patients are marked with 2, patients diagnosed with dementia are marked with 1, and patients who transitioned to 
demented between visits are marked with 0. 
3) Visit --> Time data was recorded as patients can have several visits. Easy to see how many visits a patient had.
4) MR Delay --> Number of days between visit and MRI scan
5) M/F --> Gender of patient. M means Male and F means Female
6) Age --> Age of patient at the time of visit. Important for dementia diagnosis
7) EDUC --> Patient's years of education. Good for cognitive reservation as it slows rate of cognitive decline
8) SES --> Socioeconomic status of patient such as income, occupation, etc. Can affect patient's physical and cognitive 
health
9) MMSE --> Mini-Mental State Examination score. Assesses cognitive function, ranging from 0 to 30. The lower the score
means worse cognitive impairment.
10) CDR --> Clinical Dementia Rating scale --> Quantify the severity of dementia symptoms. 0 (no dementia) to 3 
(severe dementia)
11) eTIV --> Estimate Total Intracranial Volume. Total volume of brain and skull. Changes in volume mark 
neurodegenerative diseases.
12) nWBV --> Normalized Whole Brain Volume. Total brain volume normalized to the eTIV. Good for seeing individual 
differences in brain size.
13) ASF --> Atlas Scaling Factor. Measure used in neuroimaging studies to account for differences in scaling and the 
size of brain images.
14) Cluster --> Cluster assignment that can help with showing patterns of brain activity, scores, etc.
15) Predictions --> Predicting demented, nondemented, or converted based on modeling/classification algorithm.

### Predictions Dataset

1) Age
2) CDR
3) M/F
4) MMSE
5) MR Delay
6) SES
7) Subject ID
8) Visit
9) Group --> Label
10) confidence(Nondemented) --> Model's confidence that a patient is non-demented.
11) confidence(Demented) --> Model's confidence that a patient is demented.
12) confidence(Converted) --> Model's confidence that the patient is converted.
13) confidence(Group) --> Confidence on predicting a group.

### The Rest of the Information Was Taken From The Course's Description of Dementia Classification and Prediction

#### Objective:

The object of the term project is to gain practical experience with Dementia related dataset to complete a data mining 
task.

Example tasks include: Decision Tree, Association Rules, Clustering, and Prediction. This is a semester-long project 
with groups of 1-3 students.


#### Overview

Dementia is a syndrome – usually of a chronic or progressive nature – in which there is deterioration in cognitive 
function (i.e. the ability to process thought) beyond what might be expected from norming aging. It affects memory,
thinking, orientation, comprehension, calculation, learning capacity, language, and judgement. Consciousness is not 
affected.

The impairment in cognitive function is commonly accompanied and occasionally preceded by deterioration in emotional 
control, social behavior, or motivation.

Dementia results from a variety of diseases and injuries that primarily or secondarily affect the brain, such as 
Alzheimer's disease or stroke. Dementia is one of the major causes of disability and dependency among older people 
worldwide. It can be overwhelming, not only for the people who have it, but also for their caregivers and families.

There is often a lack of awareness and understanding of dementia, resulting in stigmatization and barriers to diagnosis 
and care. The impact of dementia on caregivers, family, and society at large can be physical, psychological, and social 
and economic.

#### Dataset

This set consists of a longitudinal collection of 150 subjects aged 60 to 96. Each subject was scanned on two or more 
visits, separated by at least one year for a total of 373 imaging sessions.

For each subject, 3 or 4 individual T1-weighted MRI scans obtained in single scan sessions are included. The subjects 
are all right-handed and include both men and women. 72 of the subjects were characterized as non-demented throughout 
the study. 64 of the subjects were characterized as demented at the time of their initial visits and remained so for 
subsequent scans, including 51 individuals with mild to moderate Alzheimer's disease. Another 14 subjects were 
characterized as non-demented at the time of their initial visit and were subsequently characterized as demented at a 
later visit.

Detailed Tabled Description: https://data.mendeley.com/datasets/tsy6rbc5d4/1

#### Project Expectations

Each group is expected to:

1) Conduct a literature survey of some of the most recent data mining techniques
2) Explore the dataset and visualize the dataset
3) Propose some interesting questions to investigate in the project
4) Design a program of one or more hypotheses
5) Report findings

The goal is not to invent a new algorithm, but rather to apply existing techniques to a new data mining application, 
or explore different feature representations on the existing task. Note that not all data columns are expected to be 
used, instead, only a few selected columns will be investigated.

Research could answer, but not limited to, the following questions:

1) Investigating the Dementia dataset by addressing issues like missing data, duplicates, and more (preprocessing)
2) Getting the data ready for data mining tasks through processes like normalization, standardization, feature 
selection, and more (preprocessing)
3) Predicting subject group classification (non-demented/demented) and using factors like age, gender, race, and more.
4) Forecasting subject group outcomes exampling diverse models and assessing the performance of these prediction models.
5) Identifying anomalies within the dataset (anomaly detection).

#### Phases of the Project
1) Preprocess the dataset by dealing with missing data, duplicate data, discretion, outlier detection
2) Determine the correlations among variables and consider whether dimensionality reduction, feature selection, 
feature scaling, data normalization, or other measures are necessary.
3) Develop models, which can involve building one or multiple models. It's particularly engaging to compare various 
techniques or apply different data mining approaches such as clustering, classification, association, and prediction.
4) Evaluate your model by using measures such as precision, recall, F-1 measure, accuracy.
5) Explain the results by using visualization including displaying confusion matrix, displaying AUC, ROC, etc.

#### Project Paper

Project paper should include both a literature review and proposing/applying new or modified algorithms to the 
real-world datasets. Will report your algorithm and experiment results on realworld datasets utilizing the IMRAD 
structure for scientific writing: 
https://en.wikipedia.org/wiki/IMRAD#:~:text=In%20scientific%20writing%2C%20IMRAD%20or,of%20the%20original%20research%20type.

The paper needs to contain at least the following sections: 
(at least five pages excluding references, letter page, single-space, font size 11, margin-normal)

* Title
* Abstract
  * What topics are you going to work on? 
  * Why do you want to work on that topic? 
  * What results are you expected to get?
* Introduction
  * Brief background
  * Research questions
  * Summary of contribution
* Methodology
  * Describe in detail about the methodology
* Description 
  * (E.g. The National Longitudinal Study of Adolescent Health (add_health) is a representative school-based 
  survey of adolescents in grade 7-12 in the United States. The Wave 1 survey focuses on factors that may influence 
  adolescents' health and risk behaviors, including traits, families, friendships, romantic relationships, peer groups,
  schools, neighborhoods, and communities)
* Results
  * Present in detail about the research findings
* Discussion
  * Optional: Impact, limitation, and future work
* Conclusion
  * Summarize research and findings