# Credit_Card_Classification_ML
## Overview: 
this repository contain the Credit Card Classification System, which use machine learning techique to help the bank to recoganise the customer creditworthines i.e. will customer able to repay loan amount in future and so on

## STEPS
### 1.Data collection:
                       i collected data from Kaggle and I used KAGGLE API to access the data because data size was large that causing the issue on google colab uploading data option

### 2. Data Preprocessing : 
a. This data is so messy. i found diffuclty to clean this data because almost every column need to be cleanded
b. i did cleaning process two part 1. Numeric cleaning 2. categorical cleaning 
c. then concat both part and make single dataframe 

Challeges : i found some column cleaning too hard even the" type of loan" column took me almost 6 hour to figure it out i tried a lot of combination to cleaned that one sum them up for final code (yes this is the column where i mostly relied on ChatGpt ).
some were easy to clean 
To be Honest My 65 % of my time was spent on that part 

### 3 . Model training and Evaluation model performace:
       i tried 3-4 model on that and figured it out that Randomforest working well

### 4 . Local Deployment:
    make flask app with simple format to predict about data















<!-- first create venv and then clone the git repo(which have read.me and .gitignore) to your local machine and now check you are in repository directory -->
<!-- So today i activate myevn which is different from yesteday one (venv). Because as list down all activate environment and find that one -->
