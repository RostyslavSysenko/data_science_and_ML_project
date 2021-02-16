# Task Proposal: Predicting the level of wealth for NSW residents based on their postcode and determining what can improve an areas wealth

## Background
Wealth creation is a process that can requires different characteristics and habits depedning on the environemnt the person is in. Applying general advice is not always a good idea, especially when it comes to wealth creation. This project aims to solve this problem for all NSW residents and create a uniquely tailored advice on how to become wealthy within the state of NSW based on characteristics of wealthy NSW residents. This advice will include habits/patterns that an individual should adopt to become financially sucesfull in a state of NSW to help NSW residents avoid frustration and wastage of time on pursuit of their ambitious financial goal. 

## Focus Question & Aim 
Can we find relationships and correlations of particular attributes of residents of Australia and overall determine what someones average wealth would be based on their postcode? In this project, we analyze all NSW postcodes and rank postcodes as according to the wealth levels of their average resident. Then all of the postcodes are analysed according to various metrics to deterimene the characteristics of that postcode. The created model will allow NSW residents to see whether they are residing in areas that are promoting growth in wealth, and why they residents of that postcode are experiencing such financial growth. When we discover what attributes are directly related to residents wealth levels within these postcodes, it may be possible to examine an area, and suggest what can be done to increase the level of wealth within that community based off of these findings. This model is designed so that NSW residents can obtain quick evaluation of where they are at and what they can improve to get to where they want to be financially.
 
 ## Data sets 
All datasets are available in either csv or excel formats and thus are easy to use.

### Below are the datasets to be used:
- Income levels of Australians by postcode (Source: ATO)
- Population for Australian Suburbs (Source: #####)
- NSW speeding tickets dataset by postcode (Source: Department of Customer Service NSW)
- Rent information by postcode (Source: NSW Department of Family and Community Services)
- House sale information by postcode (Source: NSW Department of Family and Community Services)

### To make those datasets usable, below is list of things to be done:
- All datasets for each model will need to be combined on a certain attribute
- All the unnecessary attributes to be dropped to aid analysis and allow for simplification and abstraction
- All rows with null attributes might need to be removed.
- All datasets have to be filtered to only contain NSW data
-  Outliers might need to be removed which are not representative of the general statistical behaviors of other data points.
 
 ## Project Plan & Milestones (brief version)
 ### Milestone 1
 - complete data exploration, data cleaning and all datasets are combined and correlations are discovered between attribtues
 - Sucess Criteria: Reasanoble correlations are discovered and all datasets are properly integrated. 
 - Deadline: end of mid term holiday
 ### Milestone 2
 - baseline model is created
 - Sucess Criteria: the model is fully functional and reasonable when use on new data. Also the baseline model is fully optimised. 
 - Deadline: end of mid term holifday 
 ### Milestone 3
 - advanced models are created
 - Sucess Criteria: advanced models are superior in performnance to baseline models and advanced models are fully optimised
 - Deadline: week 11
 ### Milestone 4
 - presentation is finalised
 - SucessCriteria: GitHub commits look clean, it is easy to replicate and follow though the project, project is cleanly, succinctly and effectively presented in github. Also, presentation is prepared.
 - Deadline: week 12

 
 ## Criteria of success
 - In total there should be 10 statistical techniques used
 - Relationships for 3-5 attributes are identified.
 - Project is well presented, and analysis is clear.
 - Project plan is fully implemented, and the object can be easily replicated

## Attributes of Interest
1.    Attributes that will categorise postcodes into wealth brakets (to allow for ranking of the suburbs by wealth)
    - Average Income from all sources for a suburb
    - Average super account balance
2.    Types of income and their significance (to determine which sources of income make NSW residents wealthy)
    - Percentage of income as business income on average
    - Percentage of income as rental income on average
3.    Population demographics #1 (to determine the size enironment in which wealthy individuals are more likely to be created)
    - Population of the suburb
4.    Population demographic #2 (used to predict wealth based on whether the person has a habuit of submitting tax return)
    - Number of people lodging income tax return 
    - Population of the suburb
5.    Number of speeding tickets per suburb (used to determine wherther wealthy individuals are likely to be risk takers or reckless)
    - Number of speed tickets issued
6.    Number of renters vs non-renters per suburb (used to predict whether wehalty people are more likely to rent and what percentage of inome they spend on their rent)
    - Population of the suburb
    - Number of people lodging income tax return 
    - New Bonds Lodged
    - Total Bonds Held
7.    Cost of the housing (used to predict whether wehalty people are more likely to purchase more expensive houses and what percantage of their income they spend on the house)
    - Mean house sale price
    8.    Change of price of the housing (used to predict whether the wealth of individual is related to change in price of their property and weather their property growth in value. This is indicativive of good invetment habits)
    - Annual change in mean price
9.    Liquidity of the housing market (used to predict whether the wealth of individual is based on number of house sales in an area which potentially implies the liquidity of the asset)
    - House sale numbers
    
## Techniques to be used
- Regression model as a baseline model
- Logistic regression
- Neural networks and other models covered past week 6

## Prior work
 - Rough predictive estimations based on multiple assumptions and basic statistical techniques


