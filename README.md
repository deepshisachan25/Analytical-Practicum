# Analytical-Practicum- Machine Learning
## North Point Software Listing Company
### Introduction
North-Point Software Production Company specializes in the sale of games and educational software. They joined a consortium to broaden their customer base. This consortium provides access to a combined list of over 5 million potential customers, with North-Point contributing 200,000 names to the pool. Subsequently, North-Point conducted a test by randomly selecting 20,000 names to explore the potential of this collaboration. From this test, 1,065 customers emerged, resulting in a response rate of 5.3%. To enhance the efficiency of the machine learning models, a decision was made to utilize a stratified sample, ensuring an equal representation of both purchasers and non-purchasers. With an overall dataset comprising 1,000 purchasers and 1,000 non-purchasers, the response rate was observed at 50%.
### Business Problem
The business goal is to pick the list of 180,000 customers from a pool of 5 million who have the highest potential of making a purchase so that the company can send emails. It was found that sending each booklet costs $2 for things like printing, postage, and mailing (expense). In total, it will cost around $4 million to send booklets to all customers. Even though there is a lot of data, the main problem will be figuring out who the potential customers are while keeping an eye on the costs. The primary challenge lies in identifying potential customers (180,000) out of 5 million who are most likely to make a purchase so that the company can send emails to those customers.
### Business Goal
•	The goal is to pick the customers who have the highest potential of buying at least one product and are willing to pay more.
•	Second, the company's objective is to enhance its profits by retrieving a list of customers who are highly likely to make a purchase and generate higher sale value.
### Analytical Approach 
The business goal is to make more money by creating two predictive models. 
•	One will help find customers who are likely to buy something, so business can focus on them. These models are important for targeting the right customers, making it more likely that they'll buy something and increasing the company's profit, which is the goal of the company.
  • Create a classification model to classify customers as Purchaser and Non-Purchaser and identify important variables that leads to purchase. PURCHASE – Outcome variable
•	Other machine learning model will help find out the approximate amount of money customers are willing to spend in buying a gaming product which will help business in identifying the estimated revenue.
  • Create a regression model to predict the amount of sale of the purchaser and ultimately calculate expected spending by potential customer & identifying estimated profit.

### DATA EXPLORATION
North Point dataset is imported on R for data analysis. Exploring and analysis the data is a crucial step to uncover patterns within the dataset. Dataset contained 2000 rows and 25 variables. In data exploration, the number of rows and columns was analyzed, categorical columns were checked and then found the summary statistics of each variable.
 ![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/d89d694a-4423-43fc-af1f-766a3dd33413)
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/17c025ea-4e3c-4962-b336-6ddf4d4c3055)
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/652e5d73-77f6-4067-bd5f-02917a66ca22)

### PREDICTOR ANALYSIS AND RELEVANCY
Out of 25 variables here, “Purchase” and “Spending” are the outcome/ target variables which are dependent on predictor variable, which indicates whether the customer has made a purchase and if made a purchase then how much amount spent respectively. And the rest 23 variables are predictor/ independent variables. 
Correlation between the variables was checked by correlation matrix and corr plot. Variables are strongly correlated if correlation is between 0.5 to 1 (positive or negative).

![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/84f00ff5-b374-48f1-90fc-495b9dc487a2)
There are a few variables that are strongly correlated. Some examples are:
•	Source_w is negatively correlated with last_update_days_ago and 1st_update_days_ago. This would signify that when Source _w is higher (1 in this case) then the last_update_days_ago and 1st_update_days_ago have relatively smaller value (signifying that the data was updated more recently)
•	Freq & last_update_days_ago are negatively corelated which signifies that smaller the value of last_update_days_ago, higher the frequency. This would mean that records that were recently updated had a higher frequency of purchases.
•	Freq and Spending have strong correlations.

#### Histogram of Purchase variable
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/d40e9e57-9fc6-4326-a416-c74168a88c0a)
The customers who purchased or not purchased are equally distributed which was already known. X- axis represents the ‘Purchase’ variable indicating whether a customer made a purchase or not, while the y-axis shows the count. 

#### Histogram of Spending variable
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/1eb64497-84e3-4f6f-ac6b-7e9733830aaa)
(Fig 2.1) The distribution of spending where ‘purchase = 1’(through mailing) provides a visual representation of spending among customers who made a purchase, where x-axis spending indicates amount spent and y-axis shows the count.

![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/97a92dd6-defd-4a40-846a-e2580a99c43d)
(Fig 2.2)
This shows the number of purchases made by the customer. Most of the customers purchased 1 and 2 items. 

![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/6a383972-e0cd-46bf-9b8d-a693e5e33266)
(Fig 2.3) This shows most of the customers are from United states and very few are from other countries.

![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/5529da92-4204-4301-94f2-a361d41b4e54)
(Fig 2.4)

![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/9c21e370-ac4c-47b3-89ed-bdb84996eb3f)
(Fig 2.5)
Data for last_update_days_ago and 1st_ update_days_ago shows many of customer information have not been updated in a while. 

![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/73b04d65-df53-4259-b5cf-9e4f490ae0dd)
(Fig 2.6) Not many purchases were made through web order when compared to other. 

![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/cf7829e3-fcdb-4c50-b623-dc123fb1e6c7)
(Fig 2.7) Above scatterplot shows the number of purchases made by the customer in relation to spending.

![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/176824f9-aafc-41d8-b2a9-0071f1e36d4c)
(Fig 2.8) Not many customers’ addresses are residential.

![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/7ee20a4d-0b3a-47f4-af05-a0da1e164335)
(Fig 2.9) This shows that there are more male than female.

### DIMENSION REDUCTION
PCA (Principal Component Analysis) was performed to check the need of dimension reduction and then visualized the results. It was found that PCA gives the PCs for all 24 variables, so no dimension reduction is required. Hence, dimension reduction is not required in our datasets. Small data has less requirements on data. In this case, it seems that the cumulative proportion increases steadily across the first few PCs which suggests that dimension reduction might not be necessary.

![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/c69f7a3b-4c51-487d-842c-e6ec423c66e1)
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/7b691e4c-944f-4381-a26d-4a8872c22087)

### DATA PARTITIONING
Using R, model will be fitted on the training dataset, then model is evaluated on the validation dataset. This step helps to fine-tune the model & access its performance. Holdout dataset is reserved to assess the final performance of the model after training and validation is done. It is used to evaluate how well the model performs on new, unseen data. It gives an estimate of how well the model will work in real-life situations.
Set.seed() will be used to get the same partition while re-running the R code. Here, in North-Point dataset, dataset was randomly sampled the 40% of rows for training (800 records), 35% of rows for validation (700 records) and remaining 25% for holdout datasets (500 records). Dataset were created as training (train.df), valid(valid.df) and holdout (holdout.df) dataset for purchase prediction.
Partitioning is done so that the model can predict how many customers will purchase a product so that business can select the list of potential customers and target those customers by sending emails. Here, partitioning is done on training, validation, and holdout dataset on 2000 records where purchase and non-purchase ratio is 1:1 then will fit the model on train dataset and predict Purchase on test dataset whether customer will purchase or not. 
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/fda0d237-dd04-400b-aa90-86bc4e4ace77)
Then for regression model, created training (train_spend) and validation (valid_spend) dataset using train and validation dataset of Purchase (train.df) and (valid.df) by filtering in purchase= 1. This step is done for building the model to predict spending among the purchasers. 	
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/ae35c470-6a52-42b6-abe3-4594937d7554)

### MACHINE LEARNING MODEL BUILDING
#### CLASSIFICATION MODELS
classification models will be built to classify customers as purchasers or non-purchasers. For predicting purchase, Purchase is an outcome variable and will deselect Spending variable in our predictor classifier models because Spending variable is come out of purchase variables.
For classification model, important class is kept as customer being a Purchaser =1 because business wants to identify those potential customers who will make a purchase.
•	Logistic Regression
•	Naïve Bayes
•	Classification tree
•	Random forest

##### 	Logistic Regression



