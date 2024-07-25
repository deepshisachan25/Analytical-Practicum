# North Point Software Listing Company
### Introduction
North-Point Software Production Company, specializing in games and educational software, joined a consortium to expand its customer base. The consortium offers access to over 5 million potential customers, with North-Point contributing 200,000 names. To evaluate this collaboration, North-Point conducted a test by randomly selecting 20,000 names, yielding 1,065 customers and a 5.3% response rate.

To improve the efficiency of their machine learning models, North-Point used a stratified sample, ensuring equal representation of purchasers and non-purchasers. This stratified dataset included 1,000 purchasers and 1,000 non-purchasers, resulting in a balanced response rate of 50%.

### Business Problem
North-Point Software Production Company aims to identify and target the top 180,000 potential customers from a pool of 5 million to send promotional emails. Each booklet costs $2 for printing, postage, and mailing, totaling around $4 million for all customers. The challenge is to maximize the efficiency of this marketing campaign by accurately identifying the most likely purchasers while managing costs.

The business goal is to effectively select 180,000 potential customers from a pool of 5 million who are most likely to make a purchase, enabling the company to send targeted promotional emails. With the cost of sending each booklet at $2 (including printing, postage, and mailing), the total expense would be approximately $4 million if booklets were sent to all potential customers. Thus, the primary challenge is to accurately identify and target the right customers to optimize marketing expenditures and enhance the return on investment.

### Business Goal
•	The goal is to identify 180,000 customers from a pool of 5 million who have the highest potential to buy at least one product and are willing to spend more.
•	Second, the company's objective is to enhance profits by targeting customers who are most likely to make a purchase and generate higher sales value.

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

### BUILDING MACHINE LEARNING MODEL 
#### CLASSIFICATION MODELS
classification models will be built to classify customers as purchasers or non-purchasers. For predicting purchase, Purchase is an outcome variable and will deselect Spending variable in our predictor classifier models because Spending variable is come out of purchase variables.
For classification model, important class is kept as customer being a Purchaser =1 because business wants to identify those potential customers who will make a purchase.

•	Logistic Regression
•	Naïve Bayes
•	Classification tree
•	Random forest

##### 	Logistic Regression
Using the logistic regression model, will build a model to classify customers as purchasers or non-purchasers. Positive coefficient on predictors is associated with a higher probability of customer making a purchase. Eg: Customer who order from web.order are more likely to make a purchase. Source_a, Source_u is associated with purchasing a product.  Address_is_res has a negative coefficient which means that residential address customer is less likely to make a purchase. When model performance was evaluated on validation dataset, it gave the balanced accuracy of 79.50 and with sensitivity of 73.83% which tells that model can accurately identify the positive class (here customer is a Purchaser). 
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/9c29ccaf-bcfa-4639-916a-1bd6cac75be6)
###### Goal: 
This model was built to classify customers as purchasers or not. From the output of the model can find the significant variables of this model. Will predict the purchase probability of a customer and based on threshold value will classify which customer will be a purchaser and not. Based on this classification, will suggest North Point to target those customers with higher predicted probability who are likely to make a purchase. And can retrieve customers list with important variables that are making huge impact on purchase.

##### Naïve Bayes Model
Using the Naïve Bayes model will be predicting whether a customer will purchase a product or not using different predictors. Below is the output of the model. Naïve Model takes both categorical and numerical input variables. 
###### Goal: 
The purpose of building this model was to classify new customers as purchasers or non-purchasers by estimating the conditional probability of each predictor individually against each class. Model will find the conditional probability of customer being a purchaser or not. Using this model, the company will make decisions on where to invest money by sending a mailer and at the same time keeping an eye on cost.

##### Classification Tree
A decision tree is one of the most powerful tools of supervised learning algorithms used for both classification and regression tasks. Using the decision tree, the model can predict whether a customer will purchase a product or not and also how much customer will spend. Below is the plot of classification trees for classifying customers as purchaser or non-purchaser. This tells that if freq will be less than 1 then chances are customers will not purchase. Using these classification rules, user can select the chances of customer being a purchaser or not based on different features at different conditions like Freq, Web.order, source_h, source_u, source_c, X1st_update_days_ago, last_update_days_ago. 
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/c05cdcd8-b294-4f9c-96b7-60a60c5b471b)

###### Goal: 
Using this model can find the important predictors of this model which can classify new record into purchaser or not a purchaser using classification tree. Important variables were Freq, Web.Order, Source_h, source_u, X1st_update_days_ago. Based on important features can pick a list of customers based on classification rules that lead to customer being a purchaser.

##### Random Forest
Another classification model- random forest was built to find prospect customers as purchaser or not, which was fitted on training dataset and will check model performance on validation dataset.
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/9c0590d1-16b9-4b01-bb90-f96171d3bad3)
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/499b709d-0f71-4e12-bea2-e28a747d76c9) (Fig 4.2)

###### Goal:
The plot above of Random Forest tells the important variables that play major roles in the model. Variables are Freq, X1st update days ago, last_update_days_ago, source_h, Web.order, and source_u. But it does not show how that variable is having an impact on the model positive or negative. Using the important features of our model can improve the model performance by keeping only the important variables of our model. 

#### Model Development for finding Purchaser's Spending
For predicting spending variables, built several machine learning models to predict how much a customer will spend and will choose the best model among them. 
To predict customer spending, several machine learning models were built and evaluated to select the best one:

•	Linear regression model
•	Stepwise regression model
•	Regression tree model
##### Linear regression model
Using the lm() function, a linear regression model was built on the training dataset and evaluated on a validation dataset.
Key findings from the linear regression model:
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/c51b9854-4f88-4658-be27-f8b7d313fd25)
Frequency: For every 1% increase in Frequency, there is a 79% increase in Spending.
Significant Variables: Frequency (positive impact), Address_is_res (negative impact), and last_update_days_ago (negative impact).
These results suggest that Frequency boosts spending, while residential addresses and longer durations since the last update reduce spending.

##### 	Stepwise Backward Regression
Stepwise Backward Regression was used to optimize a predictive model for customer spending, reducing the AIC from 4099 to 4074, indicating a better fit with fewer predictors. The final model identified US, source_w, last_update_days_ago, Address_is_res, and Freq as key variables.
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/573b6642-82b7-4501-9371-2fafb425ae4a)

Key findings:
Positive Impact on Spending:
Source_w: A unit increase in source_w increases spending by 49.13 units.
Freq: Higher frequency correlates with higher spending.
Negative Impact on Spending:
US: Being in the US decreases spending by 34.23 units.
Businesses can use these insights to target customer segments more effectively, focusing on those residing at residential addresses (Address_is_res = 1) and those frequently interacting with the business.

##### Regression Tree Model
Using rpart () function in R, regression tree model was built on training spend data and its performance was measure on validation dataset of spending. The model will predict how much amount customer will spend on the game and educational software.
Regression models attempt to determine the relationship between one dependent variable and a series of independent variables that split off from the initial data set. 
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/8ebea2a1-8c1c-4888-9982-35d37cf6e54b)
Goal: Can find that Freq, last_update_days_ago, US and Address_is_res are important features of regression tree that will help in predicting spending value.


 ### MODEL PERFORMANCE

Model performance was checked on validation dataset for both purchase and spending outcome variable. Based on the classification models which was built for Purchase, the Logistic regression is the best model among all classification models because it gave the highest value of Sensitivity, Specificity and Accuracy of the Logistic regression model.
For Purchase Outcome Variable (Categorical)
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/b25017e7-90c2-43fe-b0d9-ca74fb432b31)
Decision: Seeing the performance of the model on validation set can find the logistic model is working good in classifying customer as purchaser or non-purchaser as it is giving a low Misclassification rate = 0.211. Which also tells us that logistic regression model can predict more accurately whether a customer will be a purchaser or not. 

For Spending Outcome Variable, Model Evaluation is done on validation dataset based on the MAE, RMSE and correlation between actual and predicted value of spending. Seeing the accuracy measures like MAE & RMSE which is comparatively lower than other model and correlation between actual and predicted value is highest than other, so stepwise regression model was selected as it works best for predicting Spending amount by a customer.
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/634ebb92-1561-48f6-bb47-a91374060606)

#### Holdout Accuracy of Selected Models
After checking the model performance on validation dataset for both purchase and spending outcome variable and selected the best model for predicting Purchase and Spending. Now, will compute accuracy of selected model - Logistic Regression and stepwise models on holdout set. 
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/de5a6ecd-b2a5-457e-80be-1a4132396ffc)
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/3cc361ae-d6f9-47de-88e9-c5b5030ebc32)

Using these selected models, we can identify the customer who will not buy the product. So, this information will be highly useful as business will not target that group of customers (by sending email) who will not buy the product and those who will buy the gaming product business can predict the spending amount that purchaser will spend on product.
Logistic Regression & Linear Regression is easy to explain to business as well because the output of these models is a mathematical equation and the coefficients for each of the variables provide the size and impact (positive and negative) on the outcome variables. Stepwise model decreases the variables and keep only important variable that make more impact on linear regression model, overall making it less complicated. So, the stepwise selection reduced the complexity of the model without compromising its accuracy. 

A new column was added as “Adjusted_Prob_Purchase” by multiplying each case predicted probability by 0.1065 which is an original purchase rate to adjust for oversampling the purchase. Then will multiply adjusted purchase probability to spending predicted value to get expected spending done by the customer.
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/318882ec-6831-4899-8267-52e4ef5987b1)
Then calculated the sum of expected spending for holdout dataset which has over 500 records & will divide by 500 to get expected spending per customer (mean). To predict overall profit on whole customer, multiplied 180,000 on individual expected spending. Will get an estimate total of 1733486. Since our expense per customer in mailing, printing, etc. goes around $2. For overall 180000 customers, expense will be 360000. By subtracting 360,000 from 1,733,486 will get Gross profit of $ 1,373,486. Seeing this value, data analyst can tell the company that this gross profit is based on this assumption that sample data resembles overall data.
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/20e86546-ecfe-4647-a029-32f0fe28e5e6)

#### Gross profit estimation: 
Estimated gross profit will be $ 1,373,486. This tells that the firm could expect gross profit of $ 1,373,486 from the remaining 180,000 customers by selecting customer lists randomly from the pool. 

#### Recommendation:
Instead of pulling 180,000 customers in one go, business can keep picking batches of 20,000 and improve the model performance based on response rate every time they observe which will help them in picking good customer list. Hence, North point Company using predictive modelling can maximize its profit by targeting potential customers by sending emails and based on response rate can modify their models and pick good customer list. 
![image](https://github.com/deepshisachan25/Analytical-Practicum/assets/170371796/b93fa662-a5d7-4665-9be1-d69b4acc60d8)

