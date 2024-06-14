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
