# Project 2: Bitcoin Linear Regression: Correlation Exploration
2 February 2018



As you may know, Bitcoin is a decentralized digital currency notorious for extreme price volatility.  To tame this beast, I set out to build a multivariable linear regression model which can be used to predict the price of Bitcoin.  In this post, I will:

1. Introduce a predictive model for the price of Bitcoin
2. Explore a couple significant underlying features of the model
3. Provide my key insights and takeaways from this project


### Model Overview
The model is a standard, multivariable linear regression model. The project scope required standard linear regression rather than a time series analysis; I hope to reconfigure this model into a time series analysis at a later time.  The difference primarily relates to whether the train-test split is shuffled or sequential - this will be discussed later in this post.  

The model has identified three key features (independent variables) which are highly correlated with the price of Bitcoin (dependent variable).  Regularization was evaluated but deemed not necessary
given the already high correlation, resulting in an R2 accuracy of 96.8%.

### Feature Exploration
The goal here is to target features that have a high correlation to Bitcoin, but aren't necessarily too close to the direct price movement of Bitcoin (an example of this would be
  cryptocurrency universe market capitalization, of which the Bitcoin market capitalization represents ~30%).  Features considered include:

Bitcoin related:
- Cryptocurrency universe market capitalization
- Ethereum price
- Volume
- Number of transactions
- Average block size
- Transaction fees
- Unique addresses
- Hash rate

Market related:
- Price of Gold
- Nasdaq Composite Index

Other:
- Google search interest
- Social media sentiment analysis (via Twitter)

Most of these features were also transformed onto a natural log scale as they relate exponentially / multiplicatively, so this transformation allows a more linear, "apples to apples" relationship on this basis.  

From this set of features, I was able to narrow down to the following for use in the model:
1. log of Google search interest
2. log of Nasdaq Composite Index
3. log of Bitcoin network transaction fees

![](charts/featureexploration.png "Feature Exploration")

![](charts/OLS.png "3 Features contribute to R2 of ~97%")

![](charts/modelpairplot.png "Correlation")
