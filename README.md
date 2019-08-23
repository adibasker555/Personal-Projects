# PersonalProjects
Personal Projects

Personal Projects Descriptions:

1) The first file is a project done for Berkeley's Data Science Society. It consists of simple Exploratory Data Analysis that compares educational statistics across different developed and non-developed countries across the world.

2) The next project involves Kaggle's college basketball dataset for the March Madness tournament. Using the data provided, I calculated a list of relevant features that would be relevant in deciding which team wins a given matchup, such as points per game throughout the season, assists, rebounds, steals, overall winning record, etc. I then trained different supervised ML algorithms such as Logistic Regression and Random Forest Classifiers to predict March Madness outcomes, which had roughly 70-75% accuracy. I gauged relative performance using validation accuracy, type 1 error rate, and other values in the confusion matrix. To see how college basketball changes across years, I trained the model on different years of data and tested on 2018 data, to compare the the relative accuracy of the model.

In the second file, I created a non ML-based probabilistic model to assess win probability. Given a matchup of 2 teams, the current game score, and the points scored by each team until then, the model outputs the probability of each team winning that matchup.

3) The next project involves creating a stock market neural network that predicts stock price movements based on a set of features. The data collection aspect involved calculating 30, 90, and 200 day moving averages, sector movements (healthcare, technology, etc.), index movements (NYSE, S&P 500, etc), etc. This involved using webscraping technologies like Selenium and Beatifulsoup to access dynamic webpages. I then trained a traditional nueral network and recurrent neural network on the accumulated data.

4) This was research done during my internship that the company I interned for let me share on my profile. The goal of the summer project was to do machine learning based document segmentation on birth certificates that were sent by clients to the company (in order to verify dependents for healthcare benefits administration). The project was to try to identify document labels (such as 'child name', 'date of birth', etc.) by running each certificate through the model. In order to do this, I compiled a list of relevant differentiating features to distinguish labels for other words in a birth certificate. In order to calculate these features, Google's OCR (optical character recognition) software was used to parse the certificates. I then trained a traditional nueral network to isolate label information from other words on the certificate.
