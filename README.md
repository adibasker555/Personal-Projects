# PersonalProjects

Personal Projects Descriptions:

*MIGHT TAKE 2-3 RELOAD ATTEMPTS TO RENDER CERTAIN JUPYTER NOTEBOOK FILES*

1) 'MarchMadness' folder: analysis of Kaggle's college basketball dataset for the March Madness tournament. Using the data provided, I calculated a list of relevant features that would be important in deciding which team wins a given matchup, such as points per game throughout the season, assists, rebounds, steals, overall winning record, etc. I then trained different supervised ML algorithms such as Logistic Regression and Random Forest Classifiers to predict March Madness outcomes, which had roughly 70-75% accuracy. I gauged relative performance using validation accuracy, type 1 error rate, and other values in the confusion matrix. To see how college basketball changes across years, I trained the model on different years of data and tested on 2018 data, to compare the the relative accuracy of the model.

  In the second file, I created a non ML-based probabilistic model to assess win probability. Given a matchup of 2 teams, the current game   score, and the points scored by each team until then, the model outputs the probability of each team winning that matchup.

2) 'StockApplication' folder: created a stock market neural network that predicts stock price movements based on a set of features. The data collection aspect involved calculating 30, 90, and 200 day moving averages, sector movements (healthcare, technology, etc.), index movements (NYSE, S&P 500, etc), etc. This involved using webscraping technologies like Selenium and Beatifulsoup to access dynamic webpages. I then trained a traditional nueral network and recurrent neural network on the accumulated data. Condensed version of collected data (without sentiment analysis, which wasn't used in the neural net) is shown in 'condensed_data.csv'

3) 'InternshipProject' folder: This was research done during my internship that I was allowed to share on my profile. The files shown are functional, early-stage versions of the end product, and they were updated significantly as the internship progressed. The goal of the summer project was to do machine learning based document segmentation on birth certificates that were sent by clients to the company (in order to verify dependents for healthcare benefits administration). The project was to try to identify document labels (such as 'child name', 'date of birth', etc.) by running each certificate through a supervised ML model. I created a set of software tools uses image processing librarise libraries to compile a list of relevant differentiating features to distinguish labels from other words in a birth certificate. In order to calculate these features, Google's OCR (optical character recognition) and openCV computer vision softwares were used to parse the certificates. I then trained a traditional nueral network to isolate label information from other words on the certificate.

4) 'EducationStatistics' folder:  is a project done for Berkeley's Data Science Society. It consists of simple Exploratory Data Analysis that compares educational statistics across different developed and non-developed countries across the world.

5) 'NBADataAnalysis' folder: part of a project done for Sports Analytics Group at Berkeley, consists of data analysis and visualization of Lebron James's player data throughout his career.
