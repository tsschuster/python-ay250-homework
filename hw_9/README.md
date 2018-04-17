AY250 Homework 9

Problem statements are in hw9_bayes_pymc.pdf
Supplied data is in location_data_hw9_2018.csv
Problem solutions are in hw_9.ipynb

The histograms shown in hw_9.ipynb are the Monte Carlo estimations for the (unnormalized) posterior distributions of the red & blue object speeds. Note that when the speeds are restricted to be identical the two histograms are identical and thus redundant.

c) The Red and Blue object speeds estimated in part a) were within ~1 standard deviation of each other, so restricting them to be identical does not seem inconsistent with the data. We see that the width of the speed posterior distributions is smaller when assuming the same speed, indicating that our estimation has become more certain (a result of having effectively twice the number of data points, for both red and blue particles, and those two data sets not contradicting each other).

The 5-95% confidence interval for crossing zero is only very slightly narrowed by the same speed restriction.

d) The uncertainty in both speeds rises when only examining the first 100 data points - this is not surprising since we expect uncertainty to decrease with increased data. However, the mean estimates for the Red and Blue speeds decreased as well. This may be noise as the change ~ -. 000015 for both Red and Blue is comparable to the standard deviation ~ .000028. If not noise it may indicate some nonlinear relationship between position and time (i.e. acceleration).
