---
title: Time series forecasting with Prophet and fast.ai
subtitle: Using deep learning to improve on additive models
image: /img/13_timeseries/time1.png
date: 2019-01-01 01:50:30
---

I tested several approaches for forecasting multivariable time series, using a database of air pollution and weather in Beijing.  I created a baseline using the additive forecasting tool Prophet, compared it against a neural network with no access to historical data, and tested a strategy for adding historical data while preventing information leakage from the future.

# Smog and weather
Forecasting works best when you have at least a few relevant cycles of feature variation to observe.  For this project I looked at weather and pollution data, using the [Beijing PM2.5 Data Data Set](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data) from the UC Irvine data repository. This is an hourly data set of pollution (PM2.5 particles) from the US Embassy in Beijing, along with meteorological data from Beijing Capital International Airport. Data spans  Jan 1st, 2010 to Dec 31st, 2014.  I chose this dataset because weather varies mostly in an annual cycle, and this dataset has four years of detailed (hourly) data that I could use.  This dataset also contained six weather features that could be fed into potential models, some of them highly seasonal (especially temperature).  Some of the pollution data was missing, but there were no nulls among the weather data. 

```python
# The weather features and their descriptive names, 
# as used throughout the project
feat_name_dict = {
        'pm2.5':'Pollution (pm2.5)',
        'dew_pt':'Dew point',
        'temp': 'Temperature',
        'pres': 'Pressure',
        'wind_dir':'Wind Direction',
        'wind_spd':'Wind speed',
        'hours_snow':'Cumulative hours of snow',
        'hours_rain':'Cumulative hours of rain'
                 }
```
For my first experiments, I selected a subset of the data spanning just over a year.  I defined a training set with data from January 2010 to March 2011, and a validation set with just Apr2011.  As you can see by the straight downward slope in the second week of april, I had to patch some holes in the pollution data with linear interpolation (first subplot below).

![time](/img/13_timeseries/time3.png)

The plots below show how some of these features are markedly seasonal.  In these autocorrelation plots, a variable with high seasonality will show sweeping curves far outside the narrow grey confidence intervals in the middle.  Pollution doesn't autocorrelate much beyond the last few hours, but temperature shows wide curves that stretch throughout the year.  As we'd expect, daily temperature correlates the most with the immediate past and with the conditions one year ago.  Dew point and pressure show a similar pattern, though note that the seasonality in temperature could show up in the other graphs if those weather variables are very temperature-dependent.  That is what I expect of dew point, so I bet that temperature is driving most of the seasonality in this dataset.

# Capturing seasonality

There are [many, many ways](https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/) to capture the cyclic patterns in an autocorrelated dataset.  The [Statsmodels Time Series Analysis library](http://www.statsmodels.org/dev/tsa.html), in particular, contains useful implementations of many models of increasing complexity, from univariate autoregressive models (AR), vector autoregressive models (VAR), and univariate autoregressive moving average models (ARMA) to non-linear models such as Markov switching dynamic regression and autoregression.

For this project, though, I decided to use [Prophet](https://facebook.github.io/prophet/). This is an open-source forecasting tool developed by Facebook, and it provides a particularly user-friendly interface for a variety of additive forecasting models.  Prophet finds the seasonal trends in historical values of a variable, and uses them to build a linear regression model for predicting future values.  Prophet can even create multivariable regressive models, with the limitation that the other variables have to be known into the future.  Data about time (holidays, days of the week, etc) are particularly easy to know about the future, so they should usually be added as extra regressors for Prophet to profit from.  I fed it historical temperature data and a list of Chinese holidays, and this is what the model foretold:

![time](/img/13_timeseries/time5.png)

A particularly insightful feature of Prophet is showing you a decomposed view of the seasonal trends found in the data.

![time](/img/13_timeseries/time6.png)

Here is a zoomed-in view of the tail end of the training data (blue) and all the validation data (red), along with Prophet's forecast and confidence intervals (orange).

![time](/img/13_timeseries/time7.png)

In contrast to the beautiful forecast for temperature above, Prophet was not able to handle the binary nature of my one-hot-encoded variables for wind direction.  It could only produce an uncertain floating-point prediction, so I dropped this variable from the dataset.  Most other variables showed variation way beyond Prophet's relatively conservative estimates.

![time](/img/13_timeseries/time8.png)

At this point, I had an idea.  Several of the weather variables are probably dependent on temperature—especially dew point, and probably pressure; maybe also pollution?  If Prophet can use regressors that extend into the future (like holidays)... would it be possible to first forecast temperature into the future, and then use that prediction as an extra regressor for all the others, thereby pulling the overall prediction up by its bootstraps?

Sadly, no.

![time](/img/13_timeseries/time9.png)

And so, in the end this is the best prediction for pollution that Prophet could generate.  It is cyclical, but otherwise far removed from the actual changes in pollution even at the very beginning of the forecast.  This was a mild surprise, as I expected that its rolling-window RMSE (dark green line) would get higher as it moved further and further away from the last historical data point.

![time](/img/13_timeseries/time10.png)

# A neural network, blind to history
Several recent Kaggle competitions have seen great success ([kaggle](https://www.kaggle.com/dromosys/fastai-rossmann-store-sales-v3), [arXiv](https://arxiv.org/abs/1604.06737)) from a model that is almost too simple: a relatively shallow neural network observing tabular data, with access to information that you can get from dates (day of the week, whether it's a weekday, etc) but no access to historical data.  It represents an approach entirely different from what Prophet does, and is nearly as easy to set up.  This is what the network looks like:

```
TabularModel(
  (embeds): ModuleList(
    (0): Embedding(3, 1)
    (1): Embedding(3, 1)
    (2): Embedding(13, 6)
    (3): Embedding(54, 26)
    (4): Embedding(32, 15)
    (5): Embedding(8, 3)
    (6): Embedding(366, 50)
    (7): Embedding(3, 1)
    (8): Embedding(3, 1)
    (9): Embedding(3, 1)
    (10): Embedding(3, 1)
    (11): Embedding(3, 1)
    (12): Embedding(3, 1)
    (13): Embedding(25, 12)
  )
  (emb_drop): Dropout(p=0.04, inplace=False)
  (bn_cont): BatchNorm1d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layers): Sequential(
    (0): Linear(in_features=127, out_features=1000, bias=True)
    (1): ReLU(inplace=True)
    (2): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.001, inplace=False)
    (4): Linear(in_features=1000, out_features=500, bias=True)
    (5): ReLU(inplace=True)
    (6): BatchNorm1d(500, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.01, inplace=False)
    (8): Linear(in_features=500, out_features=1, bias=True)
  )
)
```

I fit the model for just a few epochs, until validation error bottomed out.

![time](/img/13_timeseries/time11.png)

![time](/img/13_timeseries/time12.png)

And finally, I compared its predictions with those from Prophet. The best validation loss (overall RMSE) I was able to get was `118.24` (in the units of pm2.5).  This is clearly higher than our earlier predictions of `67.05` and `64.46` from Prophet.  As you can see below, the neural network predicts much more variation and has a much less clear cycle.  It's intriguing how the neural network actually seems to perform better forthe first half of the month or so, issuing predictions that vary boldly in the same way as the actual data (Prophet issues relatively continuous predictions with much lower variability).

![time](/img/13_timeseries/time13.png)

# But what if a neural net _could_ see history?

What I like about these neural net results is how much they reflect the seasonality of the pollution data even though the model is just a general-purpose pattern finder.  Prophet is a bundle of models explicitly designed to find autocorrelative patterns, but it didn't beat the net by _that_ much.  And a more fundamental problem is that Prophet, like other purpose-built models, is much more constrained by its design.  I couldn't find a way for Prophet to see the values of all the weather variables and use those for its predictions, even though they're very relevant.  All that the system allows you to build is a univariate regression with an awareness of holidays.  

A neural net, on the other hand, allows for creativity.  What if we gave the net an explicit awareness of history?  Specifically, for  both pollution and the six weather features we could create a time-lagged copy.  If the net can see what pollution was a day, a month, and a year ago, then it should be better able to pick up on the patterns that it was already beginning to see just from the feature data.  Much more importantly, now the net would also be able to see the weather data and how _it_ varies with time.  Domain knowledge suggests that pollution varies greatly with temperature and wind speed, so any patterns that the net sees in the weather data should make for a much stronger prediction of pollution. 

Ah, but there is a catch.

Think of the training and validation sets for a moment.  In the previous example, the model had observed only the relationship between pollution and time.  In this case, however, we've expanded to six new weather features throughout the training data.  The net will learn to predict pollution based on date features and also the current state of the weather... but we don't have weather data for the future!  (Well, we do, but I don't want the model to see it.  In most [examples](https://nbviewer.jupyter.org/github/nicolasfauchereau/Auckland_Cycling/blob/master/notebooks/Auckland_cycling_and_weather.ipynb) I found around the internet, people would still allow their model to see future values of all the other variables, or they would have a model that always predicts one day into the future and then adjusts based on what the day turned out like.  Here I'm tackling the much more interesting problem of predicting weather _and_ pollution a full month into the uncharted future.)

So we have a problem of incomplete data.  For the past, we always know both the weather and the weather's history (given by the offset variables).  The future must have the same feature structure (without nulls), but we don't know the current weather and many lag features cover only part of the period.  The validation set starts on April 1st; on April 5th, we know what the weather was like a year, a month, and a week ago, but we don't know what the weather was 1, 2, 3, or 4 days ago.  Those dates are all still in the future.

So what can we do with all these missing values?  Well, didn't we just use Prophet to forecast each of our weather features individually?  We couldn't use those forecasts to inform the one on pollution, but the data is still there and can fill in all the nulls in our validation data.  This is what it looks like:

![time](/img/13_timeseries/time14.png)

This is a decent simulation of memory itself!  Historical data is much richer than our predictions about the future, but the more we step into the future the less historical information we have.  By April 25th, the only real data we have is at least 25 days old; everything else is filled in with predictions from Prophet.

And so, we're ready for the final test.  I've switched to a larger train/test set, where the train data covers almost 4 years and the test set is November 2014.  The neural network had already performed decently when all it could see was the historical links between time and pollution, and for a much shorter time.  Now it can also see both the weather and the history of the weather.  That data is real for the training set, and a combination of real plus simulated for the validation set.  

I bundle the data up into a fast.ai learner, and start training.

![time](/img/13_timeseries/time15.png)

Huh, that's funny.  Training error plumetted right away, in less than one epoch.  By the time that the validation error started being measured, it was already really low.  I tested this a few times, and in every case a single epoch would bring the validation error all the way down.  Why is it going so fast?  Let's reduce the learning rate and see if the validation error goes any lower.

![time](/img/13_timeseries/time16.png)

Yeah, validation error got a little better, but the model basically got really good in just a moment.  Also... hold on a second.  That validation error is equivalent to an overall RMSE of ... `5.45`??  That is fantastically better than the best validation loss I was able to get before, which was `118.24`.  The best that Prophet could produce was `64.46`, back when I bootstrapped pollution predictions with temperature.  This is amazing.  In fact, it's _too_ amazing.  What does the data actually look like?

![time](/img/13_timeseries/time17.png)

No. Fricken. Way.

That green line from the neural net absolutely _hugs_ the data, all the way until the end of the month.  Not only that, do you see how it even hugs the straight line in the data around November 21st?  Those straight lines are interpolation artifacts caused by missing data, and should _not_ be predictable based on the weather.  Even worse, the model is just as perfect at the beginning of the month as at the very end, where all the recent historical data is actually mediocre predictions.  I hate to admit it, but it looks like my beloved neural network has found a way to cheat.

![time](/img/13_timeseries/time18.png)

# Epilogue
I spent quite a while trying to find the source of data leakage.  Somewhere, somehow, it must be able to see something about the future other than the Prophet predictions and my carefully censored historical data.  In the end, I decided that `perfect` would be strictly less good than `shipped`.

And so, dear reader, you now see why the real work in machine learning is not tuning the model itself, but rather designing the pipeline that feeds data to it.  Neural networks are really [creative](https://techcrunch.com/2018/12/31/this-clever-ai-hid-data-from-its-creators-to-cheat-at-its-appointed-task/) in reducing their error metric.  Learn to adopt a [security mindset](https://www.schneier.com/blog/archives/2008/03/the_security_mi_1.html), and learn to be suspicious of results that are too good to be true.  Good luck out there.

You can find the full code for this project on [GitHub](https://github.com/DanielMartinAlarcon/timeseries).
