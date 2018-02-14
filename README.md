# Identifying Phishing URLs using Naive Bayes

## Motivation

As a part of countering frauds in online platforms, a lot of work has been done in tackling various problems problem like detecting rumors, identifying fake/sybil accounts. Phishing URLs are a very old yet active technique to steal personal information such as names, passwords, social security etc. which would in turn be used in illegal activities. It has been a menace as it causes both monetary and mental loss for the victims. The modus operandi for this is to send the victim an email or a link in messaging platforms that looks to be non-conspicuous.

This is done either by posing as a legitimate URL with a slight non-visible modification to it which may not be caught by an non-observant eye. For instance, a recent message form circulated in messaging platforms trying to imitate Etihad Airline website. Here is the message that was forwarded.

```
Etihad Airways is giving away 2 Free Tickets to
celebrate its 25th anniversary. Get your free tickets at
http://www.etihạd.com/  .
```

A keen observer would notice that the "a" in the URL is not a normal "a" but an accented one "ạ". If this isn't noticed then anyone clicking on this URL will be a potential victim. The actual website for Etihad Airways is `http://www.etihad.com/en/`.

Such URLs are also used to inject tracking cookies in your browser which would eventually work in stealing your personal information. These cookies would be automatically downloaded without the user's knowledge. In order to avoid being falsified by some random hacker, big companies are striving to identify such URLs which, if not caught, will lead to big losses.


## Problem description

Abstracting from the core machine learning problem from the given input of labelled URLs, the task is to learn and predict which of them are phishing URLs.

## Dataset

I have considered the following dataset published in 2015 which is available for download and build a machine learning model to detect phishing URLs.

Dataset: [https://archive.ics.uci.edu/ml/datasets/phishing+websites](https://archive.ics.uci.edu/ml/datasets/phishing+websites)

I have cleaned the `.arff` file to a txt format for easy processing in python.

## Environment

The algorithm has been coded in Python 2. Dependencies are common ML packages such as `numpy` and `sklearn`.

## Implementation

The model is learnt using Naive Bayes approach considering all features. Since we do not have an exhaustive set of values for each feature, the training phase is not considering smoothing of values. The model is stored as a json file using the internal json tool.

## Running the code

You can run the code using the following command.

```
python phishing.py training_file [option] [value]
```

`option` - Represents how to calculate accuracy, different values are

- `random`
- `cv`

`value`

- % train-test split for `random`
- number of folds for `cv`

## Results

The accuracy on `random` split is **90.2%**. The average accuracy on `cv` over 5 folds is **90.7%**. We can see the approach works 90% of the time.

## Future work

- More analysis on the dataset and features needs to be done to improve on accuracy.
- Comparison with state of the art approaches.
- Will feature selection help?
