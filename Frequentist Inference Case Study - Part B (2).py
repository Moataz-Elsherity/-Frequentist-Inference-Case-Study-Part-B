#!/usr/bin/env python
# coding: utf-8

# # Frequentist Inference Case Study - Part B

# ## Learning objectives

# Welcome to Part B of the Frequentist inference case study! The purpose of this case study is to help you apply the concepts associated with Frequentist inference in Python. In particular, you'll practice writing Python code to apply the following statistical concepts: 
# * the _z_-statistic
# * the _t_-statistic
# * the difference and relationship between the two
# * the Central Limit Theorem, including its assumptions and consequences
# * how to estimate the population mean and standard deviation from a sample
# * the concept of a sampling distribution of a test statistic, particularly for the mean
# * how to combine these concepts to calculate a confidence interval

# In the previous notebook, we used only data from a known normal distribution. **You'll now tackle real data, rather than simulated data, and answer some relevant real-world business problems using the data.**

# ## Hospital medical charges

# Imagine that a hospital has hired you as their data scientist. An administrator is working on the hospital's business operations plan and needs you to help them answer some business questions. 
# 
# In this assignment notebook, you're going to use frequentist statistical inference on a data sample to answer the questions:
# * has the hospital's revenue stream fallen below a key threshold?
# * are patients with insurance really charged different amounts than those without?
# 
# Answering that last question with a frequentist approach makes some assumptions, and requires some knowledge, about the two groups.

# We are going to use some data on medical charges obtained from [Kaggle](https://www.kaggle.com/easonlai/sample-insurance-claim-prediction-dataset). 
# 
# For the purposes of this exercise, assume the observations are the result of random sampling from our single hospital. Recall that in the previous assignment, we introduced the Central Limit Theorem (CLT), and its consequence that the distributions of sample statistics approach a normal distribution as $n$ increases. The amazing thing about this is that it applies to the sampling distributions of statistics that have been calculated from even highly non-normal distributions of data! Recall, also, that hypothesis testing is very much based on making inferences about such sample statistics. You're going to rely heavily on the CLT to apply frequentist (parametric) tests to answer the questions in this notebook.

# In[28]:


get_ipython().system('pip install scipy')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from numpy.random import seed
medical = pd.read_csv("C:/Users/moe/Desktop/dsc/Unit 7 - Data Wrangling/7.2 - Data Collection/7.2.5 - API Mini Project/targetdir/Frequentist Case Study/insurance2.csv")


# In[17]:


medical.shape


# In[18]:


medical.head()


# __Q1:__ Plot the histogram of charges and calculate the mean and standard deviation. Comment on the appropriateness of these statistics for the data.

# __A:__

# In[19]:


_ = plt.hist(medical["charges"])
_ = plt.title("Charges")
_ = plt.xlabel("Charge Amount")
_ = plt.ylabel("Count")
_ = plt.show()


# In[20]:


charges_mean = np.mean(medical["charges"])
charges_mean


# In[33]:


charges_std = np.std(medical["charges"], ddof=1)
charges_std


# In[22]:


#these measures are not appropriate since the data is heavily skewed and not normally distributed the mean of a sample means would be more appropriate same for std 


# __Q2:__ The administrator is concerned that the actual average charge has fallen below 12,000, threatening the hospital's operational model. On the assumption that these data represent a random sample of charges, how would you justify that these data allow you to answer that question? And what would be the most appropriate frequentist test, of the ones discussed so far, to apply?

# __A:__

# In[23]:


# the data is randomly sample so it represents the population and even if the data is skewed we can still find the population paramters from the sample distribution
# the most appropriate test would be hypothesis testing using the t value 


# __Q3:__ Given the nature of the administrator's concern, what is the appropriate confidence interval in this case? A ***one-sided*** or ***two-sided*** interval? (Refresh your understanding of this concept on p. 399 of the *AoS*). Calculate the critical value and the relevant 95% confidence interval for the mean, and comment on whether the administrator should be concerned.

# In[24]:


# A one sided inerval would fit best since theyre only interested in if it fell below a certain mean


# __A:__

# In[34]:


n = len(medical["charges"])
standard_error = charges_std / np.sqrt(n)
standard_error


# In[35]:


df = n - 1


# In[36]:


t_value = t.ppf(0.05, df)
t_value


# In[37]:


margin_of_error_t = t_value * standard_error
margin_of_error_t


# In[40]:


confidence_interval_lower_bound = charges_mean - margin_of_error_t
confidence_interval_lower_bound


# In[52]:


from scipy import stats


# In[43]:


#even tho this code doesnt do the exact same thing having a positive t statistic shows that we havent fell below the hypothesised mean
#and it even goes as far as telling us that there is an evidence of increase
scipy.stats.ttest_1samp(medical["charges"], 12000)


# In[ ]:


#this result suggests that the hospital has not fell below the average charge of 12,000 bcuz we are 95% sure that the true mean is at least 13,815 or more


# The administrator then wants to know whether people with insurance really are charged a different amount to those without.
# 
# __Q4:__ State the null and alternative hypothesis here. Use the _t_-test for the difference between means, where the pooled standard deviation of the two groups is given by:
# \begin{equation}
# s_p = \sqrt{\frac{(n_0 - 1)s^2_0 + (n_1 - 1)s^2_1}{n_0 + n_1 - 2}}
# \end{equation}
# 
# and the *t*-test statistic is then given by:
# 
# \begin{equation}
# t = \frac{\bar{x}_0 - \bar{x}_1}{s_p \sqrt{1/n_0 + 1/n_1}}.
# \end{equation}
# 
# (If you need some reminding of the general definition of ***t-statistic***, check out the definition on p. 404 of *AoS*). 
# 
# What assumption about the variances of the two groups are we making here?

# __A:__

# In[ ]:


#null hypothesis 
#People with insurance are charged the same amount as people without insurance 
#hypothesis
#people with insurance are NOT charged the same amount as people without insurance


# In[ ]:


#assumptions about the variance 
#We are assuming both groups have the same variance


# __Q5:__ Perform this hypothesis test both manually, using the above formulae, and then using the appropriate function from [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-tests) (hint, you're looking for a function to perform a _t_-test on two independent samples). For the manual approach, calculate the value of the test statistic and then its probability (the p-value). Verify you get the same results from both.

# __A:__ 

# In[44]:


medical[:10]


# In[47]:


with_insurance = medical[medical["insuranceclaim"] == 1]


# In[48]:


without_insurance = medical[medical["insuranceclaim"] == 0]


# In[61]:


with_insurance[:10]


# In[62]:


without_insurance = without_insurance["charges"]


# In[69]:


t_statistic, p_value = stats.ttest_ind(with_insurance, without_insurance, equal_var=True)
print(t_statistic, p_value)


# In[70]:


mean_with_insurance = np.mean(with_insurance)
mean_without_insurance = np.mean(without_insurance)

std_with_insurance = np.std(with_insurance, ddof=1)
std_without_insurance = np.std(without_insurance, ddof=1)

n_with_insurance = len(with_insurance)
n_without_insurance = len(without_insurance)

sp = np.sqrt(((n_with_insurance - 1) * std_with_insurance**2 + (n_without_insurance - 1) * std_without_insurance**2) / (n_with_insurance + n_without_insurance - 2))

t_statistic = (mean_with_insurance - mean_without_insurance) / (sp * np.sqrt(1/n_with_insurance + 1/n_without_insurance))

df = n_with_insurance + n_without_insurance - 2

p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=df))  # Two-tailed test

print(f'T-Statistic: {t_statistic}')
print(f'P-Value: {p_value}')


# In[ ]:


#this result shows that there is significant evidence for difference between the group
#causing us to reject the null hypothesis


# Congratulations! Hopefully you got the exact same numerical results. This shows that you correctly calculated the numbers by hand. Secondly, you used the correct function and saw that it's much easier to use. All you need to do is pass your data to it.

# __Q6:__ Conceptual question: look through the documentation for statistical test functions in scipy.stats. You'll see the above _t_-test for a sample, but can you see an equivalent one for performing a *z*-test from a sample? Comment on your answer.

# __A:__

# In[ ]:


#i couldnt find such a code for the z test 


# ## Learning outcomes

# Having completed this project notebook, you now have good hands-on experience:
# * using the central limit theorem to help you apply frequentist techniques to answer questions that pertain to very non-normally distributed data from the real world
# * performing inference using such data to answer business questions
# * forming a hypothesis and framing the null and alternative hypotheses
# * testing this using a _t_-test
