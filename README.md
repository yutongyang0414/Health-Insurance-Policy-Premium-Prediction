# Health-Insurance-Policy-Premium-Prediction

```{r setup, include=FALSE}
knitr::opts_chunk$set(
	echo = TRUE,
	message = FALSE,
	warning = FALSE
)
insurance<-read.csv("insurance.csv")
library(tidyverse)
library(caret)
library(Metrics)
library(randomForest)
library(xgboost)
library(gridExtra)


```

## 1. Introduction

As the treatment expenses are increasing every day making it harder for people to afford quality medical treatments, people tend to purchase most suitable health and medical insurance plan for themselves and pay premium price in exchange of medical benefits. Also, health insurance companies can realize their profit if they collect more than what they spend on the medical care of its beneficiaries.  Hence, predicting future medical expenses of individuals based on existing medical expenses is essential to help medical insurance companies decide on charging the premium. The main objective of this analysis is to help the insurer to improve their policy premium pricing accuracy by predicting the insurance policy premium and identifying the factors that have a huge impact on medical premium price based on the data collected from the individuals.  

## 2. Exploratory Data Analysis

```{r}
head(insurance)
```

The dataset includes information about the insurance policy holder, their dependents, and their medical expenses throughout a year.

Age: Age of primary policyholder.

Sex: Sex of the policy policyholder.

BMI: Body Mass Index of policyholder, defined as the body mass divided by the square of the body height (kg/m2).

Smoker status: Whether the policyholder is a smoker or a non-smoker.

Children: Number of children/dependents covered in the policy.

Region of residence: Residential areas of the policy holder (in the US) - North East, South East, South West, North West.

Charges: Yearly medical expenses billed by the medical insurance provider ($).

```{r}
str(insurance)
summary(insurance)
```

The dataset contains 1338 observations of 7 variables. There are four numerical variables: `Age`, `bmi`, `children`, and `charges`.  There are three categorical variables:`sex`, `smoker`, and `region`.  In the summary statistics we also see that there are no missings in the dataset.

```{r echo=FALSE}
plot_age<-ggplot(insurance, aes(x = age , y= charges, color = smoker))+
  geom_point()
plot_bmi<-ggplot(insurance, aes(x = bmi, y= charges, color = smoker))+
  geom_point()
grid.arrange(plot_age, plot_bmi, ncol=2)
```
![download](https://github.com/yutongyang0414/Health-Insurance-Policy-Premium-Prediction/blob/main/Figures/download.png)

In the first plot we see that there is a trend that with older age the charges increase. There are also three groups/lines visible. In the second plot we see some sort of trend that with increasing `bmi` the `charges` increase, however this is not very clear. Here there might also be two different groups. After colored all the data with different smoking status, the first plot shows that smoker have relatively higher charges, which make sense. For the second plot, that smoker almost creates a whole new blob of points separate from non-smokers, and that blob sharply rises after bmi = 30, which indicates that there should be interaction between `smoker` and `bmi`.

```{r echo=FALSE}
insurance_new <- mutate(insurance, bmi_groups = cut(bmi, c(0,18.5,25,30,60)))
ggplot(insurance_new, aes(x = age , y= charges, color = bmi_groups))+
  geom_point()
```
![download-1](https://github.com/yutongyang0414/Health-Insurance-Policy-Premium-Prediction/blob/main/Figures/download-1.png)

Considering CDC official cutoff for obesity again, I grouped `bmi` into four groups: `Underweight`: bmi<=18.5, `Normal weight`:18.5<bmi<24.9, `Overweight`: 25,bmi<29.9, and `Obesity`:bmi>=30.  Frpm the plot, we can see that obesity will lead to higher charges.  But younger people still pay less money than older people in a consistent way so it does not appear that age interacts with `bmi` or `smoker`, meaning that it independently effects the `charge`.

```{r echo=FALSE}
plot_sex<-ggplot(insurance_new, aes(x = sex, y = charges)) +
 geom_boxplot()

plot_smoker<-ggplot(insurance_new, aes(x = smoker, y = charges)) +
 geom_boxplot()

plot_child<-ggplot(insurance_new, aes(x = as.factor(children), y = charges)) +
 geom_boxplot()

plot_region<-ggplot(insurance_new, aes(x = region, y = charges))+
 geom_boxplot()

grid.arrange(plot_sex, plot_smoker, plot_child, plot_region, ncol=2, nrow=2)
```
![download-2](https://github.com/yutongyang0414/Health-Insurance-Policy-Premium-Prediction/blob/main/Figures/download-2.png)
The first boxplot (left upper corner) shows us that females and males pay on avarage the same charges. When looking at the second boxplot (right upper corner) we see that smokers pay higher charges compared to non smokers. Also people with more childres pay more charges and it seems that the region has not an influence on the charges. In all instances the charges have a skewed distribution.

## 3. Preprocessing

```{r}
set.seed(123)
training.samples<-createDataPartition(insurance_new$charges,p=.8, list=FALSE)
train  <- insurance_new[training.samples, ]
test <- insurance_new[-training.samples, ]
```

The collecting medical expense data were randomly split into two subsets, training data, and test data. This consists of random sampling without replacement about 80 percent of the rows and putting them into training set. The remaining 20 percent is put into test set.  The training data were analyzed to investigate correlations between all factors that influence medical costs and to train a suitable model and optimize its performance to predict future costs. 

## 4. Model fitting

```{r}
set.seed(123)
train_control<-trainControl(method="cv",number=5)
```

To avoid overfitting, it is important to cross-validate the model.  We use k-fold method in this work. The training data is split into 5 folds. One of the folds will be the holdout set, and then we will fit the model on the remaining k-1 folds. Calculate the test MSE on the observations in the fold that was held out. Repeat this process 5 times, using a different set each time as the holdout set. And we will get the overall test MSE to be the average of the 5 test MSE’s, which will be used to evaluate the model performance.

### 1.Multiple Linear Regression

Multiple linear regression follows the formula :
Y = $\beta0$ + $\beta0$ * X1 + $\beta0$ * X2...
The coefficients in this linear equation denote the magnitude of additive relation between the predictor and the response. In simpler words, keeping everything else fixed, a unit change in x1 will lead to change of β1 in the outcome, and so on.

```{r,eval=FALSE}
intercept<-lm(charges~1,data=train)
all<-lm(charges~ as.factor(smoker)*age*as.factor(bmi_groups)*children*as.factor(sex)
        *as.factor(region),data = train)
forward<-step(intercept,direction = "forward",scope=list(upper=all,lower=intercept))
```

Sequential variable selection procedures offer the option of exploring some of the possible models efficiently examing other models in the neighborhood of a tentative current model and updating the current model stepwise. In this work, I use forward selection, starting with a constant mean as the initial curent model.  And we got the optimal model of:

charges = $\beta0$ + $\beta1$ * smoker + $\beta2$ * age + $\beta3$ * bmi +
           $\beta4$ * children + $\beta5$ * region + $\beta6$ * sex + $\beta13$ * smoker * bmi
                      
```{r}
model_linear<-train(charges~as.factor(smoker)+age+as.factor(bmi_groups)+children+
                      as.factor(region)+as.factor(sex)+as.factor(smoker):as.factor(bmi_groups),
                    data=train, method="lm",trControl=train_control)
print(model_linear)
getTrainPerf(model_linear)
```

After the cross-validation, we can see the average R-square of 0.8689, meaning that the model explains 86.89% of the variation in charges, which suggests a regression fit of 86.89%.

### 2. Random Forest

Random Forest combines multiple trees to predict the class of the dataset, it is possible that some decision trees may predict the correct output, while others may not. But together, all the trees predict the correct output.

```{r include=FALSE}
model_rf<-train(charges~as.factor(smoker)+age+as.factor(bmi_groups)+children+as.factor(region) +as.factor(sex)+as.factor(smoker):as.factor(bmi_groups),data=train,method="rf",trControl=train_control)
```

```{r warning=FALSE}
print(model_rf)
getTrainPerf(model_rf)
```

After the cross-validation, we can see the average R-square of 0.8552, meaning that the model explains 85.52% of the variation in charges, which suggests a regression fit of 85.52%.

### 3. XG-Boost

XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library. It provides parallel tree boosting and has enhanced performance and speed in tree-based (sequential decision trees) machine learning algorithms.

```{r include=FALSE}
model_xgb<-train(charges~as.factor(smoker)+age+as.factor(bmi_groups)+children+as.factor(region) +as.factor(sex)+as.factor(smoker):as.factor(bmi_groups),data=train,method="xgbTree",trControl=train_control)
print(model_xgb)
```

```{r}
getTrainPerf(model_xgb)
```

After the cross-validation, we can see the average R-square of 0.8689, meaning that the model explains 86.89% of the variation in charges, which suggests a regression fit of 86.89%.

### Compare three models

|model|TrainRsquared|
| :----:  | :----: |
|Multiple Linear Regression|0.8689323|
|Random Forest|0.8552542|
|XG-Boost|0.868971|

So XG-Boost shows highest trainRsquared value, which suggest the XG-Boost should be the one of the best models within these three models.

## 6. Models evalutation

To assess the accuracy of these three models, I fit the model with test data to get the predicted values. After comparing predicted value with charges in test data, we can use testRsquared value to assess the model accuracy.

```{r}
prediction_linear<-predict(model_linear,newdata = test)
error_linear<- test$charges - prediction_linear
R2_linear<-1-sum(error_linear^2)/sum((test$charges- mean(test$charges))^2)

prediction_rf<-predict(model_rf,newdata = test)
error_rf<- test$charges - prediction_rf
R2_rf<-1-sum(error_rf^2)/sum((test$charges- mean(test$charges))^2)

prediction_xgb<-predict(model_xgb,newdata = test)
error_xgb<- test$charges - prediction_xgb
R2_xgb<-1-sum(error_xgb^2)/sum((test$charges- mean(test$charges))^2)
```

```{r echo=FALSE, fig.height=12, fig.width=4, results="hide"}
par(mfrow = c(3,1))
plot(prediction_linear, test$charges)+abline(a=0,b=1)

plot(prediction_rf, test$charges,levels=1:266)+abline(a=0,b=1)

plot(prediction_xgb, test$charges,levels=1:266)+abline(a=0,b=1)

```
![download-3](https://github.com/yutongyang0414/Health-Insurance-Policy-Premium-Prediction/blob/main/Figures/download-3.png)
![download-4](https://github.com/yutongyang0414/Health-Insurance-Policy-Premium-Prediction/blob/main/Figures/download-4.png)
![download-5](https://github.com/yutongyang0414/Health-Insurance-Policy-Premium-Prediction/blob/main/Figures/download-5.png)

|model|TestRsquared|
| :----:  | :----: |
|Multiple Linear Regression|0.8302952|
|Random Forest|0.8105957|
|XG-Boost|0.8320162|

After predicting charges with test data, we can see the TestRsquared values for XG-Boost model is highest, which is consistent with the conclusion when we train the model.  This means the accuracy of XG-Boost can be 83.2%. Although it is not perfect, the model is sufficient given our purposes and level of accuracy. Also, for the plot of predicted charges vs test data charges, as though the area on the bottom left corner had the greatest concentration of charges, there is no obvious data departure from the fitted line.

## 7. Conclusion

Here, we were able to build three different models,  multiple linear regression model, random forest, and XG-Boost with our training data and use it to predict charges in our testing data. The obtained results showed that smoking and obesity has larger impact on the cost of health insurance.  After comparing the accuracy of the model, although it is not perfect, XG-Boost model shows good performance.  Its prediction accuracy is acceptable and could be used for future management of health care development.and give us enough model accuracy. 



