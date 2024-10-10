# Project 1

## Team Members:

- **Aakash Shivanandappa Gowda**  
  A20548984
- **Dhyan Vasudeva Gowda**  
  A20592874
- **Hongi Jiang**  
  A20506636
- **Purnesh Shivarudrappa Vidhyadhara**  
  A20552125

## 1. Overview

This project implements an **Elastic Net regression model** (ElasticNetModel) that combines L1 (Lasso) and L2 (Ridge) regularization techniques. It is designed to handle high-dimensional datasets, feature selection, and multicollinearity problems. The model uses **gradient descent** for optimization, function consists of the MSE loss, L1 loss, L2 loss.

## 2. Class and Function Descriptions

### `ElasticNetModel`

The `ElasticNetModel` class implements the Elastic Net regression model. Below are the key methods and their descriptions:

#### `__init__(self, lambdas=0.1, thresh=0.5, max_iter=1000, tol=1e-4, learning_rate=0.01)`

Initializes the Elastic Net model with the following parameters:

- **`lambdas`** (_float_, default=0.1): Penalty coefficient for regularization.
- **`thresh`** (_float_, default=0.5): Mixing parameter between L1 and L2 regularization. Value ranges from 0 to 1, where 0 means L1 regularization only and 1 means L2 regularization only.
- **`max_iter`** (_int_, default=1000): Maximum number of iterations for gradient descent.
- **`tol`** (_float_, default=1e-4): Tolerance for the stopping condition. If changes in weights are smaller than `tol`, training stops.
- **`learning_rate`** (_float_, default=0.01): Step size for gradient descent updates.

#### `fit(self, X, y)`

Trains the Elastic Net regression model using gradient descent on the provided feature matrix `X` and target values `y`.

- **Parameters**:
  - **`X`** (_numpy array_): Feature matrix of the training dataset (without intercept).
  - **`y`** (_numpy array_): Target values corresponding to the training dataset.
- **Returns**:
  - An instance of `ElasticNetResults`, containing the trained weight coefficients and intercept.

### `ElasticNetResults`

This class stores the results after training the Elastic Net model.

#### `predict(self, X)`

Predicts target values for the provided feature matrix `X`.

- **Parameters**:
  - **`X`** (_numpy array_): Feature matrix of the test dataset.
- **Returns**:
  - Predicted target values.

### Example Usage

```python
from elasticnet/models import ElasticNetModel

# Initialize the model
model = ElasticNetModel(lambdas=0.1, thresh=0.5, max_iter=1000, tol=1e-4, learning_rate=0.01)

# Fit the model on training data
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)
```

### `generate_negative_data`

The `generate_negative_data` function generates synthetic data with two distinct patterns:

1. The first half of the features has a **monotonic increasing trend**.
2. The second half of the features has a **linear decreasing trend** (i.e., negative slope).

This function can be useful for testing model's ability in negative correlations data.

#### Parameters:

- **`range_x`** (_tuple_): The range of feature values (min, max).
- **`noise_scale`** (_float_): Standard deviation of the Gaussian noise added to the data.
- **`size`** (_int_): Number of samples in the dataset.
- **`num_features`** (_int_): Total number of features (half with increasing trend, half with decreasing trend).
- **`seed`** (_int_): Random seed for reproducibility.

#### Returns:

- **`X`** (_numpy array_): Generated feature matrix with both increasing and decreasing trends.
- **`y`** (_numpy array_): Target values with contributions from the features and added noise.

#### Example Usage:

```python
from generate_negative_regression_data import generate_negative_data

# Generate negative slope data
X, y = generate_negative_data(range_x=(0, 10), noise_scale=1.0, size=100, num_features=6, seed=42)
```

---

### `generate_rotated_positive_data`

The `generate_rotated_positive_data` function generates synthetic data with two patterns:

1. The first half of the features follows a **monotonic increasing trend**.
2. The second half exhibits a **wavy (S-shaped) pattern**, adjusted by a rotation matrix to create a slanted shape.

This function generates more complex data to test model's ability.

#### Parameters:

- **`range_x`** (_tuple_): Specifies the range of feature values (min, max).
- **`noise_scale`** (_float_): Standard deviation of the Gaussian noise added to the data.
- **`size`** (_int_): Number of samples in the dataset.
- **`num_features`** (_int_): Total number of features (half with increasing trend, half with a wavy pattern).
- **`seed`** (_int_): Random seed for reproducibility.
- **`rotation_angle`** (_float_, default=45): Angle (in degrees) to rotate the wavy pattern, introducing a slanted S-shape.
- **`mode`** (_int_, default=0): Determines the scaling factors for the feature values.

#### Returns:

- **`X`** (_numpy array_): Generated feature matrix with increasing trends and rotated S-shaped patterns.
- **`y`** (_numpy array_): Target values influenced by the features and noise.

#### Example Usage:

```python
from generate_positive_regression_data import generate_rotated_positive_data

# Generate positive data with rotation and S-shaped curves
X, y = generate_rotated_positive_data(range_x=(0, 10), noise_scale=1.0, size=100, num_features=6, seed=42, rotation_angle=45, mode=0)
```

## 1. Linear regression with ElasticNet regularization (combination of L2 And L1 regularization)

## Q1.What does the model you have implemented do and when should it be used?

### ElasticNetModel Overview

The model we've crafted, called **ElasticNetModel**, is an implementation of Elastic Net regression. What makes it special is that it not only uses MSE as the loss function, but also combines L1 (Lasso) and L2 (Ridge) regularization techniques. Here's a clearer view on how we implemented this model:

### Loss Function

- **The loss function consists of the MSE loss, L1 loss, and L2 loss. The loss function is shown as follows** :

$\text{Loss}$(w) = $\frac{1}{n}$ $\sum_{i=1}^{n}$ $(y_i - w^T x_i)^2$ + $\lambda$ ($\rho$ $\sum_{j=1}^{d}$ $|w_j|$ + (1 - $\rho$) $\sum_{j=1}^{d}$ $w_j^2$)

### Then use the gradient descent algorithm to calculate the gradient of the loss function and then update.


- **Gradient calculation results of MSE:** :

$\nabla_{\text{MSE}} = -\frac{2}{n} X^T (\hat{y} - y)$

- **Gradient calculation results of L1:**:

$\nabla_{\text{L1}} = \lambda_1 \cdot \text{sign}(w)$

- **Gradient calculation results of L2:** :

$\nabla_{\text{L2}} = 2 \cdot \lambda_2 \cdot w$

- **Total Gradient**:

$\nabla_{\text{Total}} = -\frac{2}{n} X^T (\hat{y} - y) + \lambda_1 \cdot \text{sign}(w) + 2 \cdot \lambda_2 \cdot w$

- **Weight Update**:

$w = w - \eta \cdot \nabla_{\text{Total}}$

$\eta \text{ is the learning rate, } \lambda_1 \text{ and } \lambda_2 \text{ are the penalty coefficients}$

### ElasticNetModel Overview

In summary, our model implements hybrid regularization by combining L1 and L2 penalties. This simplifies feature selection and improves model stability. Additionally, we uses gradient descent to efficiently update parameters and weights.

### What the Model Does:

- **Hybrid Regularization**: Out of all the classifiers Elastic Net is more advantageous because it uses a combination of L1 and L2 penalties.

- **L1 penalty** Reducing the effect of weak predictors makes the model purged of unnecessary items. Skipping all the lesser significant variables makes the model slimmer and easily understandable and easy to interpret, especially when one has a large number of potential predictors, often it is dreaded to end up with a bloated or overfit model.
  **L2 penalty** This makes this approach more useful when the features are related in some form, which is very often the case.This is because, when the influences are spread over various parameters of the model it ensures that the model can perform optimally in the long run.

- **Gradient Descent Optimization**: This technique is crucial when optimizing models in the cases where models are trained on large sets of data and older methods will slow down data processing. It does this in the sense that the different adjustments flow continuously and systematically so as to arrive at optimal value of the performance and hence its value is dependent on the learning rate and accuracy.

### When to Use This Model:

- **Broad Application Spectrum**:It is on such aspects that this model actually shines when in areas where accurate and dependable forecasts are needed. It becomes particularly useful in such areas as economics or healthcare where the data may be very much complicated, and unraveling the relationships may well determine the difference between victory and defeat.

- **Managing Complexity and Ensuring Accuracy**: It is great for getting to clean insights from messy, raucus data. Be it researching about human genetics or entire industry trends, this model enables to handle tonnes of information without diverting too much from the central theme.

The **ElasticNetModel** It is an entire package where not only can you get to make predictions but also deep dive into the data. A weapon of productivity for anyone dealing with a large data set that needs to be analysed and read into, not just numbers. Its ability to generalize, while solving for intricate relationships among variables which are characteristic challenge settings of modellers and analysts; makes this model immensely useful as a statistical package.

## Q2.How did you test your model to determine if it is working reasonably correctly?

### Model Testing and Validation Plan

For the purpose of the model’s strength and accuracy, we have developed an elaborate testing and validation plan. Let’s walk through each step of this strategy to see how it contributes to a full evaluation of the model’s performance:

### 1. Synthetic Data Generation

- **Method**: To illustrate how features are related to target values, we generated so called synthetic data supported by the function `generate_rotated_s_wave_data`.
- **Why It Matters**: This setup allows reducing the amount of noise and a closer look at its capacity to read and learn the patterns we established. It means that when working with synthetic data which contains already known outcomes the degree of the model’s ability to reproduce these relationships can be assessed as an indicator of learning precision.

### 2. Training on Generated Data

- **What We Did**: The model used a lot of time studying these synthetic datasets. : The following figures show the parameters of the fitted coefficients and intercepts of the tested article in this phase, correlating them with the expected patterns.
- **Why It Matters**: This can be done by executing the step that ensures that the model has comprehension of the dynamics in the data set. It assists in affirming to a degree that what the model is based on is sound as well as ensuring that it is sensitive to how it will react to the subtleties it is like to encounter in actuality.

### 3. Predictions on Unseen Data

- **Method**: To check how well the model is doing, we fed it with data it did not learn with during the learning stage.
- **Why It Matters**: This is essential for evaluating how good the model is in using the learning to solve other unseen dataset – which is very important in ascertaining the utility of the model in real world where dataset variation is the order of the day.

### 4. Model Evaluation Metrics

- **Metrics Used**: To check the accuracy and reliability of the carried out predictions we used several standard regression parameters such as:
  - Mean Squared Error (MSE)
    Linear – Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R-squared (R²)
- **Why They’re Important**: These measure give you actual numbers to explain the degree at which the model employed is capable of predicting new data. They assist us in getting an understanding of other aspects of performance including the average magnitude of the errors and variance in the target variable accounted for by the model.

### 5. Probability versus Outcome Plots

- **How We Do It**: We employ residual plots whereby each point represents a predicted value against its actual value. The closer that the observed points are to the 45-degree line, the better are the predictive capabilities of the model in question.
- **The Benefits**: These visuals provide an immediate, tangiblerefreshing ofits gross approximation. They work especially well when used to discern whether the model bolts general or is inconsistent (high variation, systematic error). They also allow us to visualize readily whether some of the features provide better prediction accuracy than others.

### 6. Analysis and Iteration

- **Ongoing Process**: Subsequently, the model is improved based on visual data and other collected items of measure. Such adjustments may entails the fine tuning of the parameters of the model so that predictions match actual outcomes particularly in the areas we identify discrepancies.
- **Why Keep At It**: These refinements assist in acquiring significant details in the new data patterns in order for the model to provide great results. This feedback and improvement cycle is essential to keep the model on the top of its form as new issues present themselves in the field.

#### Insights from Graphs:

Comparing the plot for “Feature 6 vs. Target” to the plot for “Feature 2 vs. Target” where despite the disperse nature of the former the latter seems to be more compact near the diagonal it is possible to conclude that Feature 2 is managed more accurately. This can put us in a position to ask ourselves whether Feature 6 should apply more data pre-processing or whether the model should apply more complex feature engineering methods to capture the fine relative data detail.

By combining these specific types of testing, not only do we guarantee to ourselves that the model is mathematically correct, but we also show that it can be used in practice in as many application fields as possible. This stringent check and balance system is well suited for creating accurate high quality predictive models that can climb even more for reliability.

## Q3.What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)

## Model Optimization Parameters

We gave some parameters to change in order to improve the performance of our model. This section is a breakdown of each parameter and what each does/means, as well as touches on how tuning them could lead to improved model performance:


### 1.⁠ ⁠*Lambdas*

   - *Function:* Lambdas are the key elements to the regularization strategies of our model, which we utilized in both L1 (Lasso) and L2 (Ridge) techniques. This alleviates a common overfitting problem in which the model learns noise as if it were real data signals.
   - *Effect of Regularization:* When the value of lambdas is increased, it increases the penalty on coefficients so that they are constrained to be near zero. This adjustment is beneficial when dealing with complex datasets that could potentially lead to misleading noise.
   - *When we use it:* When we observe our model is overfitting with training data, we then increase lambdas to rebuild our model in a much generalized way to new, unseen data.

### 2. *Thresh*

- *Function:*  
Thresh decides the elimination point for feature selection after filtering in, shaving the model down to select characteristics that are necessary and excluding others that may distort.

- *Adjustment Effect:*  
Raising the thresh value may cause getting rid of secondary variables as their coefficients become zeroes. Such a reduction in the model leads to less accurate results but occasionally, the results are expected to be better for unseen data.

- *Where To Use It:*  
Thresh is used to control the measure of variable importance, allowing identifying important features, and at the same time, excluding an excessive number of predictors that have no beneficial impact or, on the contrary, may negatively affect the model’s ability to make adequate predictions. This is more applicable especially where there are many features more than the dependent variables.



### 3.*Max_iter*

- *Function*: This parameter tells our algorithm to stop after running for the permitted number of iterations as stated below.
- *Impact of Adjustment*: Increasing the number of epochs may provide better chances for the model to adjust its weights with respect to data for better performance.
- *When We Use It*: If the model is not in its best condition yet, then it may be required to increase the max_iter. However, it is possible to have the model take more time for a training session, which calls for compromise between precision and time taken on the training session.

### 4. *Tol*

- *Function*: This is the same as tolerance; it defines a small the error which must fit our training data before further adjustments are made to stop.r adjustments stop. 
- *Impact of Adjustment*: Lowering the tol value ensures the model won’t stop learning until the fit is very tight, improving accuracy but extending training time.
- *When We Use It*: If rate of accuracy must be in the consideration. The most important one is that if you set the tolerance level too low then really long training can actually make no gain at all. 

### 5. *Learning_rate*

- *Function*:  This hyperparameter controls by how much the model should change towards the optimization in the updating of the model. 
- *Impact of Adjustment*: It reduces the steps taken towards implementation and is much slower but more precise, because it is a smaller learning rate and its implementations are more careful. 
- *When We Use It*: However, if the model is over the point that represents the minimal loss, it means that the learning rate is oversized, thus should be decreased to improve the stability and accuracy of the convergence. Nonetheless, the rate of learning may be reduced to an extreme low in the presence of a low rate. 

These two parameters, therefore, can be tuned to optimize accuracy for given data distribution and complexity of task, computational cost, and model complexity. 


We can improve this model by tinkering with these parameters to help the model adapt well to the nuances of our data and the requirements of our task, ensuring that we have a good trade-off between accuracy, efficiency of computation, and model lean size.

## Q4.Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

## Specific Inputs That May Challenge the ElasticNet Model

### 1. Highly Nonlinear Data

- *The Challenge*: This is because Elastic Net has an Optimistic World View, it believes that the world can be described using straight lines and linearly separable features. But life is rarely that clean or bumpy, and data exhibits the kind of curved relationships that a model like biases against.
- *What Happens*: Elastic Net might miss with its predictions in these instances since they involve some of the sharp edges/turns and complex patterns that it cannot fully understand.
  - *Feature Engineering*: One way to make the model learn the underlying patterns is by transforming those input features. This could involve squaring or cubing the features, maybe adding in interaction terms, possibly doing mathematical transformations (like logarithms or exponentials) to help straighten out those nonlinear relationships.
  - *Kernel Methods*: Taking a cue from support vector machine, these can be kernelized as well.
  - *Hybrid Models*:Two heads (or models) as well. We can combine the advantages of Elastic Net when teamwork with nonlinear models like decision trees or neural networks, which allows it to predict well in wider set of scenarios.

### 2. Noisy Data

- *The Challenge*:Think of your data as listening to one of those quiet FM radio stations during a thunderstorm, that is what this is like. This then becomes static (noise) that can drown the music (true patterns), and thus it makes it hard for the Elastic Net to hear what is important.
- *What Happens*:If the data is too noisy, Elastic Net might fit to the noise instead of the music and learn something that works great on training data but will falter in practice.
- *What We Can Do*:
  - *Increase Regularization*:we have the option here to turn up the lambdas knob and increase penalties for complexity, allowing our model to focus in on every loud symptomatic signal while ignoring that static.
  - *Threshold Tuning*: Changing the thresh parameter to allow us to be more selective about which features we let into the model, and cut out those that tend to contain more noise than signal.

  - *Noise Reduction Techniques*: Even before feeding data into our model, we can clean it up to filter out the oddball values or make some signal smoother so that what we pass on to our mod is as clean and clear as possible.

By acknowledging these limitations and using smart strategies to overcome them, we can increase the potential of Elastic Net because another graphical preferred in predictive modelling.

- ## Visualization

### The first train dataset:
![alt text](1.png)
### Scatter Plots of Train Features

This image shows a series of scatter plots, each representing the relationship between a different feature (Feature 1 to Feature 6) and Y_train based on training data. Each plot displays the data points in two dimensions:

- The horizontal axis (X-axis) shows values of a specific feature.
- The vertical axis (Y-axis) represents the values of Y_train.

The purpose of these plots is typically to visually assess the relationship between the features and the target variable. Observations can be made regarding the distribution, trend, or any potential outliers in the data. From the plots:

- **Feature 1, Feature 2, Feature 3:** These features display a scattered distribution with respect to Y_train as well as relatively clear linear relationships.
- **Feature 4, Feature 5, Feature 6:** These features show some vertical dispersion and a less clear linear relationship.

Overall, these scatter plots can help in determining which features might be relevant for modeling, although none of the features shown here have a distinct or strong linear relationship with the target variable.

### The first test dataset:
![alt text](2.png)

### Overview of Each Feature Plot Analysis

The image shows scatter plots of six different features (labeled "Test Feature 1" through "Test Feature 6") plotted against "Y_Test". Each subplot represents the data of one specific demension of feature.


### Meaning of this graph:
These plots visualize the dataset to help understand our first dataset.


### Model fitting effect:
![alt text](3.png)


### Image Description:
The figure above shows the effect of our model on the first test set based on the first training set, where the red line is the straight line fitted using the least squares method. The yellow points are true values and the blue points are predicted values.
It can be seen that the fitting effect of our model is very good at predicting the test level.




### The second train dataset:
![alt text](4.png)

### Scatter Plot Analysis

The image shows scatter plots of six features (Feature 1 through Feature 6) plotted against the target variable labeled "Y_Train" on the y-axis. The x-axis in each subplot corresponds to the respective feature values.

### Key Observations:

1. **Feature 1 to Feature 3**:
   - All three features show a strong positive linear correlation with the target variable (Y_Train).
   - The points align closely along a positively sloped line, indicating that as these feature values increase, the target variable increases in a linear fashion.

2. **Feature 4 to Feature 6**:
   - These features show a strong negative linear correlation with the target variable.
   - The points align closely along a negatively sloped line, suggesting that as these feature values increase, the target variable decreases.

### General Insights:
- All six features display clear linear relationships with the target, either positive (Feature 1 to 3) or negative (Feature 4 to 6).
- These patterns suggest that linear

### The  test dataset:
![alt text](5.png)


### Scatter Plot Analysis for Test Features vs. Target

The image shows scatter plots of six features (Test Feature 1 through Test Feature 6) plotted against the target variable labeled "Y_Test." Each subplot represents the relationship between a specific feature (on the x-axis) and the target variable (on the y-axis).

### Key Observations:

1. **Feature 1 to Feature 6**:
   - Each scatter plot shows a wide spread of points without a clear pattern, suggesting weak or no strong linear relationships between the features and the target variable.
   - The lack of clustering along a line indicates that none of the features exhibit a strong direct correlation with the target.
   
2. **Feature 5**:
   - There appears to be a slight downward trend, indicating a weak negative correlation between Feature 5 and the target (Y_Test).

3. **Feature 6**:
   - There is some clustering in the lower right quadrant, possibly indicating a negative influence on the target, but the scatter remains wide.

### General Insights:
- None of the features show a strong, easily identifiable linear correlation with the target variable, as the data points are dispersed across all plots.
- This suggests that more complex modeling techniques (e.g., non-linear methods) or additional feature engineering may be needed to capture meaningful patterns in the data.





![alt text](6.png)

### Regression Analysis for Features vs. Actual and Predicted Target Values

The image shows six scatter plots comparing the actual target values and the predicted target values for six features (Feature 1 through Feature 6) against the target variable labeled "Y." Each subplot also includes a red dashed line representing the linear fit for each feature.

### Key Observations:

1. *Feature 1 to Feature 3*:
   - The fit lines have positive slopes, suggesting a weak positive relationship between these features and the target variable.
   - The actual and predicted values are dispersed around the fit line with noticeable variance, indicating that the model's predictions are not tightly aligned with the actual values. The fit is somewhat loose, and the features do not have a strong linear relationship with the target.

2. *Feature 4 to Feature 6*:
   - The fit lines have negative slopes, indicating a negative relationship between these features and the target variable.
   - Similar to Features 1-3, there is substantial dispersion of the actual and predicted values around the fit line, suggesting weak predictive power and variability in the model's predictions for these features.

### General Insights:
- All six features exhibit weak linear relationships with the target variable, as evidenced by the scattered points around the fit lines. The model does not capture the target values well, and the fit is not strong.
- *Features 4, 5, and 6* show negative relationships, meaning that as the feature values increase, the target decreases, though this is not strong.
- These scatter plots indicate that the model could benefit from more advanced techniques, as simple linear regression appears insufficient for accurately predicting the target values for these features. Non-linear modeling methods or additional feature engineering might improve the model's performance.