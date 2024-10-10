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

## ElasticNetModel Overview

The model we've crafted, called **ElasticNetModel**, is an implementation of Elastic Net regression. What makes it special is that it not only uses MSE as the loss function, but also combines L1 (Lasso) and L2 (Ridge) regularization techniques. Here's a clearer view on how we implemented this model:

## Loss Function

- **The loss function consists of the MSE loss, L1 loss, and L2 loss. The loss function is shown as follows** :

$\text{Loss}$(w) = $\frac{1}{n}$ $\sum_{i=1}^{n}$ $(y_i - w^T x_i)^2$ + $\lambda$ ($\rho$ $\sum_{j=1}^{d}$ $|w_j|$ + (1 - $\rho$) $\sum_{j=1}^{d}$ $w_j^2$)

## Then use the gradient descent algorithm to calculate the gradient of the loss function and then update.

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

#### Model Testing and Validation Plan

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

1. **Lambdas**

   - **Function:** Lambdas are the key elements to the regularization strategies of our model, which we utilized in both L1 (Lasso) and L2 (Ridge) techniques. This alleviates a common overfitting problem in which the model learns noise as if it were real data signals.
   - **Effect of Regularization:** When the value of lambdas is increased, it increases the penalty on coefficients so that they are constrained to be near zero. This adjustment is beneficial when dealing with complex datasets that could potentially lead to misleading noise.
   - **When we use it:** When we observe our model is overfitting with training data, we then increase lambdas to rebuild our model in a much generalized way to new, unseen data.

2. **Thresh**

   - **Operation:** Thresh For: Feature Selection Description: It creates a cutoff for feature selection. It simplifies the model by only considering a few significant features. A higher thresh value can remove less useful features by nulling their coefficients and it will improve the model's interpretability on new data.
   - **Where We Use It:** Thresh is used when there were a lot of features in our dataset and we do not want to perform the data modeling on all those features (so using thresh makes sure that spearmake calculation applied only top x number of feature).

3. **Max_iter**

   - **Function:** It decides the maximum loops (number of iterations) our algorithm performs before stopping.
   - **Adjustment Impact:** The more iterations the model adjusts its weights with the data, the better it can learn and potentially yield higher accuracy.
   - **Why We May Use It:** If our model has not yet been optimized, it may require an increase in max_iter. But, it may potentially take more training time which means a tradeoff between accuracy and speed.

4. **Tol**

   - **Function:** Tol — that is, tolerance — how closely does the model have to fit the training data trends before we stop adjusting further.
   - **Impact of Tuning:** Lowering the value means the model won't stop learning until it achieves a closer fit, this can increase accuracy but would also prolong training time.
   - **Where We Use This:** This is most useful when high precision matters a lot. However, making the tolerance too tight can result in diminishing returns, especially if more training leads to virtually no progress with prolonged training.

5. **Learning_rate**
   - **Function:** In gradient descent optimization, this parameter helps us to set the step size the model takes towards reducing the loss function.
   - **Implication of Adjustment:** Lower learning rates give smaller and safer steps that might help in more finesse model optimization.
   - **Where We Use It:** If the model reaches minimum loss and is going beyond it i.e., overshooting, by decreasing the learning rate we can make sure it converges more stably and accurately to its minima. However, if the rate is too low, it will slow the training process in an overly long way.

We can improve this model by tinkering with these parameters to help the model adapt well to the nuances of our data and the requirements of our task, ensuring that we have a good trade-off between accuracy, efficiency of computation, and model lean size.

## Q4.Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

True, there are certain inputs that the implementation finds difficult working with:

## 1. Data Dependencies (Highly Nonlinear Data):

- **The Challenge**: Elastic Net has an unspoiled ability to see a linear world, plots each point in the n-dimensional Euclidean space, and captures the straight line that passes near as many points as possible. The information might express intricate, curved relationships that the model finds difficult to learn.

- **What Happens**: If the problem consists of complex patterns among variables, Elastic Net will fail to accurately predict target values due to its inability to model the variability and flexibility.

- **Real Feature Engineering**: To capture the latent driving forces more effectively, we can manipulate input features to provide the model with a better perspective of the underlying patterns. This may involve creating new features by squaring or cubing the original variables, introducing interaction terms between variables, or using mathematical transformations such as logarithms or exponentials to linearize some nonlinear relationships.

- **Kernel Methods**: Borrowing a trick from support vector machines, kernel methods can help by projecting data into a space where once-complex relationships look linear enough for Elastic Net to perform effectively.

- **Hybrid Models**: Sometimes combining two models can be better than one. However, hybrid models have their own challenges, such as limited variance in terms of bias and variance, which Elastic Net aims to address. We can combine tree models or neural networks with Elastic Net to improve prediction across a variety of scenarios.

## 2. Noisy Data:

- **The Challenge**: Noisy data is like trying to listen to a quiet radio station during a thunderstorm. Noise can drown out the true patterns, making it hard for Elastic Net to capture what's important.

- **Example**: If the data is too noisy, Elastic Net may learn the noise along with the true signal, leading to overfitting on the training set and poor performance in the real world.

- **What We Can Do**:

  - **Increase Regularization**: Turning up the lambda parameter increases the penalties for model complexity, forcing Elastic Net to focus on the main signal and ignore noise.

  - **Filter**: Adjusting the threshold parameter can make the model more or less strict about what signals pass through, helping to filter out noise.

  - **Reduction of Noise**: We can filter out outliers or smooth signals before feeding the data into the model.

This post aims to understand these limitations and the smart ways to get around them, solidifying Elastic Net as an even more powerful tool in predictive modeling.
