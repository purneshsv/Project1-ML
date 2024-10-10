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

# 1. Linear regression with ElasticNet regularization (combination of L2 And L1 regularization)

# Q1.What does the model you have implemented do and when should it be used?

# ElasticNetModel Overview

The model we've crafted, called **ElasticNetModel**, is an implementation of Elastic Net regression. What makes it special is that it not only uses MSE as the loss function, but also combines L1 (Lasso) and L2 (Ridge) regularization techniques. Here's a clearer view on how we implement this model:

## Loss Function

The loss function consists of the MSE loss, L1 loss, and L2 loss. The loss function is shown as follows:

$\text{Loss} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^m |w_j| + \lambda_2 \sum_{j=1}^m w_j^2$

## Then use the gradient descent algorithm to calculate the gradient of the loss function and then update.

- **Gradient calculation results of MSE:**:

$\nabla_{\text{MSE}} = -\frac{2}{n} X^T (\hat{y} - y)$

- **Gradient calculation results of L1:**:

$\nabla_{\text{L1}} = \lambda_1 \cdot \text{sign}(w)$

- **Gradient calculation results of L2:**:

$\nabla_{\text{L2}} = 2 \cdot \lambda_2 \cdot w$

- **Total Gradient**:

$\nabla_{\text{Total}} = -\frac{2}{n} X^T (\hat{y} - y) + \lambda_1 \cdot \text{sign}(w) + 2 \cdot \lambda_2 \cdot w$

- **Weight Update**:

$w = w - \eta \cdot \nabla_{\text{Total}}$

$\eta \text{ is the learning rate, } \lambda_1 \text{ and } \lambda_2 \text{ are the penalty coefficients}$

### ElasticNetModel Overview

In summary, our model implements hybrid regularization by combining L1 and L2 penalties. This simplifies feature selection and improves model stability. Additionally, we uses gradient descent to efficiently update parameters and weights.

### What the Model Does:

- **Hybrid Regularization**: Elastic Net stands out because it uses a mix of L1 and L2 penalties.

  - **L1 penalty** helps thin out the less essential features, essentially cleaning up the model to focus on what really matters, making it simpler and easier to understand. This is especially useful when you’re dealing with loads of potential predictors and want to avoid a model that’s too complex to interpret or too tailored to the training data.
  - **L2 penalty** helps when features are intertwined, which is quite common. It does this by distributing the influence of these features more evenly, which helps keep the model stable and reliable.

- **Gradient Descent Optimization**: This technique is crucial for fine-tuning the model in big datasets where older methods might be too slow. It adjusts the model gradually and methodically to hone in on the best possible performance, based on the rate of learning and how precise we want the model to be.

### When to Use This Model:

- **Broad Application Spectrum**: This model shines in any scenario where you need precise and sturdy predictions. It’s fantastic for fields like economics or healthcare, where the data and relationships can be quite complex.

- **Managing Complexity and Ensuring Accuracy**: It’s perfect for diving into complicated, noisy data and coming out with clear insights. Whether it’s studying genes or predicting trends in big industries, this model helps manage a lot of information without losing focus.

The **ElasticNetModel** isn’t just about making predictions; it’s also about deeply understanding the data. It’s a powerful tool for anyone who needs to tackle complex data sets and derive meaningful conclusions, not just numbers. This model’s ability to simplify while handling complex relationships makes it invaluable for researchers and analysts who regularly face challenging modeling situations.

# Q2.How did you test your model to determine if it is working reasonably correctly?

## Model Testing and Validation Strategy

To ensure our model is both robust and accurate, we’ve put a thorough testing and validation strategy into play. Let’s walk through each step of this strategy to see how it contributes to a full evaluation of the model’s performance:

### 1. Synthetic Data Generation

- **Method**: We created synthetic datasets using a method called `generate_rotated_s_wave_data` that clearly defines how features relate to target values.
- **Why It Matters**: This setup lets us dive deep into the model’s ability to recognize and learn the patterns we’ve set up. By working with synthetic data, where the outcomes are already known, we can rigorously test how well the model mimics these set relationships, giving us a clear picture of its learning precision.

### 2. Training on Generated Data

- **What We Did**: The model spent a good deal of time learning from these synthetic datasets. During this phase, we kept a close eye on its parameters, checking how well the learned coefficients and intercepts matched the expected patterns.
- **Why It Matters**: This step is crucial for confirming that the model understands the underlying data dynamics. It helps validate that the model’s foundation is solid and that it’s tuned to respond appropriately to the nuances it will face in real scenarios.

### 3. Predictions on Unseen Data

- **Method**: We tested the model’s predictions using a new set of data that it hadn’t seen during training.
- **Why It Matters**: This is key for assessing how well the model can apply what it’s learned to new, unknown datasets—a must for ensuring the model is genuinely useful in real-world settings where data variability is the norm.

### 4. Model Evaluation Metrics

- **Metrics Used**: We measured the model’s accuracy and reliability using several standard regression metrics like:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R-squared (R²)
- **Why They’re Important**: These metrics provide hard numbers that tell us how accurately the model predicts new data. They help us understand different aspects of performance, such as the average size of errors and how much of the variance in our target variable the model can explain.

### 5. Visual Comparisons of Predicted vs. Actual Values

- **How We Do It**: We use scatter plots where each point shows a predicted value against its actual value. The closer these points are to the diagonal line (where predicted equals actual), the better the model’s predictions.
- **The Benefits**: These visuals offer an instant, gut-level indication of the model’s accuracy. They’re particularly good for spotting whether the model tends to over or underestimate across the board (systematic bias) or if it’s inconsistent (high variance). They also let us see at a glance if some features yield more reliable predictions than others.

### 6. Analysis and Iteration

- **Ongoing Process**: Based on what we learn from the visual data and other metrics, we continuously refine the model. This might mean tweaking the model’s settings to better align predictions with actual results, especially where we notice gaps.
- **Why Keep At It**: These refinements help the model adapt to the subtleties of new data, ensuring it stays relevant and effective. This ongoing cycle of feedback and improvement is vital for the model to remain at the top of its game as it encounters new challenges.

#### Insights from Graphs:

Observing how “Feature 2 vs. Target” shows a tighter cluster around the diagonal than “Feature 6 vs. Target” suggests Feature 2 is being handled more precisely. This might lead us to look into whether Feature 6 needs more preprocessing or whether the model should employ advanced feature engineering strategies to better capture complex data patterns.

By weaving together these detailed testing methods, we not only affirm that the model is theoretically sound but also ensure it’s equipped to perform reliably in a range of real-world applications. This rigorous validation and refinement process is crucial for building top-quality predictive models that can rise to new challenges and deliver dependable results.

# Q3.What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)

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

# Q4.Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

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
