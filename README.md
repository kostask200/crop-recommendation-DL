# Crop Recommendation Neural Network

## Introduction
With the goal of improving practices, cutting costs, and minimizing risks, Artificial Intelligence has revolutionized the Agriculture Industry in recent years. The goal of this project is to build a neural network that can predict, using soil and environment characteristics, which crop would be most suited for production. There will be an evaluation of two different architectures: one with two hidden layers and another with five. To determine the best settings, a thorough hyperparameter tuning method will be used. Finally, both designs' performance will be evaluated to determine whether or not they can work in a crop recommendation system.

Ever since the introduction of precision agriculture, crop recommendation systems prove to be extremely beneficial in today's world. Precision agriculture is a farming management approach that enhances agricultural production sustainability by monitoring, measuring, and adapting to temporal and geographical variability. Accurate crop recommendations based on a variety of environmental, soil, and climatic characteristics are a key component of this optimization when traditional methods are labor-intensive and many times faulty.

This project aims to develop a robust neural network model capable of accurately predicting the best crops to cultivate in a given area. By harnessing the power of deep learning, we seek to provide farmers with valuable insights that can enhance their decision-making process and ultimately maximize crop yields.

## Methodology
1. **Data Acquisition:** Acquire the dataset from Kaggle, a popular platform for hosting datasets, ensuring its relevance to the crop recommendation task.
   
2. **Data Preprocessing:**
    - Identify and handle missing values to maintain data integrity.
    - Detect and address outliers using the Interquartile Range (IQR) method to prevent them from skewing the analysis.

3. **Statistical Analysis:**
    - Visualize the distribution of data through histograms and boxplots to gain insights into its characteristics.
    - Calculate summary statistics to understand the central tendency and dispersion of the data.

4. **Data Normalization and Splitting:**
    - Normalize the input features to ensure uniformity and facilitate model convergence.
    - Partition the dataset using the Holdout Method, allocating 70% for training and 30% for testing, while ensuring a balanced representation of each class in both sets.

5. **Neural Network Model Architecture Definition:**
    - Design the architecture for two models: one with 2 hidden layers and another with 5 hidden layers, incorporating techniques such as batch normalization and dropout to enhance model generalization.

6. **Hyperparameter Tuning:**
    - Specify the search space for hyperparameters, including parameters such as learning rate, batch size, and regularization strength.
    - Employ k-Fold Cross-Validation (k = 5) to systematically explore hyperparameter combinations and identify the optimal configuration based on mean accuracy.

7. **Model Selection:**
    - Evaluate and compare the performance of the two architectures based on the highest accuracy achieved during the hyperparameter tuning phase, selecting the most effective model for each architecture.

8. **Metrics Evaluation:**
    - Assess the chosen models' performance on the testing set by making predictions and generating a comprehensive classification report, which includes metrics such as accuracy, precision, recall, and F1-score, providing a holistic view of their effectiveness in crop recommendation.

In this README, we present a detailed overview of the methodology used to develop and train our crop recommendation neural network, along with insights into hyperparameter tuning and the results achieved. Additionally, we provide instructions on how to use our model and invite contributions from the community to further enhance its capabilities.

## Hyperparameter Tuning
To optimize the performance of our neural network models, we conducted an extensive hyperparameter tuning process. The table below outlines the search space for each parameter:

| Parameter                        | Options                                           |
|----------------------------------|---------------------------------------------------|
| Batch Size                       | 10, 50, 100                                       |
| Max Epochs                       | 10, 50, 100                                       |
| Optimizer                        | Adam, Adadelta, Adagrad, Adamax                   |
| Learning Rate                    | 0.0001, 0.001, 0.01                               |
| Activation Function              | ReLU, LeakyReLU, RReLU                             |
| Number of Neurons in Hidden Layers| 10, 40, 80                                       |
| Criterion                        | CrossEntropyLoss                                  |
| Dropout Rate                     | 0.0, 0.2, 0.5                                     |
| Weight Initialization            | Xavier Uniform Initialization, Xavier Normal Initialization |

### Experimentation Results
After thorough experimentation, the following insights were gathered:

1. Larger learning rates tend to yield better average scores for both architectures across all optimizers.
2. For the architecture with 2 hidden layers, increasing the batch size results in a lower average score, whereas the architecture with 5 hidden layers achieves its peak average score at a batch size of 50.
3. The combination of Adam optimizer and RReLU activation function demonstrates superior performance on average for both architectures.
4. Overall, the Adam optimizer yields better average scores for both architectures, with similar performance observed across all optimizers.
5. While all activation functions perform relatively similarly for the architecture with 2 hidden layers, RReLU stands out for better average performance in the architecture with 5 hidden layers.
6. Both architectures benefit from increased epochs, with optimizers and activation functions exhibiting better performance as the number of epochs increases.
7. It's worth noting that the architecture with 2 hidden layers took approximately 10 hours to train for the entire experimentation process, while the architecture with 5 hidden layers required around 17 hours.

These findings provide valuable insights into the impact of hyperparameters on the performance and training time of our neural network models, guiding further optimization efforts.

## Results

### Hyperparameter Tuning Results

#### Hyperparameter Tuning Results - 2 Hidden Layer Architecture
| Parameter                 | Value                              |
|---------------------------|------------------------------------|
| Batch Size                | 100                                |
| Max Epochs                | 100                                |
| Optimizer                 | Adamax                             |
| Learning Rate             | 0.01                               |
| Activation Function       | LeakyReLU                          |
| Number of Neurons         | 40                                 |
| Criterion                 | CrossEntropyLoss                   |
| Dropout Rate              | 0.2                                |
| Weight Initialization     | Xavier Normal Initialization       |

#### Hyperparameter Tuning Results - 5 Hidden Layer Architecture
| Parameter                 | Value                              |
|---------------------------|------------------------------------|
| Batch Size                | 100                                |
| Max Epochs                | 100                                |
| Optimizer                 | Adam                               |
| Learning Rate             | 0.01                               |
| Activation Function       | LeakyReLU                          |
| Number of Neurons         | 40                                 |
| Criterion                 | CrossEntropyLoss                   |
| Dropout Rate              | 0.0                                |
| Weight Initialization     | Xavier Normal Initialization       |

#### Hyperparameter Tuning Model Accuracy
| Hyperparameter Tuning    | 2 Hidden Layer Architecture (%)  | 5 Hidden Layer Architecture (%) |
|---------------------------|----------------------------------|----------------------------------|
| Accuracy                  | 98.7                             | 98.4                             |

### Classification Report

#### Classification Report - 2 Hidden Layer Architecture
| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| apple          | 1.00      | 1.00   | 1.00     | 30      |
| banana         | 1.00      | 1.00   | 1.00     | 30      |
| ...            | ...       | ...    | ...      | ...     |
| Weighted Avg   | 0.98      | 0.98   | 0.98     | 660     |

#### Classification Report - 5 Hidden Layer Architecture
| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| apple          | 1.00      | 1.00   | 1.00     | 30      |
| banana         | 1.00      | 1.00   | 1.00     | 30      |
| ...            | ...       | ...    | ...      | ...     |
| Weighted Avg   | 0.98      | 0.98   | 0.98     | 660     |

Overall, the hyperparameters for both model architectures are performing well and are aligned with the general findings of the experimentation process. Both models reveal consistent high performance across most classes, with both of them achieving an accuracy of 0.98. However, there are some subtle variations in the metrics for certain classes that highlight differences in the model’s ability to accurately classify specific instances. Overall, while both models can accurately classify, selecting the most suitable model may depend on the specific priorities of the classification at hand. Both of them prove to be suitable for a crop recommendation system that utilizes AI.


