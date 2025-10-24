# DL-Prac-10_Project_VII_B_70

# Project: Uncertainty Quantification in Deep Learning for Image Classification
**Author:** Devesh Vyas
**Course:** B.Tech CSE-Data Science (Final Year) 

---

## 1. Problem Statement

Standard Deep Learning models, such as Convolutional Neural Networks (CNNs), are powerful but have a significant flaw: they are often **overconfident**. A standard CNN will output a prediction (e.g., "This is a '5'") with high confidence, even when it is wrong.

Furthermore, they have no reliable mechanism to identify **Out-of-Distribution (OOD)** data. If a model is trained only on handwritten digits (0-9), and you show it a picture of a cat, it will still confidently (and incorrectly) classify it as one of the digits. This lack of "self-awareness" is dangerous for real-world applications like medical diagnosis or autonomous driving.

This project solves this problem by exploring, analyzing, and applying **Probabilistic Deep Learning (PDL)** models. These models are designed to quantify their own uncertainty, allowing them to:
1.  Express low confidence on ambiguous or difficult predictions.
2.  Signal high uncertainty when faced with novel OOD data.

## 2. Explanation & Solution

To address the problem, we build and compare three different models trained on the **MNIST** dataset of handwritten digits. We then test their "honesty" by evaluating them on both the MNIST test set (In-Distribution) and the **Fashion-MNIST** dataset (Out-of-Distribution).

### Dataset Links

The datasets are loaded directly from the `tf.keras.datasets` API.
* **In-Distribution (ID):** [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
    * `tf.keras.datasets.mnist.load_data()`
* **Out-of-Distribution (OOD):** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
    * `tf.keras.datasets.fashion_mnist.load_data()`

### Techniques Used

1.  **Baseline: Standard (Deterministic) CNN**
    * **What it is:** A regular Keras CNN. Its weights are single, fixed numbers.
    * **Prediction:** Outputs a single probability distribution. It is deterministic, meaning the same input will always produce the same output.
    * **Hypothesis:** It will be overconfident on incorrect predictions and OOD data.

2.  **Technique 1: Monte Carlo (MC) Dropout**
    * **What it is:** A standard CNN with `Dropout` layers. The "trick" is that we keep `Dropout` **active during inference (testing)**.
    * **Prediction:** We run the same input through the model 100 times. Because `Dropout` randomly deactivates different neurons each time, we get 100 different predictions.
    * **Uncertainty:** The **variance (or spread)** of these 100 predictions tells us the model's uncertainty. If all 100 predictions are the same, the model is certain. If they are all over the place, the model is uncertain. This is a practical approximation of Bayesian inference.

3.  **Technique 2: Bayesian Neural Network (BNN)**
    * **What it is:** A true probabilistic model built using **TensorFlow Probability (TFP)**.
    * **How it works:** Instead of learning a single fixed value for each weight, a BNN learns a full **probability distribution** (e.g., a mean and a standard deviation) for every weight in the network. We use `tfp.layers.DenseFlipout` for this.
    * **Prediction:** A single pass through the network involves *sampling* weights from these distributions. By passing the same input 100 times, we get 100 different predictions.
    * **Uncertainty:** Like MC Dropout, we measure the variance of the predictions. This is a more principled, but computationally heavier, way to model uncertainty.

## 3. Evaluation

* **Incorrect In-Distribution Predictions:** When the Standard CNN misclassifies an MNIST digit, it remains highly confident. The MC Dropout and BNN models correctly show high uncertainty, with probabilities spread across multiple possible digits.
* **Out-of-Distribution Data:** When shown a T-shirt from Fashion-MNIST, the Standard CNN confidently (and absurdly) predicts it's a digit. The MC Dropout and BNN models show very high uncertainty (a flat, uniform-like probability distribution), which is the correct way to say **"I don't know what this is."**
* **Entropy Plots:** The final histograms clearly show that the predictive entropy (a measure of uncertainty) for OOD data (Fashion-MNIST) is significantly higher and more separated for the BNN and MC Dropout models compared to the standard CNN.
