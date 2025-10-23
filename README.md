# DL-Prac-10_Project_VII_B_70

# Project: Probabilistic Deep Learning for Reliable Image Classification and Uncertainty Quantification

**Author:** Devesh Vyas
**Course:** B.Tech CSE-Data Science (Final Year) 

---

## 1. Problem Statement

Standard deep learning classifiers, like a Convolutional Neural Network (CNN), are powerful but have a critical flaw: they are often **overconfident**.

When trained to classify images (e.g., shirts, trousers, shoes), a standard CNN will still try to classify a completely unrelated image (e.g., a handwritten digit, a cat) as one of the known categories, often with a very high (but wrong) confidence score. This "confidently wrong" behavior is dangerous in real-world systems like medical diagnosis or self-driving cars.

The objective of this project is to build and compare deep learning models that can **quantify their own uncertainty**. A reliable model should not only be accurate on data it has seen before (in-distribution) but also express high uncertainty—effectively saying "I don't know"—when shown new, unfamiliar types of data (out-of-distribution).

## 2. Explanation & Solution

We will solve this problem by comparing three models trained on the **Fashion-MNIST** dataset. We will then test their ability to identify **MNIST** (handwritten digits) as "unknown" or "out-of-distribution" (OOD).

### Dataset Links

* **In-Distribution (Training):** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) - A dataset of 70,000 grayscale images (28x28) of 10 clothing categories.
* **Out-of-Distribution (OOD):** [MNIST](http://yann.lecun.com/exdb/mnist/) - A dataset of 70,000 grayscale images (28x28) of 10 handwritten digits.

### Techniques Used

1.  **Model 1: Baseline (Deterministic) CNN**
    * **Architecture:** A standard Keras `Sequential` model with `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` layers.
    * **Output:** A 10-unit `softmax` layer, giving a single probability distribution.
    * **Hypothesis:** This model will be accurate on the Fashion-MNIST test set but will produce high-confidence (but incorrect) predictions on the MNIST dataset.

2.  **Model 2: Probabilistic CNN via MC Dropout**
    * **Architecture:** The *same* CNN, but with `Dropout` layers added after `Conv2D` and `Dense` layers.
    * **Technique:** Dropout is a regularization technique, but it can be re-purposed as a Bayesian approximation. We keep dropout **active during inference** (testing) and run the prediction 50-100 times for the *same image*.
    * **Output:** We get 100 different `softmax` predictions. The *mean* of these is our final prediction. The **variance (or entropy)** of these predictions is our measure of uncertainty.
    * **Hypothesis:** This model will show low variance (high confidence) for Fashion-MNIST images and high variance (low confidence) for MNIST digit images.

3.  **Model 3: Bayesian Neural Network (BNN)**
    * **Architecture:** A similar CNN structure, but we replace standard Keras layers with probabilistic layers from the **TensorFlow Probability (TFP)** library (e.g., `tfp.layers.Convolution2DReparameterization`, `tfp.layers.DenseReparameterization`).
    * **Technique:** In this model, each **weight is a probability distribution** (e.g., a Gaussian) instead of a single number. The model learns the `mean` and `standard deviation` for every weight.
    * **Loss Function:** The loss is a combination of the standard (cross-entropy) loss and a **KL Divergence** term, which regularizes the model's complexity. This is based on Variational Inference.
    * **Hypothesis:** This model will explicitly learn the model's uncertainty. Like MC Dropout, it should produce predictions with high uncertainty for the OOD (MNIST) dataset.

## 3. Evaluation

The models will be compared on two criteria:
1.  **Accuracy:** Standard classification accuracy on the Fashion-MNIST test set.
2.  **Uncertainty Quantification:** We will feed both Fashion-MNIST (in-distribution) and MNIST (out-of-distribution) images to all three models. We will then plot histograms of their uncertainty scores (e.g., predictive entropy).

**Success:** A successful probabilistic model (MC Dropout, BNN) will show two clearly separated distributions: one for low uncertainty (for Fashion-MNIST) and one for high uncertainty (for MNIST). The baseline CNN will fail this test.
