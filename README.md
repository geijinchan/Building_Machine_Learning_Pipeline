# Building Machine Learning Pipeline on Startup Acquisition
This machine learning pipeline predicts the acquisition potential of startups through a two-stage classification process. In the first stage, it performs binary classification to categorize startups as either "operating" or "non-operating." In the second stage, it uses multi-class classification to assign one of four specific outcomes: "acquired," "operating," "closed," or "IPO." By processing key data points like funding, revenue, and market trends, the pipeline enables accurate, data-driven predictions, helping investors identify high-potential startups and make informed acquisition decisions efficiently.

## Understanding the Dataset
The dataset we are working on is a collection of information related to companies sourced from **Crunchbase**.

- This dataset includes **196,553** entries and **40 columns**, each representing a company along with various attributes.

- Key columns in the dataset include:
  - **founded_at**
  - **funding_rounds**
  - **funding_total_usd**
  - **milestones**
  - **relationships**
  - **isClosed**
  - **status**

- This data can be utilized for various analyses, including evaluating company performance, investment opportunities, and market trends within different sectors.


## Preprocessing and Transformation

In the data preprocessing phase, we systematically handled missing values, transformed categorical variables, and removed outliers to ensure the dataset is clean and suitable for analysis.

1. **Dropping Unnecessary Columns**:
    - We dropped columns such as `region`, `city`, `id`, and others that were deemed irrelevant for our analysis to reduce noise in the dataset. This resulted in a reduction from **196,553** rows to **109,624** rows after dropping duplicates.

2. **Handling Missing Values**:
    - We calculated the number of NaN values and their percentage in each column. Columns with over **98%** missing values were identified and removed.
    - Specified columns (`status`, `country_code`, `category_code`, `founded_at`) were checked for NaN values, and any rows with missing values in these columns were dropped.

3. **Outlier Detection and Removal**:
    - For the `funding_total_usd` and `funding_rounds` columns, we calculated the Interquartile Range (IQR) to identify and remove outliers based on a threshold of **1.5 times the IQR**. This helped maintain the integrity of our data.

4. **Date Conversion**:
    - Columns containing dates (`founded_at`, `closed_at`, etc.) were converted to datetime format, and only the year was extracted for further analysis.

5. **Categorical Encoding**:
    - The `category_code` and `country_code` columns were transformed. The top 10 categories/countries were retained, while others were labeled as "Other".
    - One-hot encoding was applied to these columns to convert them into numerical format, facilitating machine learning processes.

6. **Feature Engineering**:
    - A new column `isClosed` was created to indicate whether a company is operating or has exited.
    - The `closed_at` column was filled based on the company's status, and the `active_days` were calculated to measure the duration of operation.
    - The `status` column was encoded into numerical values for analytical purposes.

7. **Handling Remaining NaN Values**:
    - After all preprocessing steps, we checked for any remaining NaN values. Numerical columns were filled with their respective mean values, ensuring a complete dataset for analysis.

8. **Final Dataset Check**:
    - The final dataframe was checked to ensure all preprocessing steps were completed successfully, with all NaN values addressed.


## Exploratory Data Analysis (EDA)

### **Introduction:**

The **company_data** dataset, which includes variables related to funding, investment, milestones, and company status, underwent thorough EDA to explore distributions and relationships between both numerical and categorical variables. The analysis provides insights into the structure and patterns within the dataset through various visualizations and statistical techniques.

### **Univariate Analysis:**

- **Numerical Columns:** We focused on numerical features such as `funding_rounds`, `funding_total_usd`, `investment_rounds`, `active_days`, `milestones`, and `founded_at`. The distribution of these columns was visualized using histograms and KDE plots. This provided insights into:
  
  - **Funding Rounds & Investment Rounds:** Skewed distributions were observed for both features.
  - **Funding Total (USD):** The majority of companies have relatively lower funding amounts, but a few companies received very high amounts, leading to a long right tail in the distribution.
  - **Milestones & Active Days:** These features displayed varying distributions, with some companies showing a higher number of milestones or longer periods of activity.

- **Categorical Columns:**

  - **Status Distribution:** The dataset's categorical variable `status` was analyzed using a pie chart, revealing the proportions of companies that are active, acquired, closed, or in other statuses.
  - **Categories and Countries:** One-hot encoded variables for `category` and `country` were analyzed to visualize their distribution. Pie charts were used to show the distribution of companies across different categories and countries.

### **Bivariate and Multivariate Analysis:**

- **State Code vs Status:** A scatter plot was created to examine the relationship between `state_code` and `status`, indicating patterns and outliers based on the company's operational status across different states.

- **Status Encoded vs isClosed:** A cross-tabulation and bar plot were created to explore the relationship between `status_encoded` and `isClosed`. This helped identify how many companies are closed based on their status encoding.

- **Box Plots:**

  - **Founded_at by Status:** Box plots showed the distribution of founding years (`founded_at`) for companies in different statuses. It revealed the range and outliers for when companies were founded.
  - **Funding Total (USD) by Status:** Box plots and bar plots were used to compare funding totals across different company statuses, showing that certain statuses (like "acquired") tend to have significantly higher funding totals.

- **Pair Plot for Numerical Features:** Pair plots were generated for numerical features, allowing us to observe potential relationships, clusters, or outliers between variables such as `funding_total_usd`, `milestones`, and `investment_rounds`.

- **Correlation Heatmap:** A Pearson correlation matrix was computed to explore the relationships between numerical variables. Strong positive correlations were observed between variables like `funding_rounds` and `investment_rounds`. However, negative correlations were found for some variables such as `isClosed` and `active_days`. The heatmap visually illustrated these correlations with color gradations representing the strength of relationships.

### **Visualization of Variables:**

- **Box Plots Across Categorical Variables:** For each categorical variable, box plots were created to compare their impact on numerical features. This allowed for a deeper understanding of how categorical variables like `status`, `category`, and `country` relate to numerical features like `funding_total_usd` and `milestones`.

- **Scatter Plots and Pair Plots:** Scatter and pair plots were used for a deeper look into numerical feature relationships, especially to detect correlations and potential linear or nonlinear relationships between variables.

- **Correlation Matrix:** A heatmap of the correlation matrix revealed which variables were strongly or weakly related, providing insights into potential feature interactions for further analysis or modeling.

---

This detailed EDA helps identify important patterns, outliers, and relationships within the dataset, providing a foundation for feature engineering and predictive modeling in later stages of the project.

 
 
## Feature Engineering

### **Introduction:**
Feature engineering was conducted to create new variables that provide additional insights into company characteristics and help improve model performance. These new features aim to capture investment timelines, funding efficiency, and the pace of investment activities relative to the company's milestones and lifespan.

### **Important Features:**
Based on the correlation analysis with `status_encoded`, the most important features for predicting company status were identified:
- **Milestones** (Correlation: 0.43)
- **Relationships** (Correlation: 0.38)

Given the significant correlation of these features with company status, new features were engineered to enhance the dataset by incorporating information related to company milestones, investment activities, and funding.

### **New Feature Creation:**

- **New Feature 1: Investment Time (in Days)**
  - **Definition:** The number of days between a company's first and last investment.
  - **Formula:** `investment_time_days = (last_investment_at - first_investment_at) * 365`
  - **Purpose:** Captures the time span of investments to understand the duration of a company's investment lifecycle.

- **New Feature 2: Funding Per Milestone**
  - **Definition:** The total funding received by the company divided by the number of milestones achieved.
  - **Formula:** `funding_per_milestone = funding_total_usd / milestones`
  - **Purpose:** Provides a measure of funding efficiency, indicating how much funding the company requires to achieve each milestone.

- **New Feature 3: Investment Rounds Per Year**
  - **Definition:** The number of investment rounds divided by the number of years since the company was founded.
  - **Formula:** `investment_rounds_per_year = investment_rounds / (2021 - founded_at)`
  - **Purpose:** Quantifies how frequently a company raises funds relative to its age, providing insights into its investment pace.

## Data Preprocessing & Feature Scaling (Exploratory Analysis)

During the development process, several data transformations were explored to understand how different scaling techniques affect the numerical features. These transformations were **not applied** to the main dataset used for modeling but were instead used to observe and compare the results of different scaling methods.

### Exploratory Steps Performed:

1. **Min-Max Scaling (Normalization)**
   - Scales all values between 0 and 1.
   - Applied to: `investment_time_days`, `funding_per_milestone`, `investment_rounds_per_year`, and `funding_total_usd`.

2. **Standardization (Z-score Scaling)**
   - Centers data to a mean of 0 and a standard deviation of 1.
   - Applied to the same numerical features as Min-Max scaling.

3. **Log Transformation**
   - Logarithmic transformation was applied to handle skewness in `funding_total_usd`. This transformation helps to reduce the impact of large outliers.

4. **Polynomial Features**
   - Created interaction terms between `investment_time_days` and `funding_per_milestone` (degree 2).
   - Polynomial features capture non-linear relationships between variables.

5. **Quantile Transformation**
   - Transforms features to follow a uniform distribution. This is useful when the data contains outliers or is highly skewed.
   - Applied to the same set of numerical features.

## Feature Selection

After preprocessing and transforming the dataset, several feature selection techniques were employed to identify the most significant features contributing to the predictive power of the models.

### **Principal Component Analysis (PCA)**

PCA was performed for feature reduction, resulting in the following explained variance ratios for the first two components:

* **Explained Variance Ratio**:
  * PCA Component 1: **47.58%**
  * PCA Component 2: **28.27%**

### **ANOVA (Analysis of Variance)**

ANOVA was conducted to identify features that significantly contribute to the model's predictive capabilities. The selected features based on ANOVA results are:

* **Selected Features (ANOVA)**:
  * `investment_time_days`
  * `investment_rounds_per_year`

These features showed a statistically significant relationship with the target variable, making them crucial for model training.

### **Lasso and Ridge Regression**

To further refine the selection of features, both Lasso and Ridge regression techniques were applied. The outputs from these regression models are as follows:

* **Lasso Selected Features**:
  * `funding_per_milestone`
  * `funding_total_usd`

Lasso regression effectively reduced the coefficients of less significant features to zero, thus selecting the above features as the most impactful.

* **Ridge Coefficients**:
Coefficients: [ 4.77089763e-04 -3.42135762e-08 -2.82162431e-01 3.14431399e-08]

Ridge regression provided a more distributed set of coefficients, suggesting a more nuanced relationship among features. The coefficients indicate that `funding_total_usd` may be a significant predictor, given its negative coefficient.

### **Recursive Feature Elimination (RFE)**

RFE was applied to identify the best subset of features. The selected features from RFE are:

* **Selected Features (RFE)**:
  * `investment_rounds_per_year`
  * `funding_total_usd`

These features were retained for their significant influence on the target variable, contributing to a more streamlined and effective model.


## Model Building

## Decision Tree Classifier for Binary Classification

To predict the binary target variable `isClosed`, the following steps were performed:

1. **SMOTE Application**: The Synthetic Minority Over-sampling Technique (SMOTE) was applied to address class imbalance in the dataset. This technique generated synthetic samples for the minority class, ensuring a more balanced distribution.

    ```python
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("After SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())
    ```

    Before applying SMOTE, the class distribution was as follows:

    ```
    isClosed
    1    29928
    0     3533
    ```

    After applying SMOTE, the distribution became:

    ```
    isClosed
    1    23942
    0    23942
    ```

2. **Train-Test Split**: The dataset was split into training and testing sets in an 80:20 ratio, while preserving the class distribution:

    ```python
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,             # 20% for testing
        random_state=42,           # For reproducibility
        stratify=y                 # Preserve class distribution
    )
    ```

3. **Decision Tree Classifier**: A Decision Tree Classifier was initialized and trained on the resampled dataset:

    ```python
    from sklearn.tree import DecisionTreeClassifier

    dt = DecisionTreeClassifier(
        random_state=42,          # Set a random state for reproducibility
        class_weight='balanced'   # Handle class imbalance
    )

    dt.fit(X_train_resampled, y_train_resampled)
    ```

### Model Evaluation

After training, predictions were made on the test set, and the model performance was evaluated using the classification report:

```python
from sklearn.metrics import classification_report

y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred))

              precision    recall  f1-score   support

           0       1.00      1.00      1.00       707
           1       1.00      1.00      1.00      5986

    accuracy                           1.00      6693
   macro avg       1.00      1.00      1.00      6693
weighted avg       1.00      1.00      1.00      6693



SVM Classifier with Recursive Feature Elimination (RFE)

A Support Vector Machine (SVM) model was trained on the dataset with Recursive Feature Elimination (RFE) to optimize feature selection and improve model performance. The `SVC` classifier with a linear kernel was used, and RFE was configured to select the top 10 features that contribute most to the classification. 

After feature selection, the model was trained on the optimized feature set and tested on the test set to evaluate its performance.
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, accuracy_score, f1_score

svm = SVC(kernel='linear', random_state=42)
rfe = RFE(estimator=svm, n_features_to_select=10)  # Select optimal features

rfe.fit(X_train, y_train)
y_pred = rfe.predict(X_test)

# Model evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

The model achieved an accuracy of 0.8823 and an F1 score of 0.8306, indicating that it performs well on the main class while showing some limitations on minority classes, likely due to class imbalance. Below is the detailed classification report:


- Accuracy: 0.8823
- F1 Score: 0.8306

Classification Report:

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.89      | 1.00   | 0.94     | 5923    |
| 1     | 0.16      | 0.01   | 0.01     | 499     |


# Binary Classification using XGBoost

To predict the binary target variable `isClosed`, we employed the XGBoost Classifier, optimizing for accuracy and handling class imbalance with `scale_pos_weight` and SMOTE.

## Initial Steps

1. **Data Preparation**:
   - Used `isClosed` as the binary target variable, with `1` indicating closed companies and `0` for operational companies.
   - Dropped unnecessary columns, including `status_encoded`, `founded_at`, `first_investment_at`, and `last_investment_at`, which were not essential for binary classification.
   - One-hot encoded categorical features to prepare the data for XGBoost.

    ```python
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    # Target and feature variables
    y = company_data['isClosed']
    X = company_data.drop(columns=['isClosed', 'status_encoded', 'founded_at', 'first_investment_at', 'last_investment_at'])
    X_encoded = pd.get_dummies(X, drop_first=True)
    ```

2. **Train-Test Split**:
   - Split the dataset into training and testing sets (80:20) while maintaining reproducibility.
   
    ```python
    X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    ```

## Option 1: Handling Class Imbalance with `scale_pos_weight`
   - Calculated the imbalance ratio, setting the `scale_pos_weight` parameter to balance the classes within XGBoost.
   
    ```python
    # Calculate imbalance ratio for scale_pos_weight
    imbalance_ratio = len(company_data[company_data['isClosed'] == 1]) / len(company_data[company_data['isClosed'] == 0])

    # Initialize and train the XGBoost model
    xgb_binary = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=3, scale_pos_weight=imbalance_ratio)
    xgb_binary.fit(X_train_encoded, y_train)
    ```

3. **Cross-Validation and Model Evaluation**:
   - Evaluated model performance with cross-validation to assess consistency.
   
    ```python
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(xgb_binary, X_encoded, y, cv=5, scoring='accuracy')
    print("Cross-validation scores:", scores)
    print("Mean cross-validation score:", scores.mean())
    ```

4. **Evaluation on Test Set**:
   - Predicted and evaluated the model on the test set using accuracy, F1-score, and a classification report.

    ```python
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    # Predictions
    y_pred = xgb_binary.predict(X_test_encoded)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Classification Report:\n", classification_rep)
    ```

## Option 2: Handling Class Imbalance with SMOTE

1. **SMOTE Application**:
   - Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset by generating synthetic samples for the minority class.

    ```python
    from imblearn.over_sampling import SMOTE

    # Oversampling the minority class
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_encoded, y)
    ```

2. **Train-Test Split (Post-SMOTE)**:
   - Split the resampled dataset into training and test sets, ensuring no imbalance in training data.

    ```python
    X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    ```

3. **XGBoost Model Training (Without `scale_pos_weight`)**:
   - Trained XGBoost without `scale_pos_weight` since SMOTE had already balanced the classes.

    ```python
    xgb_smote = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=3)
    xgb_smote.fit(X_train_res, y_train_res)
    ```

4. **Model Evaluation**:
   - Predicted and evaluated the model on the test set, focusing on accuracy, F1-score, and the classification report.

    ```python
    # Predictions after SMOTE
    y_pred_res = xgb_smote.predict(X_test_res)

    # Evaluation metrics
    accuracy_res = accuracy_score(y_test_res, y_pred_res)
    f1_res = f1_score(y_test_res, y_pred_res)
    classification_rep_res = classification_report(y_test_res, y_pred_res)

    print("SMOTE - Accuracy:", accuracy_res)
    print("SMOTE - F1 Score:", f1_res)
    print("SMOTE - Classification Report:\n", classification_rep_res)
    ```

## Results Summary
The XGBoost model demonstrated high accuracy with both methods:
- **Option 1 (`scale_pos_weight`)** provided accurate predictions on the test set but indicated overfitting with an accuracy of 1.00.
- **Option 2 (SMOTE)** achieved balanced class performance, mitigating overfitting and preserving the model’s generalizability.

This binary classification using XGBoost and balancing techniques helped improve model robustness, supporting effective predictions for startup acquisition status.

## Machine Learning Pipeline

The machine learning pipeline is structured to handle a two-stage classification process for predicting startup acquisition status. This pipeline incorporates binary and multiclass classification models, providing predictions on both the operational status and specific outcomes (acquired, operating, closed, IPO).

### Pipeline Overview

The pipeline consists of the following stages:

1. **Binary Classification Stage**:
    - The binary classifier predicts whether a startup is **operating** or **non-operating**.
    - This stage uses a **Logistic Regression** model that outputs the probability of the startup being closed (`isClosed`).
    - The resulting probability is appended as a new feature to the dataset to enhance the multiclass classification model’s ability to differentiate among specific outcomes.

2. **Multiclass Classification Stage**:
    - With the augmented dataset (original features + binary probability), the multiclass classifier predicts one of the four specific outcomes for each startup:
      - **Acquired**
      - **Operating**
      - **Closed**
      - **IPO**
    - This stage employs a **Random Forest Classifier** to leverage both the original features and the binary classification output for accurate multiclass prediction.

### Pipeline Components

- **Custom Transformer (`BinaryClassifierTransformer`)**:
    - A custom transformer was developed to integrate the binary classification into the pipeline. This transformer:
      - Fits a binary classification model (Logistic Regression) on the dataset.
      - Appends the probability of the startup being closed as an additional feature to the dataset, aiding in multiclass prediction.

- **Binary Pipeline**:
    - The binary pipeline handles feature scaling and binary classification transformation. 
    - Steps include:
      1. **Standard Scaling**: Normalizes feature values to enhance model convergence and improve interpretability.
      2. **Binary Transformation**: Uses `BinaryClassifierTransformer` to generate and append the binary probability as a new feature.

    ```python
    # Binary Pipeline
    binary_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Feature scaling
        ('binary_transform', BinaryClassifierTransformer(binary_model=binary_clf, probability=True))
    ])
    ```

- **Multiclass Pipeline**:
    - The multiclass pipeline uses the augmented dataset (including binary probability) and fits the Random Forest classifier for multiclass predictions.

    ```python
    # Multiclass Classifier
    multiclass_clf = RandomForestClassifier(random_state=42, n_estimators=100)
    ```

### Pipeline Workflow

1. **Training**:
    - The binary pipeline is first trained on the dataset to fit the binary classifier and generate binary probabilities.
    - The binary probability feature is added to the original dataset, creating an augmented dataset.
    - The augmented dataset is then used to train the multiclass classifier.

2. **Prediction**:
    - For new data, the binary pipeline transforms the input by adding the binary probability feature.
    - The multiclass classifier then uses the augmented data to predict the specific acquisition status.

3. **Evaluation**:
    - The pipeline’s performance is evaluated using classification metrics, with a focus on precision, recall, and F1-score for each class in the multiclass prediction.

### Saving and Loading the Pipeline

The pipeline components are saved using `cloudpickle` to allow for easy reloading and model reuse. The saved files include:
- `binary_pipeline.pkl`: Saves the binary pipeline with preprocessing steps and binary classifier.
- `multiclass_clf.pkl`: Saves the multiclass classification model.

```python
# Save the binary and multiclass pipelines
with open('binary_pipeline.pkl', 'wb') as f:
    cloudpickle.dump(binary_pipeline, f)

with open('multiclass_clf.pkl', 'wb') as f:
    cloudpickle.dump(multiclass_clf, f)
```

## Project Directory Structure

The following is the directory structure for the Flask app and machine learning pipeline:

```
Building-Machine-Learning-Pipeline-on-Startup-Acquisitions/
│   app.py                 # Main Flask application file
│   custom_transformers.py # Custom transformers for binary classification
│   final_pipeline.pkl     # Saved model pipeline file
│   Procfile               # Configuration file for deployment on platforms like Heroku
│   requirements.txt       # Dependencies required to run the application
│   runtime.txt            # Specifies Python version for deployment
│
├───templates
│       index.html         # HTML template for user input page
│       result.html        # HTML template for displaying prediction results
│
└───__pycache__
        custom_transformers.cpython-312.pyc  # Compiled cache file for custom transformers
```

### Explanation of Key Files

- **`app.py`**: The main Flask application file that sets up the web server, routes, and logic for handling requests and generating predictions.
- **`custom_transformers.py`**: Contains the `BinaryClassifierTransformer`, a custom transformer used to add binary classification predictions as features for the multiclass classifier.
- **`final_pipeline.pkl`**: The complete machine learning pipeline saved as a serialized file for easy loading and deployment.
- **`Procfile`**: Configures the command to start the app when deployed to cloud services (e.g., Heroku).
- **`requirements.txt`**: Lists all necessary Python packages for the app to run, ensuring consistent setup in any environment.
- **`runtime.txt`**: Specifies the Python version required for deployment.
- **`templates/index.html`** and **`templates/result.html`**: HTML templates that define the frontend of the app for user interaction and display of results.

### Flask App Overview

The Flask app, defined in `app.py`, handles user requests to predict the acquisition status of startups based on features such as founding year, funding rounds, and relationships. It includes:

- **Main Route (`/`)**: Displays the homepage (using `index.html`) where users enter startup information.
- **Prediction Route (`/predict`)**: Processes the input data, passes it through the binary and multiclass classifiers, and returns the acquisition status, which is displayed in `result.html`.

### Closing Notes

This project demonstrates a comprehensive machine learning pipeline from data preprocessing and feature engineering to model deployment using a Flask web application. By enabling predictions on startup acquisition status, it provides valuable insights for potential investors or analysts seeking to evaluate startup performance.

