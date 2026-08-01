# Explainable AI System for Heart Disease Prediction



## Project Overview

This project develops an Explainable Artificial Intelligence (XAI) system for predicting the risk of heart disease using a Random Forest classifier. Unlike traditional black-box models, this application explains the reasoning behind the prediction using SHAP , making the model more transparent and trustworthy. The system is deployed through a Streamlit web application that allows users to interactively predict heart disease risk and visualize feature importance.



## Features

- Predicts heart disease risk using a Random Forest classifier

- Explains predictions using SHAP

- Interactive Streamlit web interface

- Visualizes feature importance

- Improves transparency of AI predictions


## Technologies Used

- Python

- Pandas

- NumPy

- Scikit-learn

- SHAP

- Matplotlib

- Streamlit
  

## Screenshots

### Home Page


<img width="662" height="851" alt="Screenshot 2026-07-25 101913" src="https://github.com/user-attachments/assets/3253192d-258e-4e5a-9e5d-334e854e0558" />


### Prediction and SHAP Explanation


<img width="637" height="856" alt="Screenshot 2026-07-25 101946" src="https://github.com/user-attachments/assets/284d240b-cd9a-40e2-bd38-df21a75f170d" />


### Key Factors Affecting the Prediction


<img width="652" height="868" alt="Screenshot 2026-07-25 102044" src="https://github.com/user-attachments/assets/a6823e6c-abb1-4610-8d82-4ae9bd379aa9" />




## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/divyanshi2323/Explainable-AI-System.git
```

### 2. Navigate to the project folder

```bash
cd Explainable-AI-System
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit application

```bash
streamlit run app.py
```



## Output

The application generates

• Heart disease prediction

• SHAP Waterfall Plot

• Feature contribution explanation

• Interactive Streamlit dashboard



## Conclusion

Traditional machine learning models often behave as "black boxes". This project demonstrates how Explainable AI techniques like SHAP can improve transparency, interpretability and trust in AI-based healthcare applications while providing meaningful insights into model predictions.

