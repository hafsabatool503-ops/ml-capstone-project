# 🚀 End-to-End Machine Learning Capstone Project

An end-to-end Machine Learning web application built with Scikit-Learn (RandomForest) and Streamlit, containerized using Docker for seamless deployment.

---

## 📌 Project Overview
This repository contains a production-ready Machine Learning web application built as part of the Capstone Project. The project trains a **Random Forest Classifier** using Scikit-Learn, serves predictions through an interactive **Streamlit** web dashboard, and is fully containerized with **Docker** for easy replication and deployment across environments.

---

## 📁 Repository Structure
```text
CAPSTONE PROJECT/
├── model/
│   ├── train.py          # Script to train the RandomForest classifier
│   └── model.pkl         # Serialized trained model artifact
├── app.py                # Streamlit web application interface
├── Dockerfile             # Docker container configuration
├── requirements.txt       # Python dependencies list
└── README.md              # Project documentation
