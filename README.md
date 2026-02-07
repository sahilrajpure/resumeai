# 📄 Streamlit PDF Text Classification App

This is a **Streamlit-based Machine Learning web app** that allows users to upload a **PDF file**, extract its text, preprocess it, and run predictions using a **TensorFlow / Keras model**.

---

## 🚀 Features

* Upload and read PDF files
* Extract and clean text using regex
* Pad sequences for model compatibility
* Load a pre-trained TensorFlow/Keras model
* Display prediction results instantly
* Simple and clean Streamlit UI

---

## 🛠 Tech Stack

* **Frontend**: Streamlit
* **Backend / ML**: TensorFlow, Keras
* **Data Handling**: NumPy, Pandas
* **PDF Processing**: PyPDF2

---

## 📂 Project Structure

```
├── app.py                # Main Streamlit app
├── model.h5              # Trained TensorFlow/Keras model
├── tokenizer.pkl         # Saved tokenizer
├── requirements.txt      # Dependencies
├── runtime.txt           # Python version (for deployment)
└── README.md             # Project documentation
```

---

## 📦 Installation (Local)

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## ☁️ Streamlit Cloud Deployment

1. Push your project to **GitHub**
2. Add these files:

   * `requirements.txt`
   * `runtime.txt`
3. Set Python version in `runtime.txt`:

   ```txt
   python-3.10
   ```
4. Deploy from **Streamlit Cloud Dashboard**

---

## ⚠️ Notes

* Built-in Python libraries like `pickle`, `re`, `io`, and `datetime` are not listed in `requirements.txt`
* Recommended Python version: **3.9 or 3.10**

---

## 📈 Future Improvements

* Add confidence score visualization
* Support multiple PDF uploads
* Add model retraining option
* Improve UI with charts

---

## 👨‍💻 Author

**Sahil Rajpure**
BSc Computer Science | Machine Learning & AI Enthusiast

---

✨ If you like this project, feel free to ⭐ the repository!
