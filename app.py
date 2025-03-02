import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ใช้ tf.keras จาก tensorflow
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense

# ----- Dataset 1: Weather Data for K-Means -----
# ตัวอย่างสร้างข้อมูลจำลอง (คุณสามารถแทนที่ด้วยการโหลด dataset จาก URL ของ Google Drive หรือ AWS S3 ได้)
weather_data = {
    'Temperature': np.random.uniform(20, 35, 200),
    'Humidity': np.random.uniform(50, 90, 200),
    'Cloud Cover': np.random.choice(['Sunny', 'Cloudy', 'Overcast'], 200)
}
weather_df = pd.DataFrame(weather_data)
weather_df['Cloud Cover'] = weather_df['Cloud Cover'].map({'Sunny': 0, 'Cloudy': 1, 'Overcast': 2})

scaler = StandardScaler()
weather_scaled = scaler.fit_transform(weather_df[['Temperature', 'Humidity', 'Cloud Cover']])

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
weather_df['Cluster'] = kmeans.fit_predict(weather_scaled)

# ----- Dataset 2: Air Quality Data for Neural Network -----
# ตัวอย่างสร้างข้อมูลจำลอง (สำหรับ dataset ขนาดใหญ่ คุณสามารถโหลดจาก URL ได้)
air_data = {
    'PM2.5': np.random.uniform(10, 150, 200),
    'Humidity': np.random.uniform(40, 90, 200),
    'Temperature': np.random.uniform(15, 35, 200),
    'Pollution Level': np.random.choice([0, 1], 200)
}
air_df = pd.DataFrame(air_data)

X = air_df[['PM2.5', 'Humidity', 'Temperature']]
y = air_df['Pollution Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(8, activation='relu', input_shape=(3,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=0)

# ----- Streamlit App with Multiple Pages -----
st.title('Intelligent System Project')

# เมนูหลักใน sidebar
menu = st.sidebar.radio('Navigation', [
    'Home',
    'Data Explanation',
    'Process Explanation',
    'K-Means Clustering',
    'Neural Network'
])

if menu == 'Home':
    st.write("## Welcome to the Intelligent System Project")
    st.write("This app demonstrates models using K-Means Clustering and Neural Network.")
    st.write("Use the navigation menu on the left to view detailed explanations and model demos.")

elif menu == 'Data Explanation':
    st.write("## Data Explanation")
    st.write("### Dataset 1: Weather Data (for K-Means Clustering)")
    st.write("Data is generated randomly. In a real project, you might download the dataset from Google Drive or AWS S3.")
    st.write(weather_df.head())
    st.write("### Dataset 2: Air Quality Data (for Neural Network)")
    st.write("Data is generated randomly. For large datasets, consider using external storage.")
    st.write(air_df.head())

elif menu == 'Process Explanation':
    st.write("## Process Explanation")
    st.write("### Step 1: Data Preparation")
    st.write("- Clean and preprocess the datasets (e.g., scaling, mapping categorical variables).")
    st.write("### Step 2: Model Development")
    st.write("- Apply K-Means Clustering on the weather dataset.")
    st.write("- Develop a Neural Network model on the air quality dataset.")
    st.write("### Step 3: Web Application Development")
    st.write("- Build a multi-page app using Streamlit for demonstration and explanation.")

elif menu == 'K-Means Clustering':
    st.write("## K-Means Clustering Results")
    fig, ax = plt.subplots()
    scatter = ax.scatter(weather_df['Temperature'], weather_df['Humidity'], c=weather_df['Cluster'], cmap='viridis')
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Humidity")
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(fig)
    st.write("Click the back button in the sidebar to return to Home.")

elif menu == 'Neural Network':
    st.write("## Air Pollution Prediction with Neural Network")
    pm = st.slider('PM2.5', 10, 150, 50)
    humidity = st.slider('Humidity', 40, 90, 60)
    temp = st.slider('Temperature', 15, 35, 25)

    input_data = np.array([[pm, humidity, temp]])
    prediction = model.predict(input_data, verbose=0)
    pollution_prob = prediction[0][0]

    st.write(f'### Pollution Probability: {pollution_prob:.2f}')
    st.write("Click the back button in the sidebar to return to Home.")

# Note: For a large dataset, consider using pd.read_csv() with a URL from Google Drive or AWS S3.

