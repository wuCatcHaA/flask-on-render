import pickle
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

# โหลดข้อมูล
df = pd.read_csv('data_claims_cleaned.csv')

# แยกตัวแปรอิสระ (X) และตัวแปรเป้าหมาย (y)
X = df[['npol_auto', 'client_sex', 'client_age', 'lic_age', 'client_nother', 'cities2', 'north', 'rest']]
y = df[['nclaims_md_Log', 'cost_md_Log']]

# แบ่งข้อมูล train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['Has_claim'])

# สร้างและฝึกโมเดล
model = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=1300,
        max_depth=9,     
        min_samples_split=130,
        min_samples_leaf=60,
        random_state=42 )
    )
model.fit(X_train, y_train)

# บันทึกโมเดล
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# ตั้งค่า Flask
app = Flask(__name__)
CORS(app)

# โหลดโมเดล
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# เส้นทางสำหรับแสดงหน้าเว็บ
@app.route('/')
def home():
    return render_template('tob2.html')

# เส้นทางสำหรับพยากรณ์
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    npol = data['npol']
    gender = data['gender']
    age = data['age']
    lic_age = data['lic_age']
    client_norther = data['client_norther']
    city = data['city']
    North = data['North']
    rest = data['rest']

    input_data = [[npol, gender, age, lic_age, client_norther, city, North, rest]]
    prediction = model.predict(input_data)
    
    claims_frequency = prediction[0][0]
    claims_severity = prediction[0][1]

    # แปลงค่าให้อยู่ในช่วงที่เหมาะสม
    claims_frequency = np.exp(claims_frequency) - 10**-2
    claims_severity = np.exp(claims_severity) - 10**-2

    if claims_frequency < 0 or claims_severity < 0:
        claims_frequency = 0
        claims_severity = 0

    if claims_frequency > 0.5:
        claims_frequency = int(claims_frequency + 0.5)
    else:
        claims_frequency = int(claims_frequency)

    if claims_frequency == 0:
        claims_severity = 0

    if claims_severity == 0:
        claims_frequency = 0

    claims_severity = round(claims_severity, 4)

    return jsonify({
        'claims_frequency': claims_frequency,
        'claims_severity': claims_severity
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
