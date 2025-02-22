import pickle
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
<<<<<<< HEAD
import time
=======
>>>>>>> 59b64e29203428e3e919f78bb96a2ba798f0f8f4
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

<<<<<<< HEAD
# โหลดข้อมูล
df = pd.read_csv('data_claims_cleaned.csv')

# แยกตัวแปรอิสระ (X) และตัวแปรเป้าหมาย (y)
X = df[['npol_auto', 'client_sex', 'client_age', 'lic_age', 'client_nother', 'cities2', 'north', 'rest']]
y = df[['nclaims_md_Log', 'cost_md_Log']]

# แบ่งข้อมูล train-test
=======
# โหลดข้อมูล (สำหรับทดสอบ)
df = pd.read_csv('data_claims_cleaned.csv')

# แยกข้อมูลเป็น features และ target
X = df[['npol_auto', 'client_sex', 'client_age', 'lic_age', 'client_nother', 'cities2', 'north', 'rest']]
y = df[['nclaims_md_Log', 'cost_md_Log']]

# แบ่งข้อมูลเป็น training และ testing set
>>>>>>> 59b64e29203428e3e919f78bb96a2ba798f0f8f4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['Has_claim'])

# สร้างและฝึกโมเดล
model = MultiOutputRegressor(RandomForestRegressor(
<<<<<<< HEAD
        n_estimators=1300,
        max_depth=9,     
        min_samples_split=130,
        min_samples_leaf=60,
        random_state=42 )
    )
=======
    n_estimators=1300,
    max_depth=9,     
    min_samples_split=130,
    min_samples_leaf=60,
    random_state=42
))
>>>>>>> 59b64e29203428e3e919f78bb96a2ba798f0f8f4
model.fit(X_train, y_train)

# บันทึกโมเดล
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

<<<<<<< HEAD
# ตั้งค่า Flask
app = Flask(__name__)
CORS(app)

# โหลดโมเดล
=======
# สร้างแอป Flask
app = Flask(__name__)
CORS(app)

# โหลดโมเดลที่บันทึกไว้
>>>>>>> 59b64e29203428e3e919f78bb96a2ba798f0f8f4
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
    
    # แปลงผลลัพธ์จากการทำนาย
    claims_frequency = prediction[0][0]
    claims_severity = prediction[0][1]

<<<<<<< HEAD
    # แปลงค่าให้อยู่ในช่วงที่เหมาะสม
=======
>>>>>>> 59b64e29203428e3e919f78bb96a2ba798f0f8f4
    claims_frequency = np.exp(claims_frequency) - 10**-2
    claims_severity = np.exp(claims_severity) - 10**-2

    # ตรวจสอบค่าสำหรับ claims_frequency และ claims_severity
    if claims_frequency < 0 or claims_severity < 0:
        claims_frequency = 0
        claims_severity = 0

    # การปัดค่า
    if claims_frequency > 0.5:
        claims_frequency = int(claims_frequency + 0.5)
    else:
        claims_frequency = int(claims_frequency)

    if claims_frequency == 0:
        claims_severity = 0

    if claims_severity == 0:
        claims_frequency = 0

    # ปัดค่า claims_severity
    claims_severity = round(claims_severity, 4)
<<<<<<< HEAD

=======
    
>>>>>>> 59b64e29203428e3e919f78bb96a2ba798f0f8f4
    return jsonify({
        'claims_frequency': claims_frequency,
        'claims_severity': claims_severity
    })

if __name__ == "__main__":
<<<<<<< HEAD
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
=======
    # กำหนดให้แอปใช้พอร์ตจากตัวแปรสภาพแวดล้อม (PORT)
    port = int(os.environ.get("PORT", 5000))  # กำหนดพอร์ตเริ่มต้นที่ 5000 ถ้าไม่มี
    app.run(host="0.0.0.0", port=port)
>>>>>>> 59b64e29203428e3e919f78bb96a2ba798f0f8f4
