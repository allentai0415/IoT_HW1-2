from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.dates import DateFormatter

app = Flask(__name__)

# 讀取並處理資料
data = pd.read_csv('2330-training.csv')  # 修改為上傳的檔案路徑
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data['y'] = pd.to_numeric(data['y'], errors='coerce')
data['x1'] = pd.to_numeric(data['x1'], errors='coerce')
data.dropna(inplace=True)

# 自回歸模型訓練
X = data[['x1']]
y = data['y']
model = LinearRegression()
model.fit(X, y)
data['predicted_y'] = model.predict(X)

@app.route('/')
def home():
    # 繪製預測結果圖表
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['y'], label='Actual Values', color='blue')
    plt.plot(data['Date'], data['predicted_y'], label='Predicted Values', linestyle='--', color='red')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('Auto Regression Prediction')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()

    # 將圖表轉換為字串格式並顯示
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
