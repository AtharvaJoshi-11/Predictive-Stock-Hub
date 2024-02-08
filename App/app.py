from flask import Flask,render_template ,jsonify,request
import torch
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from torch_utils import Stock_LSTM , load_lstm_model

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    try:
        data = request.get_json()

        symbol = data['symbol']

        print('Received symbol:', symbol)

        stock = yf.Ticker(symbol)
        print(stock)
        stock_data = stock.history(period='1mo')['Close'].values

        print("Stock data:",stock_data.shape)
        scaler = MinMaxScaler(feature_range=(-1,1))
            
        stock_data_scaled = scaler.fit_transform(stock_data.reshape(-1,1))

        stock_data_scaled = stock_data_scaled.reshape(1,-1,1)

        print("Stock data scaled:",stock_data_scaled.shape)
        
        stock_data_tensor = torch.from_numpy(stock_data_scaled).to(torch.float32)
        print("Stock data tensor:",stock_data_tensor)

        model = load_lstm_model()
        print("Model parameters:")
        for name, param in model.named_parameters():
            print(name, param.shape)

        with torch.inference_mode():
            print('Inside torch.inference_mode')
            # dummy_input = torch.randn(1, 20, 1)
            prediction = model(stock_data_tensor)
            print(prediction)

        prediction_unscaled = scaler.inverse_transform(prediction)
        print(prediction_unscaled)

        return jsonify({'symbol': symbol, 'predicted_price': float(prediction_unscaled.item())})
    
    except Exception as e:
        return jsonify({'error': str(e)})
    

if __name__ == '__main__':
    app.run(debug=True)
