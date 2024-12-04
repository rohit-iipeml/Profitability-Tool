from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('tool.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the JSON request
        data = request.json

        # Extract features from the JSON data
        frequency_factor = data['frequency_factor']
        avg_daily_drawdown_percentage = data['avg_daily_drawdown_percentage']
        win_rate = data['win_rate']
        trader_type = data['trader_type']
        avg_win = data['avg_win']
        avg_trade_drawdown = data['avg_trade_drawdown']

        # Map trader type to an integer
        trader_type_mapping = {'intraday': 0, 'swing': 1, 'scalper': 2}
        trader_type_encoded = trader_type_mapping.get(trader_type.lower())

        if trader_type_encoded is None:
            return jsonify({'error': 'Invalid trader type'}), 400

        # Make predictions using your model
        prediction = model.predict([[
            frequency_factor,
            avg_daily_drawdown_percentage,
            win_rate,
            trader_type_encoded
        ]])

        # Additional feature: Calculate avg_trade_drawdown/avg_win ratio
        avg_trade_drawdown_ratio = avg_trade_drawdown / avg_win
        
        # print("avg_trade_drawdown:", avg_trade_drawdown)
        # print("avg_win:", avg_win)
        # print("avg_trade_drawdown_ratio:", avg_trade_drawdown_ratio)
        # print(prediction)
        # Return the prediction result and recommendation
        if prediction[0] == 'Good':  # Good prediction
            recommendation = 'Recommended' if avg_trade_drawdown_ratio >= -1 else 'Not Recommended'
        else:  # Bad prediction
            recommendation = 'Not Recommended'

        result = {'prediction': 'Good' if prediction[0] == 'Good' else 'Bad', 'recommendation': recommendation}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app on localhost:8000
    app.run(host="0.0.0.0", port=8000)
