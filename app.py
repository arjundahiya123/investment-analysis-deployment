# Import the Flask class from the flask module
from flask import Flask, render_template, request

import Models.ARIMA, Models.lstm

# Create an instance of the Flask class
app = Flask(__name__)
    
###
@app.route('/')
def home():
    return render_template("index.html")

# @app.route("/predict", methods=['POST'])
# def predict():
#     stock = request.form.get('content')
#     try:
#         prediction = "Predicted value for tomorrow: " + str(round(Models.ARIMA.predictNextDayClose(stock),2))
#     except ValueError as e:
#         error_message = str(e)
#         if "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day." in error_message:
#             prediction = "UNAVAILABLE - The maximum amount of daily API calls has been reached. Please try again tomorrow."
#         else:
#             prediction = "UNAVAILABLE - An error occurred: " + error_message
#     return render_template("index.html", prediction=prediction, stock=stock)

@app.route("/predict", methods=['POST'])
def predict():
    stock = request.form.get('content')
    model_type = request.form.get('model_type')

    if model_type == 'ARIMA':
        try:
            prediction = "Predicted value for tomorrow: " + str(round(Models.ARIMA.predictNextDayClose(stock),2))
        except ValueError as e:
            error_message = str(e)
            if "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day." in error_message:
                prediction = "UNAVAILABLE - The maximum amount of daily API calls has been reached. Please try again tomorrow."
            else:
                prediction = "UNAVAILABLE - An error occurred: " + error_message
    elif model_type == 'LSTM':
        try:
            prediction = "Predicted value for tomorrow: " + str(round(Models.lstm.predictNextDayClose(stock),2))
        except ValueError as e:
            error_message = str(e)
            if "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day." in error_message:
                prediction = "UNAVAILABLE - The maximum amount of daily API calls has been reached. Please try again tomorrow."
            else:
                prediction = "UNAVAILABLE - An error occurred: " + error_message

    return render_template("index.html", prediction=prediction, stock=stock)


# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5002, debug=True)
