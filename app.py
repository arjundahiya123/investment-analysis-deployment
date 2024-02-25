# Import the Flask class from the flask module
from flask import Flask, render_template, request

import Models.ARIMA

# Create an instance of the Flask class
app = Flask(__name__)

# # Register a route
# @app.route('/', methods = ['GET', 'POST'])
# def home():
#     text = ''
#     if request.method == 'POST':
#         text = request.form.get('content', '')
#         text = f"You entered: {text}"
#     return render_template('index.html', text=text)
    
###
@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    stock = request.form.get('content')
    try:
        prediction = "Predicted value for tomorrow:" + round(Models.ARIMA.predictNextDayClose(stock),2)
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