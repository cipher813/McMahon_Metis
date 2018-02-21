<<<<<<< HEAD
import pandas as pd
from flask import Flask, request, render_template

# df = pd.read_csv("kickstarter_bostockscatterplot.csv")

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
=======
from flask import Flask, render_template

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def index():
    return render_template('home.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
>>>>>>> cb01e1c4dd08e8661e121dfce6d8fa7da80b2846
