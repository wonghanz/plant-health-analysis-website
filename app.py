from flask import Flask, render_template
from plant_health import analyze_plant_health

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    result = analyze_plant_health()
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
