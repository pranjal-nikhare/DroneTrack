from flask import Flask, render_template,send_file
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/execute_tello')
def execute_tello():
    subprocess.run(["python", "tello.py"])
    return "Tello script executed successfully!"

@app.route('/execute_testing')
def execute_testing():
    subprocess.run(["python", "testing.py"])
    return send_file('output.txt', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
