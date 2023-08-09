from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        title = request.form['titleText']
        body = request.form['bodyText']
        method = request.form['method']

        if method == 'method1':
            return real()
        elif method == 'method2':
            return fake()
        else:
            return f"Title: {title} , Body: {body}"

    else:
        return render_template('index.html')

@app.route('/real', methods=['GET', 'POST'])
def real():
    return render_template('real.html')

@app.route('/fake', methods=['GET', 'POST'])
def fake():
    return render_template('fake.html')

if __name__ == '__main__':
    app.run(host='localhost', port=3000)
