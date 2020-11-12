from flask import Flask, flash, jsonify, redirect, render_template, request, session

app = Flask(__name__)

# App autoreloads
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/")
def index():
    """Home page"""
    return render_template("homepage.html")

@app.route("/testMe", methods=["GET"])
def storyGeneration():
    """Testing Page"""
    if request.method == "POST":
        if request.method == "POST":
            print("FORM DATA RECEIVED")

            # Make sure file exists, if not reload
            if "file" not in request.files:
                return redirect(request.url)

            # Make sure file has a name, if not reload
            file = request.files["file"]
            if file.filename == "":
                return redirect(request.url)

            # If file exists, process
            if file:

        return render_template('results.html', result=testResult)