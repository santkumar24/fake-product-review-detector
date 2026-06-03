from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    input_type = request.form.get("input_type")

    # =========================
    # CSV FILE ANALYSIS
    # =========================
    if input_type == "csv":

        file = request.files.get("file")

        if not file or file.filename == "":
            return "Please upload a CSV file."

        try:
            data = pd.read_csv(file)
        except Exception as e:
            return f"Error reading CSV file: {e}"

        if "review" not in data.columns:
            return "CSV file must contain a column named 'review'."

        fake_count = 0
        genuine_count = 0
        results = []

        for review in data["review"]:

            vector = vectorizer.transform([str(review)])
            prediction = model.predict(vector)[0]

            results.append({
                "review": review,
                "prediction": prediction
            })

            if str(prediction).lower() == "fake":
                fake_count += 1
            else:
                genuine_count += 1

        total = fake_count + genuine_count

        trust_score = (
            round((genuine_count / total) * 100, 2)
            if total > 0 else 0
        )

        return render_template(
            "result.html",
            total=total,
            fake=fake_count,
            genuine=genuine_count,
            trust_score=trust_score,
            results=results
        )

    # =========================
    # URL ANALYSIS
    # =========================
    elif input_type == "url":

        url = request.form.get("url")

        if not url:
            return "Please enter a product URL."

        # Future: Add review scraping here
        return f"""
        <h2>URL Received Successfully</h2>
        <p><b>Product URL:</b> {url}</p>
        <p>URL review analysis feature is under development.</p>
        """

    # =========================
    # INVALID INPUT
    # =========================
    else:
        return "Please select CSV or URL input type."


if __name__ == "__main__":
    app.run(debug=True)