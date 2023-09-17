import requests
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from utils.model import RobertaClass
from utils.inference import load_model, inference

app = Flask(__name__)
CORS(app)

API_URL = "https://newsapi.org/v2/top-headlines?country=us&apiKey=f06ef7749b404755ba9f85d18f23d07a"
MODEL_PATH = "model/roberta_model/pytorch_roberta_news_1.bin"

LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']

model = load_model(MODEL_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/fetch_news")
def fetch_news():
    response = requests.get(API_URL)
    if response.status_code == 200:
        print("DEBUG: NEWS API successful.")
        news_data = response.json()["articles"]
        for news_item in news_data:
            description = news_item["description"]
            if description:
                predicted_class = inference(model, description)
                # news_item["category"] = f"Category {predicted_class+1}"
                news_item["category"] = LABELS[predicted_class]
            else:
                news_item["category"] = "Unknown"
        print("DEBUG: Predictions available.")
        return jsonify(news_data)

    return jsonify({"error": "Failed to fetch news"})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
