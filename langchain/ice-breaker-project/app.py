import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify

from ice_breaker import ice_break


# Load environment variables
load_dotenv()


# Initialize Flask app
app = Flask(__name__)


@app.route("/process", methods=["POST"])
def process():
    payload = json.loads(request.data)
    name = payload.get("name")
    reference = payload.get("reference")

    linkedin_profile_url = ice_break(name=name, reference=reference)

    return jsonify(linkedin_profile_url)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
