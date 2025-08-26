from flask import Flask, render_template, request, jsonify
import requests
from datetime import datetime

app = Flask(__name__, template_folder="templates")

# Replace with your API base URL if running separately
API_BASE = "http://192.168.31.195:5000/"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    try:
        form = request.form.to_dict()

        # Convert numeric fields
        try:
            form["teacher_number_of_previously_posted_projects"] = int(
                form.get("teacher_number_of_previously_posted_projects", 0)
            )
        except ValueError:
            return jsonify({"error": "Invalid number for previously posted projects."})

        # Ensure datetime is in ISO format
        dt = form.get("project_submitted_datetime")
        if dt:
            try:
                form["project_submitted_datetime"] = datetime.fromisoformat(dt).isoformat()
            except ValueError:
                return jsonify({"error": "Invalid datetime format. Use ISO format."})

        # Parse resources (from dynamic form fields)
        resources = []
        idx = 0
        while f"resource_desc_{idx}" in form:
            try:
                desc = form.get(f"resource_desc_{idx}")
                qty = int(form.get(f"resource_qty_{idx}", 0))
                price = float(form.get(f"resource_price_{idx}", 0))
                if desc:
                    resources.append({"description": desc, "quantity": qty, "price": price})
            except ValueError:
                return jsonify({"error": f"Invalid resource entry at index {idx}."})
            idx += 1

        form["resources"] = resources

        # Send to backend API
        r = requests.post(f"{API_BASE}/predict", json=form)
        if r.status_code != 200:
            return jsonify({"error": f"API error {r.status_code}: {r.text}"})

        try:
            return jsonify(r.json())
        except Exception:
            return jsonify({"error": "Invalid response from API. Ensure backend returns JSON."})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)