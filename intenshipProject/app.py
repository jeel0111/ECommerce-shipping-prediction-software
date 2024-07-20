from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('finalized_knn_regression.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/result", methods=["POST"])
def submit():
    # Extract form data
    global cost_of_the_product, weight_in_gms, discount_offered
    global warehouse_block_A, warehouse_block_B, warehouse_block_C, warehouse_block_D, warehouse_block_F
    global mode_of_shipment_Flight, mode_of_shipment_Road, mode_of_shipment_Ship
    global product_importance_high, product_importance_low, product_importance_medium
    global customer_care_calls, prior_purchases, gender

    if request.method == "POST":
        warehouse_block = request.form["warehouse_block"]
        if warehouse_block == "A":
            warehouse_block_A = 1
            warehouse_block_B = 0
            warehouse_block_C = 0
            warehouse_block_D = 0
            warehouse_block_F = 0
        elif warehouse_block == "B":
            warehouse_block_A = 0
            warehouse_block_B = 1
            warehouse_block_C = 0
            warehouse_block_D = 0
            warehouse_block_F = 0
        elif warehouse_block == "C":
            warehouse_block_A = 0
            warehouse_block_B = 0
            warehouse_block_C = 1
            warehouse_block_D = 0
            warehouse_block_F = 0
        elif warehouse_block == "D":
            warehouse_block_A = 0
            warehouse_block_B = 0
            warehouse_block_C = 0
            warehouse_block_D = 1
            warehouse_block_F = 0
        elif warehouse_block == "F":
            warehouse_block_A = 0
            warehouse_block_B = 0
            warehouse_block_C = 0
            warehouse_block_D = 0
            warehouse_block_F = 1

        mode_of_shipment = request.form["mode_of_shipment"]
        if mode_of_shipment == "Ship":
            mode_of_shipment_Flight = 0
            mode_of_shipment_Road = 0
            mode_of_shipment_Ship = 1
        elif mode_of_shipment == "Flight":
            mode_of_shipment_Flight = 1
            mode_of_shipment_Road = 0
            mode_of_shipment_Ship = 0
        elif mode_of_shipment == "Road":
            mode_of_shipment_Flight = 0
            mode_of_shipment_Road = 1
            mode_of_shipment_Ship = 0

        product_importance = request.form["product_importance"]
        if product_importance == "low":
            product_importance_high = 0
            product_importance_low = 1
            product_importance_medium = 0
        elif product_importance == "medium":
            product_importance_high = 0
            product_importance_low = 0
            product_importance_medium = 1
        elif product_importance == "high":
            product_importance_high = 1
            product_importance_low = 0
            product_importance_medium = 0

        customer_care_calls = int(request.form["customer_care_calls"])
        gender = int(request.form["gender"])
        prior_purchases = int(request.form["prior_purchases"])
        cost_of_the_product = int(request.form["cost_of_the_product"])
        discount_offered = int(request.form["discount_offered"])
        weight_in_gms = int(request.form["weight_in_gms"])

    # Prepare the feature array for prediction
    x = np.array([cost_of_the_product, weight_in_gms, discount_offered,
                  warehouse_block_A, warehouse_block_B, warehouse_block_C, warehouse_block_D, warehouse_block_F,
                  mode_of_shipment_Flight, mode_of_shipment_Road, mode_of_shipment_Ship,
                  product_importance_high, product_importance_low, product_importance_medium,
                  customer_care_calls, prior_purchases, gender])
    x = x.reshape((1, -1))

    # Make the prediction (regression output)
    prediction = model.predict(x)[0]
    prediction = int(prediction*100)
    # Render result.html with the prediction
    return render_template('home.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
