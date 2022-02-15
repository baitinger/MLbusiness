# USAGE
# Start the server:
# 	python run_server.py
# Submit a request via Python:
#   Prediction.ipynb
# import the necessary packages
import dill
import pandas as pd
import flask
import os

dill._dill._reverse_typemap['ClassType'] = type

app = flask.Flask(__name__)


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    return model


# Please make sure you've set correct path to ada_model.dill
modelpath = "ada_model.dill"
model = load_model(modelpath)


@app.route("/", methods=["GET"])
def general():
    return "Predicting diabetes using flask rest api server. Please use <b>/predict</b> route to receive response. " \
           "Please use the following features for a prediction:" \
           "<br>" \
           "<ul>" \
           "<li><b>Pregnancies</b> - the condition of being pregnant.</li>" \
           "<li><b>Glucose</b> - A simple monosaccharide (sugar)</li>" \
           "<li><b>BloodPressure</b> - Blood pressure is the force of your blood pushing " \
           "against the walls of your arteries. Each time your heart beats, it pumps blood into the arteries.</li>" \
           "<li><b>SkinThickness</b> - We measured skin thickness in 66 IDDM patients aged 24–38 yr and " \
           "investigated whether it correlated with long-term glycemic control and the presence of certain</li>" \
           "<li><b>Insulin</b> - A polypeptide hormone that regulates carbohydrate metabolism.</li>" \
           "<li><b>BMI</b> - Inclusion-body myositis (IBM) primarily affects men</li>" \
           "<li><b>DiabetesPedigreeFunction</b> - a function which scores " \
           "likelihood of diabetes based on family history</li>" \
           "<li><b>Age</b> - patient age (years)</li>" \
           "<li><b>Outcome</b> - target feature (0 if non-diabetic, 1 if diabetic)</li>" \
           "</ul>"


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        try:
            request_json = flask.request.get_json()
            preds = model.predict_proba(pd.DataFrame({"pregnancies": [request_json['pregnancies']],
                                                      "glucose": [request_json['glucose']],
                                                      "bloodpressure": [request_json['bloodpressure']],
                                                      "skinthickness": [request_json['skinthickness']],
                                                      "insulin": [request_json['insulin']],
                                                      "bmi": [request_json['bmi']],
                                                      "diabetespedigreefunction": [
                                                          request_json['diabetespedigreefunction']],
                                                      "age": [request_json['age']]}))
            data["predictions"] = preds[:, 1][0]
            data["success"] = True
        except Exception as e:
            data["predictions"] = str(e)

    return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    port = int(os.environ.get('PORT', 8180))
    app.run(host='0.0.0.0', debug=True, port=port)
