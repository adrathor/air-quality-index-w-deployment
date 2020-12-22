from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle

loaded_model=pickle.load(open("random_forest.pkl","rb"))
app= Flask(__name__)
@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict",methods=["POST"])
def predict():
    data= pd.read_csv("real_2018.csv")
    rf= loaded_model.predict(data.iloc[:,:-1].values)
    prediction=rf.tolist()
    return render_template("result.html", prediction=prediction)

if __name__=="__main__":
    app.run(debug=True)