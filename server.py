from flask import Flask, request, jsonify
import numpy
import pickle

app = Flask("NLPEmotionApp")
model = pickle.load("nlpemotionmodel.pkl", "rb")

@app.route('/model', methods=['POST'])
def prediction():
    setence = request.get_json(force=True)
    result = model.predict(sentence["userinput"])
    output = result[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
