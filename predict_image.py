import argparse, json
from pathlib import Path
import joblib
from train_and_evaluate import classical_features

CLASSES = ["blackspot", "canker", "fresh", "grenning"]

def main():
    p=argparse.ArgumentParser(description="Predict orange class using trained HSV+HOG SVM")
    p.add_argument("--image",required=True,help="Path to an image")
    p.add_argument("--model",default="outputs/svm_hog_hsv.joblib")
    args=p.parse_args()
    model=joblib.load(args.model)
    pred=model.predict(classical_features(args.image).reshape(1,-1))[0]
    print(json.dumps({"image":args.image,"predicted_class":str(pred),
                      "note":"Prediction is for the orange dataset labels, not avocado ripening stages."},indent=2))

if __name__=="__main__": main()
