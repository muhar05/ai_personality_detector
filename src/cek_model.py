import joblib

model = joblib.load("personality_clf.joblib")
print(model.keys())
print(type(model['vec']))
print(type(model['clf']))