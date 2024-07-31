import bentoml

from sklearn import svm
from sklearn import datasets

# Load the training dataset
iris = datasets.load_iris()

X,y = iris.data, iris.target

# Train the model
clf = svm.SVC(gamma='scale')
clf.fit(X,y)

# Save the model to the Bentoml local model store
save_model = bentoml.sklearn.save_model("iris_clf",clf)
print(f"Model Saved: {save_model}")


# "iris_clf:lse2fv2pmg2rdxct"