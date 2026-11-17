# PARTIE 2 - Reconnaissance de chiffres manuscrits
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split, GridSearchCV

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

parameters = {'kernel':['poly', 'rbf', 'sigmoid'], 'C':[1, 5, 10], 'degree':[5, 6], 'coef0':[1]}
svc = svm.SVC(gamma=0.001, probability=True)

classifier = GridSearchCV(svc, parameters)

classifier.fit(X_train, y_train)

predicted = classifier.predict(X_test)

print("Résultats de la grid search :\n", classifier.best_params_)
print("Meilleur score (CV) :", classifier.best_score_)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

print(
    f"Classification report for classifier {classifier}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()