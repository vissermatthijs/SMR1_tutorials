# Load libraries
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.datasets import load_wine


dataset = datasets.load_wine()

X_train, X_test, Y_train, Y_test = train_test_split(dataset.data, dataset.target, test_size=0.60, random_state=43)

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

lda = LinearDiscriminantAnalysis(n_components=3)
X_train_lda = lda.fit_transform(X_train_norm, Y_train)
X_test_lda = lda.transform(X_test_norm)

svm = SVC(kernel="rbf", C=1.0, random_state=0)
svm.fit(X_train_lda, Y_train)

z = svm.predict(X_test_lda)

print ('accuracy: %0.2f' % accuracy_score(Y_test, z))