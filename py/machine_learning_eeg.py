import os
from mne import read_epochs
from mne.preprocessing import read_ica
from matplotlib import pyplot
from numpy import vstack, array, mean, std, round
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

def slice_epochs(epochs, step_size=0.01):
    start, stop = epochs.times[0], epochs.times[-1]
    n_steps = int((stop-start)/step_size)
    for i in range(n_steps):
        epoch_slice = epochs.copy()
        epoch_slice.crop(tmin=start, tmax=start+step_size)
        start += step_size
        accuracy = kfold_svm(epoch_slice,k=5, kernel="linear", decision_function_shape="ovo")
        print("classifying interval from %s to %s yields an accuracy of %s" % (start, start+step_size, round(mean(accuracy)*100, decimals=1)))


def kfold_svm(epochs, k=5, scale_data=True, test_size=0.5, kernel="linear", decision_function_shape="ovo"):

    epoch_data = [epoch.T for epoch in epochs._data]
    targets=epochs_ica.events[:,2]
    data = create_dataset(epoch_data, targets)
    kf = KFold(n_splits=k)
    accuracy=list()
    for train, test in kf.split(data):
        X_train, y_train = data[train][:, :-1], data[train][:, -1]
        X_test, y_test = data[test][:, :-1], data[test][:, -1]

        if scale_data: #scale the data
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.fit_transform(X_test)

        clf = SVC(kernel=kernel, decision_function_shape=decision_function_shape)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        #print(classification_report(y_test,y_pred))
        accuracy.append(accuracy_score(y_test, y_pred))

    return accuracy

def svm_classifier(epochs, scale_data=True, test_size=0.5, kernel="linear", decision_function_shape="ovo"):

    epoch_data = [epoch.T for epoch in epochs._data]
    targets=epochs_ica.events[:,2]
    data = create_dataset(epoch_data, targets)

    X, y = data[:, :-1], data[:, -1] # split into inputs and outputs
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size)

    if scale_data: #scale the data
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

    clf = SVC(kernel=kernel, decision_function_shape=decision_function_shape)
    clf.fit(X_train, y_train)
    pred_clf = clf.predict(X_test)
    print(classification_report(y_test,pred_clf))

def compare_models(epochs):

    epoch_data = [epoch.T for epoch in epochs_ica._data]
    targets=epochs_ica.events[:,2]
    data = create_dataset(epoch_data, targets, n_vars=epoch_data[0].shape[1], n_steps=epoch_data[0].shape[0])
    X, y = data[:, :-1], data[:, -1] # split into inputs and outputs

    models, names = list(), list() # create a list of models to evaluate
    models.append(LogisticRegression())
    names.append('LR') # logistic
    models.append(KNeighborsClassifier())
    names.append('KNN') # knn
    models.append(DecisionTreeClassifier())
    names.append('CART') # cart
    models.append(SVC())
    names.append('SVM') # svm
    models.append(RandomForestClassifier())
    names.append('RF') # random forest
    models.append(GradientBoostingClassifier())
    names.append('GBM') # gbm

    # estimtate pefromance accuracy for each algorithm, using 5-fold cross-validation
    all_scores = list()
    for i in range(len(models)):
    	# create a pipeline for the model
    	s = StandardScaler()
    	p = Pipeline(steps=[('s',s), ('m',models[i])])
    	scores = cross_val_score(p, X, y, scoring='accuracy', cv=5, n_jobs=-1)
    	all_scores.append(scores)
    	# summarize
    	m, s = mean(scores)*100, std(scores)*100
    	print('%s %.3f%% +/-%.3f' % (names[i], m, s))
    pyplot.boxplot(all_scores, labels=names)
    pyplot.show()

def create_dataset(sequences, targets):

    transformed = list() 	# create the transformed dataset
    # process each trace in turn
    for i in range(len(sequences)):
        seq = sequences[i]
        vector = list()

        for row in range(1, sequences[0].shape[0]+1):
            for col in range(sequences[0].shape[1]):
                vector.append(seq[-row, col])
        # add output
        vector.append(targets[i])
        # store
        transformed.append(vector)
        # prepare array
    transformed = array(transformed)
    transformed = transformed.astype('float32')
    return transformed

if __name__ =="__main__":
    os.environ["SUBJECT"] = "eegl03"
    epochs = read_epochs(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"_Augenmitte-epo.fif"))
    ica = read_ica(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"-ica.fif"))
    epochs_ica = ica.apply(epochs) #reject components 0 (blinks) & 5(eye-movement)
    epochs_ica.apply_baseline(baseline=(0.5,0.6)) # use 100ms before stimulus onset as baseline
    epochs_ica.crop(tmin=0.5,tmax=1.0)
    epochs_ica = epochs_ica["1","2","3"]
    slice_epochs(epochs_ica)
    #accuracy = kfold_svm(epochs_ica,k=5, test_size=0.5, kernel="linear", decision_function_shape="ovo")
    #print(mean(accuracy))
