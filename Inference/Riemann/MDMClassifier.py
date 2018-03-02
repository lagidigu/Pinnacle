import os

from Meditation.Control_To_Meditation_Classification.DataStructureParser import dataStructureParser

os.chdir("C:/Users/JimBobKingJambo/PycharmProjects/EEG_Final/Praktikum-Neuronale-Netze")

dataStructureParser().getFeaturesAndLabels(useCropping=False)

# trainX, trainY, testX, testY, validationX, validationY = dataStructureParser().getFeaturesAndLabels(useCropping=False)
# print(len(trainX), len(testX), len(validationX))
# trainX = np.array(trainX)
# trainY = np.argmax(trainY, axis = 1)
#
# testX = np.array(testX)
# testY = np.argmax(testY, axis = 1)
#
# covest = Covariances(estimator='lwf')
# ts = TangentSpace()
# svc = SVC(kernel='linear')
#
# clf = make_pipeline(covest,ts,svc)
# clf.fit(trainX, trainY)
#
# y_pred = clf.predict(testX)
# print(accuracy_score(testY, y_pred,True))



# test_array = np.array([testX[val_one], testX[val_two], testX[val_three], testX[val_four]])
# print(clf.predict(test_array))
# print(testY[val_one], testY[val_two], testY[val_three], testY[val_four])
#accuracy = cross_val_score(clf, trainX, trainY)
#print(accuracy.mean())