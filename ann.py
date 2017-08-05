import numpy as np

# Weights
defaultWeights = np.array([[0.5, 0.8]]).T

# Tunning vars
defaultAlpha = 0.6
defaultTheta = 0.4

# print inputDataSet
# print weights
# print outputDataSet

# syn0 = 2*np.random.random((3,1)) - 1

# print syn0

# sigmoid function
def nonlin(x, useTheta=False, t = 0):
    if useTheta == True:
        if x >= t:
            return 1
        return 0
    return 1/(1+np.exp(-x))

def validateANN(i, o, w, t):
    net = np.dot(i, w)

    h = np.array([])
    for n in net:
        h = np.append(h, nonlin(n, True, t))
    h = np.array([h]).T

    return all(h == o)

def trainANN(inputDataSet, outputDataSet, W, alpha, theta):
    trainingSuccess = 0
    trainingCycles = 0
    while trainingSuccess < len(inputDataSet) and trainingCycles < 100:

        trainingCycles += 1
        # print "\n\n\nTraining Cycle : ", trainingCycles

        index = 0
        for x in inputDataSet:
            # print "\n\ncurrent input dataSet x : ", x
            # print "current weights : ", W.T

            # Compute Net
            net = np.dot(x, W)[0]
            # print "net : ", net


            # Compute Hypothesis
            h = nonlin(net, True, theta)

            # Compute Error
            o = outputDataSet[index]
            # print "current output : ", o
            index += 1

            t = o[0]
            e = t - h
            # print "t : ", o
            # print "h : ", h
            # print "e : ", e

            if e == 0:
                trainingSuccess += 1
            else:
                trainingSuccess = 0
                # Backword propagation
                # W = W + DeltaW
                # W = W + a.e.x

                deltaW = alpha * e * x
                deltaW = np.array([deltaW]).T

                W += deltaW

                # print "deltaW : ", deltaW.T
                # print "modified weights : ", W.T

            if trainingSuccess == len(inputDataSet):
                break

    return trainingSuccess, trainingCycles



def buildANNTrainingStatsForDataSetWithRange(inputDataSet, outputDataSet, range = [-0.1, 1.0], changeTheta = False, changeAlpha = False):

    currentRange = range[0]
    successfulTrainings = 0

    theta = defaultTheta
    alpha = defaultAlpha
    totalTrainings = 0

    while currentRange <= range[1]:
        # Weights
        weights = np.array(defaultWeights)

        if changeTheta:
            theta = currentRange

        if changeAlpha:
            alpha = currentRange

        currentRange += 0.1
        totalTrainings += 1

        trainingSuccess, trainingCycles = trainANN(inputDataSet, outputDataSet, weights, alpha, theta)

        if trainingSuccess == len(inputDataSet):
            print ">>>>> Training Successful after", trainingCycles, "training cycles, for alpha", alpha, ", for theta", theta, " with Final Weights : ", weights.T
            successfulTrainings += 1
            print "ANN Validated After Training :", validateANN(inputDataSet, outputDataSet, weights, theta)

        else:
            print "Training NOT Successful after training cycles ", trainingCycles, ", for alpha", alpha, ", for theta", theta, " with Final Weights : ", weights.T

    print "\nTotal Successful Trainings : ", successfulTrainings, "/", totalTrainings


def buildANNTrainingStatsForDataSet(inputDataSet, outputDataSet):
    range = [-1.0, 1.0]
    print "\n\n====== ANN Training Stats with Theta Variation, alpha =", defaultAlpha, ", theta =", range, ", starting weights =", defaultWeights.T, "======== \n"
    buildANNTrainingStatsForDataSetWithRange(inputDataSet, outputDataSet, range, changeTheta = True)


def buildANNTrainingStats():

    print "\n\n"
    print "=============================="
    print "======== AND Training ========"
    print "=============================="

    # Input Dataset
    inputDataSet = np.array([[0, 0],
                             [0, 1],
                             [1, 0],
                             [1, 1]])

    # Output Dataset
    outputDataSet = np.array([[0, 0, 0, 1]]).T

    buildANNTrainingStatsForDataSet(inputDataSet, outputDataSet)

    # print "\n\n"
    # print "=============================="
    # print "======== Random Training ========"
    # print "=============================="
    #
    # # Input Dataset
    # inputDataSet = np.array([[0, 0],
    #                          [0, 1],
    #                          [1, 0],
    #                          [1, 1]])
    #
    # # Output Dataset
    # outputDataSet = np.array([[0, 0, 1, 0]]).T

    # buildANNTrainingStatsForDataSet(inputDataSet, outputDataSet)

    print "\n\n"
    print "=============================="
    print "======== OR Training ========="
    print "=============================="

    # Input Dataset
    inputDataSet = np.array([[0, 0],
                             [0, 1],
                             [1, 0],
                             [1, 1]])

    # Output Dataset
    outputDataSet = np.array([[0, 1, 1, 1]]).T

    buildANNTrainingStatsForDataSet(inputDataSet, outputDataSet)

    # print "\n\n"
    # print "=============================="
    # print "======== XOR Training ========="
    # print "=============================="
    #
    # # Input Dataset
    # inputDataSet = np.array([[0, 0],
    #                          [0, 1],
    #                          [1, 0],
    #                          [1, 1]])
    #
    # # Output Dataset
    # outputDataSet = np.array([[0, 1, 1, 0]]).T
    #
    # buildANNTrainingStatsForDataSet(inputDataSet, outputDataSet)

    # print "\n\n"
    # print "=============================="
    # print "======== Positive Negitive Training ========"
    # print "=============================="
    #
    # # Input Dataset
    # inputDataSet = np.array([[0, 0],
    #                          [1, 1],
    #                          [2, -1],
    #                          [1, 1]])
    #
    # # Output Dataset
    # outputDataSet = np.array([[1, 1, 1, 1]]).T
    #
    # buildANNTrainingStatsForDataSet(inputDataSet, outputDataSet)



buildANNTrainingStats()