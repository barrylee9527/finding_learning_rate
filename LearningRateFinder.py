from keras.callbacks import LambdaCallback
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile
class LearningRateFinder:
    def __init__(self,model,stopFactor=4,beta=0.98):
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def reset(self):
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None
    def is_data_iter(self,data):
        iterClasses = ["NumpyArrayIterator", "DirectoryIterator",
                       "DataFrameIterator", "Iterator", "Sequence"]
        return data.__class__.__name__ in iterClasses
    def on_batch_end(self,batch,logs):
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)
        l = logs['loss']
        self.batchNum +=1
        self.avgLoss = (self.beta*self.avgLoss)+((1-self.beta)*l)
        smooth = self.avgLoss/(1-(self.beta**self.batchNum))
        self.losses.append(smooth)
        stopLoss = self.stopFactor*self.bestLoss
        if self.batchNum>1 and smooth>stopLoss:
            self.model.stop_training = True
            return
        if self.batchNum==1 or smooth <self.bestLoss:
            self.bestLoss = smooth
        lr *=self.lrMult
        K.set_value(self.model.optimizer.lr,lr)

    def find(self,trainData,startLR,endLR,epochs=None,
             stepsPerEpoch=None,batchSize=32,sampleSize=2048,verbose=1):
        self.reset()
        useGen = self.is_data_iter(trainData)
        if useGen and stepsPerEpoch is None:
            msg = "Using generator without supplying stepsPerEpoch"
            raise Exception(msg)
        elif not useGen:
            numSamples = len(trainData[0])
            stepsPerEpoch = np.ceil(numSamples/float(batchSize))
        if epochs is None:
            epochs = int(np.ceil(sampleSize/float(stepsPerEpoch)))
        numBatchUpdates = epochs*stepsPerEpoch
        self.lrMult = (endLR/startLR)**(1.0/numBatchUpdates)
        self.weightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.weightsFile)
        origLR = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr,startLR)
        callback = LambdaCallback(on_batch_end=lambda batch,logs:
                                  self.on_batch_end(batch,logs))
        if useGen:
            self.model.fit_generator(
                trainData,
                steps_per_epoch=stepsPerEpoch,
                epochs=epochs,
                verbose=verbose,
                callbacks=[callback]
            )
        else:
            self.model.fit(
                trainData[0],trainData[1],
                batch_size=batchSize,
                epochs=epochs,
                callbacks=[callback],
                verbose=verbose
            )
        self.model.load_weights(self.weightsFile)
        K.set_value(self.model.optimizer.lr,origLR)

    def plot_loss(self, skipBegin=10, skipEnd=1, title=""):
        lrs = self.lrs[skipBegin:-skipEnd]
        losses = self.losses[skipBegin:-skipEnd]
        plt.plot(lrs,losses)
        plt.xscale('log')
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
        if title != "":
            plt.title(title)
