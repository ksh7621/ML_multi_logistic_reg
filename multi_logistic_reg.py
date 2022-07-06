# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


def zscore_norm(data_list):
    normalized_data = []
    for data in data_list:
        z_score = (data-np.mean(data))/np.std(data)
        normalized_data.append(z_score)
    return normalized_data


class LogisticRegression:
    def __init__(self, learning_rate = 0.1, epoch = 2000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.w = []
        self.b = 0
        
    def initialize_weight(self,dim, category_shape):
        w = np.random.normal(0,category_shape,(dim,category_shape))  # W shape: [784, 10]
        b = np.random.rand(category_shape)  # b shape: [10]
        return w,b 
    
    def sigmoid(self, x):
        output = 1/(1+ np.exp(-x))
        return output
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def hypothesis(self, w, X, b):
        y_hat = self.softmax(np.matmul(X,w)+b)
        return y_hat
    
    def BCE_cost(self,y_hat,y,N):
        cost = -(1/N)*np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))
        cost = np.squeeze(cost)
        return cost
    
    def CE_cost(self,y_hat,y,N):
        yyy = -np.sum(y*np.log(y_hat+1e-3), axis=1)  # shape: [class_no] 클래스 차원 합
        output = yyy.sum()  # shape: [1] 데이터 개수 합
        return output
    
    def cal_gradient(self,w,y_hat,X,y):
        N = X.shape[0]
        delta_w =  (1/N)*np.matmul(X.T,(y_hat-y))             
        delta_b = (1/N)*np.sum(y_hat-y)
        grads = {"delta_w":delta_w,
                 "delta_b":delta_b}
        return grads
    
    def gradient_position(self,w,b,X,y):
        N = X.shape[0]
        y_hat = self.hypothesis(w,X,b)
        cost = self.CE_cost(y_hat,y,N)
        grads = self.cal_gradient(w,y_hat,X,y)
        return grads, cost
    
    def gradient_descent(self, w, b, X, y, print_cost = False):
        costs = []
        for i in range(self.epoch):
            
            grads, cost = self.gradient_position(w,b,X,y)
            
            delta_w = grads["delta_w"]
            delta_b = grads["delta_b"]
            
            # delta_w = delta_w.reshape(-1,1)
            w = w-(self.learning_rate * delta_w)
            b = b-(self.learning_rate * delta_b)
            
            if i % 100 == 0:
                print('iter: ', i)
                costs.append(cost)
            
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" %(i, cost))
    
            params = {"w": w, 
                      "b": b}

            grads = {"delta_w":delta_w,
                     "delta_b":delta_b}

        return params, costs
    
    def train_model(self, X_train,Y_train, X_test, Y_test, print_cost = False):
        dim = np.shape(X_train)[1]
        w,b = self.initialize_weight(dim, Y_train.shape[1])
        parameters, costs = self.gradient_descent(w,b,X_train,Y_train,print_cost = False)
        
        self.w = parameters["w"]
        self.b = parameters["b"]
        
        _, Y_prediction_test = self.predict(X_test, Y_test)
        _, Y_prediction_train = self.predict(X_train, Y_train)     
  
        
        train_score = 100 - np.mean(np.abs(Y_prediction_train-Y_train)) * 100
        test_score = 100 - np.mean(np.abs(Y_prediction_test-Y_test)) * 100
        
        print("test accuracy: {} %".format(100-np.mean(np.abs(Y_prediction_test - Y_test))* 100))
        
        result_dict = {"costs" : costs,
                      "Y_prediction_test" : Y_prediction_test,
                      "Y_prediction_train" : Y_prediction_train,
                      'w' : self.w,
                      "b" : self.b,
                      "learning_rate" : self.learning_rate,
                      "num_iterations": self.epoch,
                      "train accuracy" : train_score,
                      "test accuracy" : test_score
                      }
        
        return result_dict
    
    def predict(self,X,label):
        X = np.array(X)
        
        N = X.shape[0]
        Y_prediction = np.zeros([N,10])
        
        w = self.w
        b = self.b
        
        y_hat = self.hypothesis(w,X,b)  # shape: [10000, 10]    
        
        return Y_prediction, y_hat
        

if __name__ == "__main__":  
    
    # Data load
    X, y = fetch_openml("mnist_784", version=1,return_X_y = True, as_frame=False)
    _X = zscore_norm(X)
    
    train_X = np.array(_X[:-10000])
    test_X = np.array(_X[-10000:])
    
    train_y = y[:-10000].astype(np.int)  # onehot encoding
    test_y = y[-10000:].astype(np.int)
    
    n_values = np.max(train_y) + 1
    train_y = np.eye(n_values)[train_y]
    test_y = np.eye(n_values)[test_y]   
    
    
    # Training
    LR_cls = LogisticRegression(learning_rate=0.1, epoch=1000)
    result_dict = LR_cls.train_model(train_X,train_y,test_X,test_y)
   
   
    # Test    
    n = np.random.randint(50,size=10)
    for i in range(10):
        sample_idx = n[i]
        _, pred = LR_cls.predict(test_X[sample_idx], test_y[sample_idx])       
        pred_class = np.argmax(pred)
       
        plt.imshow(np.resize(test_X[sample_idx,:], (28,28)))
        plt.title('Label: {}\n Prediction: {}'.format(np.argmax(test_y[sample_idx]),pred_class))
        plt.show()
        
   
    

    
