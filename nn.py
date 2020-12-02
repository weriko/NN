import numpy as np


#Added logistic
#Add regressor
class NeuralNetwork:
    def __init__(self, layer_sizes, layer_activations, epsilon = 0.1, lr=0.1):
        self.epsilon = epsilon
        self.layer_sizes = layer_sizes
        self.layer_activations = layer_activations
        self.initialize()
        self.lr = lr
        self.activation_functions = {
            "relu":self.relu,
            "sigmoid":self.sigmoid,
            "grelu":self.grelu,
            "gsigmoid":self.gsigmoid            
            
            }
        
    def initialize(self):
        self.W = [] #NN weights
        self.b = [] #NN betas
        for i in range(len(self.layer_sizes)-1):
            self.W.append(np.random.randn(self.layer_sizes[i+1],self.layer_sizes[i]  )*self.epsilon) #Randomly initializes weights for all layers 
            self.b.append(np.random.randn(self.layer_sizes[i+1],1)*self.epsilon) #Randomly initializes betas for all layers 

        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    
    
    def gsigmoid(self,da,x):
        temp = self.sigmoid(x)
   
        return da * temp * (1-temp)
    
    
    def relu(self,x):
        return np.maximum(0,x)
    def grelu(self,da,x):

        temp = np.array(da, copy=True)
        temp[x<=0] = 0
        return temp
    

    def propagate(self,x,is_pred=False):
        cacheW = []
     
        z = x
       
      
        #print(z)
        a = self.activation_functions[self.layer_activations[0]](z)
        
        
        for i in range(len(self.W)):
            cacheW.append((a,z))
           

           
            z = np.dot(self.W[i],a ) + self.b[i]
           
            a = self.activation_functions[self.layer_activations[i]](z)
            assert(a.shape == z.shape)
        cacheW.append((a,z))

        if not is_pred:        
            
            self.cache =  cacheW
        else:
            return cacheW[-1][0]
  
    
    
    
    
    def backpropagate(self,y,y_pred):
        gradsW = []
        gradsb= []
     
        
        
        da= -(np.divide(y,y_pred) - np.divide(1-y,1-y_pred))
        #print(da.shape)
        
       
        for i in range(len(self.W))[::-1]:
            
          
            
            a_prev =  self.cache[i][0]
            #print(a.shape)
            
            #print(self.cache[0][0].shape)
           
          
           
            z = self.cache[i+1][1] #Need to get the z for the next layer
            #print(z.shape)
            #print(a.shape)
            
            w = self.W[i]
          
            b = self.b[i]
            
            dz = self.activation_functions["g"+self.layer_activations[i]](da,z)
          
            #print(dz.shape)
            #print(a.shape)
            
            
         
            
            dw = np.dot(dz, a_prev.T)/a_prev.shape[1]
            
            db = np.sum(dz, axis=1,keepdims=True)/a_prev.shape[1]
           
            
            #print(dw.shape)
            da = np.dot(w.T, dz)
            
            #print(a.shape)
           
         
            
            gradsW.append(dw)
            gradsb.append(db)
            
        self.gradsw = gradsW
        self.gradsb = gradsb
        
    def update(self):
        
       
        
       
        for i in range(len(self.gradsw)):
            
            #print(self.gradsw[i].shape)
         
           
            self.W[i] -= self.lr*self.gradsw[-(i+1)] #It updates the weights of the first layer, second layer... the grads lists are inverted because of the way they were stored (last first)
            self.b[i] -= self.lr*self.gradsb[-(i+1)]
        #print(self.W[-1])
          
            
    def fit(self,x,y,epochs=1):
        for i in range(epochs):
            self.propagate(x)
            self.backpropagate(y,self.cache[-1][0])
            self.update()
            
    def predict(self,x):
        a =self.propagate(x,is_pred=True)
        return a
        
        
            
            
            
            
            
        
        
            
        
        
        
        
        
NN = NeuralNetwork([2,5,7,6,4,1],
                   ["relu","relu","relu","sigmoid","sigmoid"])
x = np.array([[566,696],
             [855,784],
             [2,6],
             [8,5],
             [235,35],
             [464,342],
             [634,643],
             [745,455],
             [3,2],
             [6,8],
             [2,6],
             [4,7]])


x= x.T
y = np.array([
    [1],
    [1],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [0],
    [0],
    [0],
    [0]
    ])
y = y.T
xpred = np.array([[457,435],
             [4865,7878],
             [2,3],
             [3,4],
             [754,656],
             [234,765],
             [769,567],
             [457,234],
             [1,3],
             [8,4],
             [7,3],
             [9,6]]).T
NN.fit(x,y,epochs=20)
print(NN.predict(xpred))
#print(NN.W)
        
