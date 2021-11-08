import numpy as np                                              
class matrix_factorization():

    def __init__(self,data,features):
        
        self.data = data
        self.features = features
        self.user_count = data.shape[0]
        self.item_count = data.shape[1]
        self.user_features = np.random.uniform(low=0.1,high=0.9,size=(self.user_count,self.features))
        self.item_features = np.random.uniform(low=0.1,high=0.9,size=(self.features,self.item_count))

    def MSE(self):
        """
        Mean squared error function comparing dot product of user-feature row and feature-item column to user-item cell
        """

        matrix_product = np.matmul(self.user_features, self.item_features)
        return np.sum((self.data - matrix_product)**2)
    
    def single_gradient(self,user_row,item_col,wrt_user_idx=None,wrt_item_idx=None):
        """
        Computes gradient of single user-item cell to a single user-feature or feature-item cell
        """

        if wrt_user_idx!=None and wrt_item_idx!=None:
            return "Too many elements"
        
        elif wrt_user_idx==None and wrt_item_idx==None:
            return "insufficient elements"

        else:
            u_row = self.user_features[user_row,:]
            i_col = self.item_features[:,item_col]
            ui_rating = float(self.data[user_row,item_col])
            prediction = float(np.dot(u_row,i_col))

            if wrt_user_idx != None:
                row_elem = float(i_col[wrt_user_idx])
                gradient = 2*(ui_rating-prediction)*row_elem
            
            else:
                col_elem = float(u_row[wrt_item_idx])
                gradient = 2*(ui_rating - prediction)*col_elem
            return gradient
    
    def user_feature_gradient(self,user_row,wrt_user_idx):
        """
        Averages the gradients of single user-item row with respect to a single user-feature parameter
        """

        summation = 0
        for col in range(0,self.item_count):
            summation += self.single_gradient(user_row = user_row,item_col=col,wrt_user_idx=wrt_user_idx)
        return summation/self.item_count
    
    def item_feature_gradient(self,item_col,wrt_item_idx):
        """
        Averages the gradient of a single user-item column with respect to a single feature-item parameter
        """

        summation = 0
        for row in range(0,self.user_count):
            summation += self.single_gradient(user_row=row,item_col=item_col,wrt_item_idx=wrt_item_idx)
        return summation/self.user_count
    
    def update_user_features(self,learning_rate):
        """
        Updates every user-feature parameter according to supplied learning rate
        """

        for i in range(0,self.user_count):
            for j in range(0,self.features):
                self.user_features[i,j] += learning_rate*self.user_feature_gradient(user_row=i,wrt_user_idx=j)

    def update_item_features(self,learning_rate):
        """
        updates every feature-item parameter according to supplied learning rate
        """

        for i in range(0,self.features):
            for j in range(0,self.item_count):
                self.item_features[i,j] += learning_rate*self.item_feature_gradient(item_col=j,wrt_item_idx=i)

    def train_model(self,learning_rate=0.1,iterations=1000):
        """
        Trains models, outputting MSE cost/loss every 50 iterations, using supplied learning rate and iterations
        """

        for i in range(iterations):
            self.update_user_features(learning_rate=learning_rate)
            self.update_item_features(learning_rate=learning_rate)
            if i % 50 == 0:
                print(self.MSE())

d = np.array([[5,3,1],[1,3,5],[3,5,1]])
d2 = matrix_factorization(d,2)
d2.train_model(learning_rate = 0.1)
