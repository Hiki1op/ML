from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import pandas as pd 

# metrics
def accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

# model··
class Decision_Tree_Node:  # node of decision Tree
    def __init__(self, D, A):
        self.D = D
        self.A = A
        self.attr = None
        self.vals = []
        self.children = []
        self.typeid = -1
    
    def settype(self, type):
        self.typeid = type
        
    def show(self):
        return f"DecisionTreeNode:data = {self.D} attributes = {self.A}"
        
class DecisionTreeClassifier:
    def __init__(self) -> None:
        self.tree = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        # X: [n_samples_train, n_features], 
        # y: [n_samples_train, ],
        # TODO: implement decision tree algorithm to train the model
        self.X = X.reset_index(drop = True)
        self.y = y
        self.isContinuous = {}
        index = list(self.X.index)
        attrs = list(self.X.columns)
        for a in attrs:
            Xa = X.loc[:, a]
            diffvals = len(np.unique(Xa))
            if diffvals > 10:
                self.isContinuous[a] = True
            else:
                self.isContinuous[a] = False
        self.tree = self.tree_generate(index, attrs)
        self.X = None
        self.y = None
        
    def predict(self, X: pd.DataFrame):
        # X: [n_samples_test, n_features],
        # return: y: [n_samples_test, ]
        y = np.zeros(X.shape[0])
        # TODO:
        id = 0
        for _, row in X.iterrows():
            node = self.tree
            while node.typeid == -1: # not a leaf
                if self.isContinuous[node.attr]:
                    val = row[node.attr]
                    pivot = node.vals[0]
                    if val >= pivot:
                        node = node.children[0]
                    else:
                        node = node.children[1]
                else:
                    val = row[node.attr]
                    val_id = node.vals.index(val)
                    node = node.children[val_id]
                ''' try:
                        val_id = node.vals.index(val)
                        node = node.children[val_id]
                    except Exception as e:
                        print("当前id" + str(id) + "\n")
                        print("有训练集中未出现的离散值！\n") 
                        print("出错属性:" + node.attr + "\n")  
                        print("结点是否连续:" + str(self.isContinuous[node.attr]) + "\n")  
                        print("node结点子孩子:" + ' '.join(map(str, node.vals)))   '''
            y[id] = node.typeid
            id += 1
        return y
    
    def tree_generate(self, D: list, A: list) -> Decision_Tree_Node:
        node = Decision_Tree_Node(D, A)
        X = self.X.loc[D]
        y = self.y[D]
        if len(np.unique(y)) == 1: 
            node.typeid = y[0]
            return node
        if len(A) == 0 or len(X.loc[:, A].drop_duplicates()) == 1:
            _, counts = np.unique(y, return_counts=True)
            node.typeid = y[np.argmax(counts)]
            return node
        t, node.attr = self.best_spilt(D, A)
        Xa = X.loc[:, node.attr]
        if self.isContinuous[node.attr] == True:
            D_plus = list(Xa[Xa >= t].index)
            D_minus = list(Xa[Xa < t].index)
            node.vals = [t]
            node_plus = self.tree_generate(D_plus, A)
            node_minus = self.tree_generate(D_minus, A)
            node.children = [node_plus, node_minus]
        else:
            A.remove(node.attr)
            vals, _ = np.unique(Xa, return_counts=True)
            for v in vals:
                D_new = list(Xa[Xa == v].index)
                if len(D_new) == 0:
                    node_new = Decision_Tree_Node(D_new, A)
                    _, counts = np.unique(y, return_counts=True)
                    node_new.settype(y[np.argmax(counts)])
                else:
                    node_new = self.tree_generate(D_new, A)
                node.vals.append(v)
                node.children.append(node_new)
        return node
    
    def Ent(self, D):
        _, counts = np.unique(self.y[D], return_counts=True)
        probabilities = counts / len(self.y)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def infogain(self, D: list, a: list):
        ent_pos = self.Ent(D)
        ent_neg = 0
        pivot = 0
        X = self.X.loc[D]
        Xa = X.loc[:, a]
        if self.isContinuous[a] == True:
            vals = np.sort(Xa)
            maxt = 0
            minent = np.inf
            for i in range(len(vals) - 1):
                t = (vals[i] + vals[i+1]) / 2
                D_plus = list(Xa[Xa >= t].index)
                D_minus = list(Xa[Xa < t].index)
                totminus = (self.Ent(D_plus) * len(D_plus) + self.Ent(D_minus) * len(D_minus) ) / len(D)
                if totminus < minent:
                    minent = totminus
                    maxt = t
            pivot = maxt
            ent_neg = minent    
        else:
            vals, cnts = np.unique(Xa, return_counts=True)
            pros = cnts / len(D)
            for i, val in enumerate(vals):
                D_new = list(Xa[Xa == val].index)
                ent_neg += self.Ent(D_new) * pros[i]
        return pivot, ent_pos - ent_neg
    
    def best_spilt(self, D, A):
        max_info_gain = -np.inf
        max_attr = None
        pivot = 0
        for a in A:
            t, gain = self.infogain(D, a)
            if gain > max_info_gain:
                max_info_gain = gain
                max_attr = a
                pivot = t
        return pivot, max_attr
            
        

def load_data(datapath:str='./DecisionTree/data/ObesityDataSet_raw_and_data_sinthetic.csv'):
    df = pd.read_csv(datapath)
    continue_features = ['Age', 'Height', 'Weight', ]
    discrete_features = ['Gender', 'CALC', 'FAVC', 'FCVC', 'NCP', 'SCC', 'SMOKE', 'CH2O', 'family_history_with_overweight', 'FAF', 'TUE', 'CAEC', 'MTRANS']
    
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    # encode discrete str to number, eg. male&female to 0&1
    labelencoder = LabelEncoder()
    for col in discrete_features:
        X[col] = labelencoder.fit(X[col]).transform(X[col])
    y = labelencoder.fit(y).fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__=="__main__":
    X_train, X_test, y_train, y_test = load_data('./DecisionTree/data/ObesityDataSet_raw_and_data_sinthetic.csv')
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(accuracy(y_test, y_pred))