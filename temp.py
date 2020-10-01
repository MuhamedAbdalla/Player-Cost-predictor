#importing libraries
import pandas as pd
import numpy as np

#get the data
dataset = pd.read_csv("E:\\player_data.csv")#string is location of excel file in pc

#get the independed and depended variables
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,12].values

#make the depended variable numeric
indx = 0
#convert value of player to numeric data
for i in y:
            num = i[1:-1]
    if(i[len(i) - 1] == 'M'):
        var = float(num) * 1000000
    elif(i[len(i) - 1] == 'K'):
        var = float(num) * 1000
    else:
        var = 0
    y[indx] = var
    indx += 1

#Encode the 4th column to 0 for left and 1 for right
from sklearn.preprocessing import LabelEncoder,OneHotEncoder 
lb = LabelEncoder()
X[:, 3] = lb.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])#split to two colume
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]#take second colume
#get data and make randome state for prediction
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)
#get prediction by train the machine with liner regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pre = lr.predict(X_test)
accur = lr.score(X_test, y_test)
#get prediction by train the machine with DecisiontTreeRegressor
from sklearn.tree import DecisionTreeRegressor
DTC= DecisionTreeRegressor()
DTC.fit(X_train, y_train)#train
y_pre = DTC.predict(X_test)#predict
accur = DTC.score(X_test, y_test)#percentage of correct result

import sys
from PyQt5 import QtWidgets

class GUI(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setGeometry(300, 300, 1000, 1000)
        self.l1 = QtWidgets.QLabel(self)
        self.l1.setText('Age')
        self.l1.move(20, 20)
        self.le1 = QtWidgets.QLineEdit(self)
        self.le1.move(125, 15)
        self.le1.resize(200, 25)
        
        self.l2 = QtWidgets.QLabel(self)
        self.l2.setText('Overall')
        self.l2.move(20, 80)
        self.le2 = QtWidgets.QLineEdit(self)
        self.le2.move(125, 75)
        self.le2.resize(200, 25)
        
        self.l3 = QtWidgets.QLabel(self)
        self.l3.setText('Potential')
        self.l3.move(20, 140)
        self.le3 = QtWidgets.QLineEdit(self)
        self.le3.move(125, 135)
        self.le3.resize(200, 25)
        
        self.l4 = QtWidgets.QLabel(self)
        self.l4.setText('Preferred Foot 0 For Left And 1 For Right')
        self.l4.move(20, 200)
        self.le4 = QtWidgets.QLineEdit(self)
        self.le4.move(250, 195)
        self.le4.resize(200, 25)
        
        self.l5 = QtWidgets.QLabel(self)
        self.l5.setText('Skill Moves')
        self.l5.move(20, 260)
        self.le5 = QtWidgets.QLineEdit(self)
        self.le5.move(125, 255)
        self.le5.resize(200, 25)
        
        self.l6 = QtWidgets.QLabel(self)
        self.l6.setText('Crossing')
        self.l6.move(20, 320)
        self.le6 = QtWidgets.QLineEdit(self)
        self.le6.move(125, 315)
        self.le6.resize(200, 25)
        
        self.l7 = QtWidgets.QLabel(self)
        self.l7.setText('Finishing')
        self.l7.move(20, 380)
        self.le7 = QtWidgets.QLineEdit(self)
        self.le7.move(125, 375)
        self.le7.resize(200, 25)
        
        self.l8 = QtWidgets.QLabel(self)
        self.l8.setText('HeadingAccuracy')
        self.l8.move(20, 440)
        self.le8 = QtWidgets.QLineEdit(self)
        self.le8.move(125, 435)
        self.le8.resize(200, 25)
        
        self.l9 = QtWidgets.QLabel(self)
        self.l9.setText('SprintSpeed')
        self.l9.move(20, 500)
        self.le9 = QtWidgets.QLineEdit(self)
        self.le9.move(125, 495)
        self.le9.resize(200, 25)
        
        self.s1 = QtWidgets.QLabel(self)
        self.s1.setText('ShotPower')
        self.s1.move(20, 560)
        self.se1 = QtWidgets.QLineEdit(self)
        self.se1.move(125, 555)
        self.se1.resize(200, 25)
        
        self.s2 = QtWidgets.QLabel(self)
        self.s2.setText('Penalties')
        self.s2.move(20, 620)
        self.se2 = QtWidgets.QLineEdit(self)
        self.se2.move(125, 615)
        self.se2.resize(200, 25)
        
        self.s3 = QtWidgets.QLabel(self)
        self.s3.setText('GKHandling')
        self.s3.move(20, 680)
        self.se3 = QtWidgets.QLineEdit(self)
        self.se3.move(125, 675)
        self.se3.resize(200, 25)
        
        self.b1 = QtWidgets.QPushButton('Get Value', self)
        self.b1.move(400, 620)
        self.b1.resize(200, 25)
        
        self.l10 = QtWidgets.QLabel(self)
        self.l10.setText('Market Value')
        self.l10.move(400, 680)
        self.le10 = QtWidgets.QLineEdit(self)
        self.le10.move(500, 680)
        self.le10.resize(200, 25)
       
        self.setWindowTitle('Player Market')

        self.b1.clicked.connect(self.press)

        self.show()

    def press(self):
        X1 = self.le1.text()
        X2 = self.le2.text()
        X3 = self.le3.text()
        X4 = self.le4.text()
        X5 = self.le5.text()
        X6 = self.le6.text()
        X7 = self.le7.text()
        X8 = self.le8.text()
        X9 = self.le9.text()
        X10 = self.se1.text()
        X11 = self.se2.text()
        X12 = self.se3.text()
        y_pred = DTC.predict([[X4, X1, X2, X3, X5, X6, X7, X8, X9, X10, X11, X12]])
        if(len(str(y_pred)) >= 6):
            self.le10.setText('€' + str((float(y_pred) / 1000000.0)) + "M")
        elif(len(str(y_pred)) >= 3):
            self.le10.setText('€' + str((float(y_pred) / 1000.0)) + "K")
        else:
            self.le10.setText('€ 0')
        
app = QtWidgets.QApplication(sys.argv)
GUInxt = GUI()
sys.exit(app.exec_())

#####################################################################
#Second app
#####################################################################

import statistics
import sys
from PyQt5 import QtWidgets

ls = []

class GUI(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setGeometry(300, 300, 1000, 1000)
        self.l1 = QtWidgets.QLabel(self)
        self.l1.setText('Enter Number')
        self.l1.move(20, 20)
        self.le1 = QtWidgets.QLineEdit(self)
        self.le1.move(125, 15)
        self.le1.resize(200, 25)
        
        self.b1 = QtWidgets.QPushButton('Get INFO.', self)
        self.b1.move(125, 80)
        self.b1.resize(200, 25)
        
        self.l10 = QtWidgets.QLabel(self)
        self.l10.setText('Mean')
        self.l10.move(125, 125)
        self.le10 = QtWidgets.QLineEdit(self)
        self.le10.move(125, 150)
        self.le10.resize(200, 25)
        
        self.l11 = QtWidgets.QLabel(self)
        self.l11.setText('Median')
        self.l11.move(125, 185)
        self.le11 = QtWidgets.QLineEdit(self)
        self.le11.move(125, 210)
        self.le11.resize(200, 25)
        
        self.l12 = QtWidgets.QLabel(self)
        self.l12.setText('Mode')
        self.l12.move(125, 245)
        self.le12 = QtWidgets.QLineEdit(self)
        self.le12.move(125, 270)
        self.le12.resize(200, 25)
        
        self.l13 = QtWidgets.QLabel(self)
        self.l13.setText('Standard diviation')
        self.l13.move(125, 305)
        self.le13 = QtWidgets.QLineEdit(self)
        self.le13.move(125, 330)
        self.le13.resize(200, 25)
        
        self.l14 = QtWidgets.QLabel(self)
        self.l14.setText('Variance')
        self.l14.move(125, 365)
        self.le14 = QtWidgets.QLineEdit(self)
        self.le14.move(125, 390)
        self.le14.resize(200, 25)
        
        self.setWindowTitle('INFO')

        self.b1.clicked.connect(self.press)

        self.show()

    def press(self):
        X1 = self.le1.text()
        val = int(X1)
        if(len(X1) == 0):
            return
        ls.append(val)
        self.le10.setText(str(statistics.mean(ls)))
        self.le11.setText(str(statistics.median(ls)))
        self.le12.setText(str(statistics.mode(ls)))
        self.le13.setText(str(statistics.stdev(ls)))
        self.le14.setText(str(statistics.variance(ls)))
        self.le1.setText('')
        
app = QtWidgets.QApplication(sys.argv)
GUInxt = GUI()
sys.exit(app.exec_())

#####################################################################
#third app
#####################################################################

import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8,9]
y = [5,2,4,2,1,4,5,2,4]

plt.scatter(x, y, label='skitscat', color = 'k')

plt.xlabel('x')
plt.xlabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()

#####################################################################
#forth app
#####################################################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(1)

# 1000 random integers between 0 and 50
x = np.random.randint(0, 50, 1000)

# Positive Correlation with some noise
y = x + np.random.normal(0, 10, 1000)

np.corrcoef(x, y)

matplotlib.style.use('ggplot')

plt.scatter(x, y)
plt.show()

#####################################################################
#five app
#####################################################################

import numpy as np 
import matplotlib.pyplot as plt 
  
def estimate_coef(x, y): 
    n = np.size(x) 
  
    m_x, m_y = np.mean(x), np.mean(y) 
  
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 
  
def plot_regression_line(x, y, b): 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    y_pred = b[0] + b[1]*x 
  
    plt.plot(x, y_pred, color = "g") 
  
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    plt.show() 
  
def main(): 
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 
  
    b = estimate_coef(x, y) 
    print("Estimated coefficients:\nb_0 = {}  \ \nb_1 = {}".format(b[0], b[1])) 
  
    plot_regression_line(x, y, b) 
  
if __name__ == "__main__": 
    main() 