from asyncio.windows_events import NULL
import flask
from flask import Flask, request, flash, redirect, jsonify
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import operator
import numpy as np
import regex as re
import joblib

app = Flask(__name__)


# app = Flask('author_script')


app.debug = True

###################################################

def feature_engineering_perm(data, fe_imp):
    '''
    This method is used to engineer new features(+, -, *, /, cos, cosh, sin , sinh, exp of existing features) from the given
    list of feature transformations and the dataset.
    
    ----------------------
    Parameter
    data    : The input dataset.
    
    fe_imp  : A list containing the top important feature combinations of arithmetic and trignometric
              operations.
              
    ----------------------                       
    Returns
    data_fe : A pandas DataFrame containing dataset with only the given transformed features.
    '''
    #print('Inside method - feature_engineering_perm')
    data_fe = pd.DataFrame()
    #defining a dict of operation name and the operations
    op_dict = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv,
              'cos': np.cos, 'sin': np.sin, 'cosh': np.cosh, 'sinh': np.sinh, 'exp': np.exp}
    
    for i in fe_imp:
        oper = ''.join(re.findall('[^0-9_]', i)) #extracting the arith or trig operation from the name of col.Eg:12+32 or cos_21
        if(not oper):
            data_fe[i] = data[i].copy() #if there is no operation involved then the same feature is taken as it is.
        else:
            op = op_dict[oper]
            if('_' in i): #checking if the operation is trig
                cols = i.split(oper+'_') #splitting based on the '_' Eg: cos_12 -> cos, 12
                data_fe[i] = op(data[cols[1]]) #appying the trig operation
            else: 
                cols = i.split(oper) #splitting based on the operation Eg: 12+13 -> splitting on '+' -> 12,13
                data_fe[i] = op(data[cols[0]], data[cols[1]]) #applying the arth operation
                
        if(data_fe[i].isin([np.inf, -np.inf, np.nan]).any()):
            data_fe[i].replace([np.inf, -np.inf], np.nan, inplace=True) #replaces the inf, -inf to nan(coz mean of r.v with inf is nan)
            data_fe[i].replace(np.nan, data_fe[i].mean(), inplace=True) # replaces nan with mean val
        
    #print('Exiting method - feature_engineering_perm')
    return data_fe
    
top_features = ['33+65', '33-217', '217-33', '33-133', '133-33', '91-33', '33-91',
       '33+199', 'sin_33', '33-295', '295-33', '73-33', '33-73', '33-258',
       '258-33', '33-117', '117-33', '33+183', '33+226', '33', '33-134',
       '134-33', '65+199', '33-268', '268-33', '80-33', '33-80', '194-33',
       '33-194', '33-129', '129-33', '33-82', '82-33', '65-217', '217-65',
       '43-33', '33-43', '16-33', '33-16', '189-33', '33-189', '33-252',
       '252-33', '33+201', '33+114', '33+101', '150-33', '33-150', '220-33',
       '33-220', '239-33', '33-239', '33-108', '108-33', '33-90', '90-33',
       '117-65', '65-117', '24+33', '33-39', '39-33', '276-33', '33-276',
       '33+164', '33+285', '165-33', '33-165', '33+89', '33-182', '182-33',
       '33-4', '4-33', '65-134', '134-65', '73-65', '65-73', '30+33', '211-33',
       '33-211', '45-33', '33-45', '33+105', '17+33', '63-33', '33-63',
       '33-237', '237-33', '33-98', '98-33', '33+215', '33+221', '65-91',
       '91-65', '33-180', '180-33', '189-65', '65-189', '33-298', '298-33',
       '24+65']
###################################################


@app.route('/')
def home():
    return flask.render_template('index.html')


@app.route('/index')
def index():
    return flask.render_template('index_2.html')

@app.route("/predict", methods=['POST'])
def predict():
    print('-'*100)
    print('Inside Predict method')
    print('-'*100)
    try:
        file = request.files['file']
        X = file.read().decode("utf-8")
        if(not X):
            print('File content: ',X)
            print('File has no content')
            return jsonify({'error': 'Please check the format of the txt file'})
        
        #X = np.array([float(i) for i in X.split(',')]).reshape(1,-1)
        X = np.array(list(map(float, X.split(',')))).reshape(1,-1)
        if(X.shape[1] != 300):
            print('Wrong no. of columns: ',X.shape)
            return jsonify({'error': 'Please check the format of the txt file'})

        #Feature engineering
        X = pd.DataFrame(X, columns=list(map(str, range(0,300))))
        X_fe = feature_engineering_perm(X, top_features)
        
        #predicting
        print('Predicting....')
        clf = joblib.load('BestModel.sav')
        pred = clf.predict_proba(X_fe)[0,1]
        print('predicted probability:', pred)
        prediction = 1 if pred > 0.5 else 0
        print('prediction:', prediction)
        return jsonify({'prediction': prediction, 'prediction_prob': pred})
    except Exception as e:
        print('An exception occurred')
        print(e)
        return jsonify({'error': 'Please check the format of the txt file'})


app.run("localhost","8080")