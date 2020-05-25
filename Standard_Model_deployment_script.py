#!/usr/bin/env python
# coding: utf-8
#Importing all dependencies


import os


import flask
from flask import Flask, render_template, request ,jsonify

# Library for deep larning models  (I have used FastAI. You can use Keras ,tf or Pytorch based on your model)
from fastai import *
from fastai.text import *

# Library to handle json format
import simplejson as json

# Library to  maintain logs
import logging
from logging.handlers import RotatingFileHandler
import collections

# Library to handle pickle format models (ML model here )
import pickle




#error log creation
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
logFile = 'log.txt'
my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=50*1024*1024,backupCount=25, encoding=None, delay=0)

#error log monitoring
my_handler.setFormatter(log_formatter)
my_handler.setLevel(logging.INFO)
app_log = logging.getLogger('root')
app_log.setLevel(logging.INFO)
app_log.addHandler(my_handler)
app_log.info('------Started----')

app= Flask(__name__)
try:
    # loading  ML model (Random Forest in this case)
    with open('ML_model.pkl', 'rb') as f:
       ML_model = pickle.load(f)  

	# loading Deep learning model using respective library functions
    CV_image_classification_Model=load_learner( os.getcwd(),'CV_image_classification_Model.pkl')
    NLP_model=load_learner( os.getcwd(),'NLP_Model.pkl')


except Exception as e:
    error_response_status={"code":301, "message":"unable to load model"} 
    response_status={"error":str(e)+" : Unable to load_model" ,"status":error_response_status}    
    app_log.info(e)
    
    
# setting up API endpoint  for ML model
@app.route('/predict_ml', methods=['POST'])
def predict_ml():
    try:
    	# validating the type of request
        if request.method=='POST': 
            try:
            	#taking input from the request
                req_data = request.get_json()             
                input_data=req_data["data"] 
                
            except Exception as e:
                error_response_status={"code":301, "message":"error"} 
                response_status={"error":str(e)+" : Unable to prcoess request ,check format " ,"status":error_response_status}    
                app_log.info(e)
                return jsonify(response_status),301   

            try:
                # predictions from  model (Here using Probability)
                prob = model.predict_proba(input_data)
                n =5
                # selecting top 5 predicted classes using the probability 
                top_n = np.argsort(prob)[:,:-n-1:-1]
                top_n_classes=model.classes_[top_n]
                confidence_list=[]
                top_n=top_n[0]
                
                # coverting Probabilites in Confidence for Users 
                
                for i in top_n:
                    conf=prob[0][i]*100
                    confidence_list.append(conf)                
                # formating  Predcited categories and confidences in JSON Format  
                result=dict(zip(top_n_classes[0],confidence_list))    
                lists=[]
                for u,v in result.items():
                    dict_temp={"Category":u,"Confidence":v}
                    lists.append(dict_temp)               
                
                success_response_status={"code":200, "message":"Success"} 
                results={"TOP 5 Predicted classes":lists,"status":success_response_status}
                # return results to user endpoint
                return jsonify (results),200
            
            except Exception as e:
                error_response_status={"code":301, "message":"error"} 
                response_status={"error":str(e)+" : Unable to predict" ,"status":error_response_status}    
                app_log.info(e)
                return jsonify(response_status),301   
                           
        else:
            error_response_status={"code":301, "message":"use Post method only"} 
            response_status={"error":" : Unable to process get request" ,"status":error_response_status}    
            app_log.info(e)
            return jsonify(response_status),301   
            
    except Exception as e:
        error_response_status={"code":301, "message":"error"} 
        response_status={"error":str(e)+" : Unable to connect" ,"status":error_response_status}    
        app_log.info(e)
        return jsonify(response_status),301   





    

@app.route('/predict_img', methods=['POST'])
def predict_img():
    try:
        if request.method=='POST': 
            try:
                 

                #read image file string data
                filestr = request.files['file']
                image_name=filestr.filename
				#saving image in server
                filestr.save('temp.png')


                
            except Exception as e:
                error_response_status={"code":301, "message":"error"} 
                response_status={"error":str(e)+" : Unable to prcoess request ,check format of image " ,"status":error_response_status}    
                app_log.info(e)
                return jsonify(response_status),301   

            try:
                
				# predictions from  model for image
                confidence_list=[]
                
                img=open_image('temp.png')                
                preds,tensor,probs=CV_image_classification_Model.predict(img)    
                 
                           
                classes=CV_image_classification_Model.data.classes
                # using small functions to get top 5 predictions with probability             
                top_5_predictions,top_5_confidence=top_5_pred_labels(probs,classes)  
                                         
                # coverting Probabilites in Confidence for Users 
                                  
                predict=str(top_5_predictions)
                for i in top_5_confidence:    
                    i=str(i)
                    i=i.replace('tensor(','')
                    i=i.replace(')','')
                    i=float(i)*100
                    confidence_list.append(i)
                    
                    
                # formating  Predcited categories and confidences in JSON Format                
                result=dict(zip(top_5_predictions,confidence_list))    
                lists=[]
                for u,v in result.items():
                    dict_temp={"Category":u,"Confidence":v}
                    lists.append(dict_temp)
                                    
                success_response_status={"code":200, "message":"Success"} 
                results={"TOP 5 Predicted classes":lists,"status":success_response_status}
                # return results to user endpoint
                return jsonify (results),200
            
            except Exception as e:
                error_response_status={"code":301, "message":"error"} 
                response_status={"error":str(e)+" : Unable to predict" ,"status":error_response_status}    
                app_log.info(e)
                return jsonify(response_status),301   
                           
        else:
            error_response_status={"code":301, "message":"use Post method only"} 
            response_status={"error":str(e)+" : Unable to process get request" ,"status":error_response_status}    
            app_log.info(e)
            return jsonify(response_status),301   
            
    except Exception as e:
        error_response_status={"code":301, "message":"error"} 
        response_status={"error":str(e)+" : Unable to connect" ,"status":error_response_status}    
        app_log.info(e)
        return jsonify(response_status),301   

@app.route('/predict_nlp', methods=['POST'])
def predict_nlp():
    try:
        if request.method=='POST': 
            try:
            	#taking input text from the request

                req_data = request.get_json()             
                input_data=req_data["data"] 
            except Exception as e:
                error_response_status={"code":301, "message":"error"} 
                response_status={"error":str(e)+" : Unable to prcoess request ,check format " ,"status":error_response_status}    
                app_log.info(e)
                return jsonify(response_status),301   

            try:
            	# predictions from  model for text
            	confidence_list=[]            
                
                preds,tensor,probs=NLP_model.predict(img)                  
                classes=NLP_model.data.classes            
                # using small functions to get top 5 predictions with probability    
                top_5_predictions,top_5_confidence=top_5_pred_labels(probs,classes)  
                                         
                # coverting Probabilites in Confidence for Users                   
                predict=str(top_5_predictions)
                for i in top_5_confidence:    
                    i=str(i)
                    i=i.replace('tensor(','')
                    i=i.replace(')','')
                    i=float(i)*100
                    confidence_list.append(i)
                    
                    
                # formating  Predcited categories and confidences in JSON Format                 
                result=dict(zip(top_5_predictions,confidence_list))    
                lists=[]
                for u,v in result.items():
                    dict_temp={"Category":u,"Confidence":v}
                    lists.append(dict_temp)
                                    
                success_response_status={"code":200, "message":"Success"} 
                results={"TOP 5 Predicted classes":lists,"status":success_response_status}
                # return results to user endpoint
                return jsonify (results),200
            
            except Exception as e:
                error_response_status={"code":301, "message":"error"} 
                response_status={"error":str(e)+" : Unable to predict" ,"status":error_response_status}    
                app_log.info(e)
                return jsonify(response_status),301   
                           
        else:
            error_response_status={"code":301, "message":"use Post method only"} 
            response_status={"error":str(e)+" : Unable to process get request" ,"status":error_response_status}    
            app_log.info(e)
            return jsonify(response_status),301   
            
    except Exception as e:
        error_response_status={"code":301, "message":"error"} 
        response_status={"error":str(e)+" : Unable to connect" ,"status":error_response_status}    
        app_log.info(e)
        return jsonify(response_status),301    





#function to get top 5 predictions based on probabilites
def top_5_preds(preds):
    
    preds_s = preds.argsort(descending=True)
    preds_s=preds_s[:5]    
    return preds_s

    
 
#function to get classes of top 5 predictions
def top_5_pred_labels(preds, classes):
    top_5 = top_5_preds(preds)
    labels = []
    confidence=[]
    for i in top_5:
        x=classes[i]
        p=preds[i]
        labels.append(x)
        confidence.append(p)        
     
    return labels ,confidence 






if __name__ == '__main__':
    app.run(host='0.0.0.0',threaded=True,debug=True)

