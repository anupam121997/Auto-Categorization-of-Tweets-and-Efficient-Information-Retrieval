# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:01:38 2021

@author: kumar
"""

import speechtotext
from flask import Flask, render_template,request,redirect
import dummy
app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('test.html')

@app.route('/',methods=['POST','GET'])
def getValues():
        query=request.form.get('query')
        model=request.form.get('models')
        tweet_class,res=dummy.return_params(query, model)
        return render_template("test.html",tweet_class = tweet_class,a=res[0],b=res[1],c=res[2],d=res[3],e=res[4],f=res[5],
        g1=res[6],h=res[7],i=res[8],j=res[9])



@app.route('/ak',methods=['POST','GET'])
def my_link():
    mytext=speechtotext.gettext()
    model=request.form.get('models')
    tweet_class,res=dummy.return_params(mytext, model)
    return render_template("test.html",micSearch=mytext,tweet_class = tweet_class,a=res[0],b=res[1],c=res[2],d=res[3],e=res[4],f=res[5],
    g1=res[6],h=res[7],i=res[8],j=res[9])

	
if __name__ == "__main__":
    app.run(debug=False)