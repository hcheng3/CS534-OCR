import json

from flask import Flask, request, jsonify
from flask import render_template
import numpy as np

import multiclass


GRID = 8
image_matrix = np.zeros([GRID, GRID])
app = Flask(__name__)
c=None


@app.route("/", methods=['GET', 'POST'])
def main():
    global c
    c=all_classifiers().classifier
    return render_template('index.html')


@app.route("/draw.html", methods=['GET', 'POST'])
def single():
    if request.method == 'GET':
        return render_template('draw.html')
    if request.method == 'POST':
        value = get_result()
        print value
        return json.dumps(value)


def get_result():
    global image_matrix
    result = request.form['result']
    result = json.loads(result)
    import numpy as np
    r=np.array(result, dtype="float64")
    #print "++++++++++",r[r==1].size
    width = len(result)
    height = len(result)
    m = height/GRID
    n = width/GRID
    for i in range(len(result) - 1):
            for j in range(len(result) - 1):
                if result[i][j] == 1:
                    image_matrix[i/n, j/m] += 1
    #print "r shape is: ", r.shape
    test = get_feature_(r)
    #print "test sample is: ", test
    image_matrix = np.zeros([GRID, GRID])
    p = multiclass.predict_new(test, c)
    return p


def get_feature_(r):
    import feature_extraction
    f=feature_extraction.feature_extraction(r)
    return f.combine_all()
    #return image_matrix.ravel()*1.0/64

#path="features_all_without_skeleton"
path="features_all_compress5"
class all_classifiers():
    def __init__(self):
        global path
        u=np.load(path+'/labels.npz')['ulabels']
        print "loading classifiers...labels are: ", u
        self.classifier={}
        self.classifier['kernel']=np.load(path+'/labels.npz')['k']
        self.classifier['labels']=u
        uu=list(u)
        for i in u:
            uu.remove(i)
            for j in uu:
                id=str(i)+","+str(j)
                bias=np.load(path+'/'+id+'.npz')['bias']
                support_multipliers = np.load(path+'/'+id+'.npz')['support_multipliers']
                support_vectors = np.load(path+'/'+id+'.npz')['support_vectors']
                support_vector_labels = np.load(path+'/'+id+'.npz')['support_vector_labels']
                c=classifier(bias,support_vectors,support_multipliers,support_vector_labels)
                self.classifier[id]=c
        print len(self.classifier)-2, "classifiers are loaded..."

class classifier():
    def __init__(self, bias, support_vectors, support_vector_multipliers, support_vector_labels):
        self.bias=bias
        self.support_vectors=support_vectors
        self.support_vectors_multipliers=support_vector_multipliers
        self.support_vector_labels=support_vector_labels



if __name__ == '__main__':
    app.debug = True
    app.run()
