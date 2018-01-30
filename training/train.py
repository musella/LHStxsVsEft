#!/bin/env python

# load all libraries
import pandas as pd
## import matplotlib.pyplot as plt
import numpy as np

from optparse import OptionParser, make_option

from pprint import pprint

import json
import os

from gzip import open as gopen
try:
   import cPickle as pickle
except:
    import pickle

from sklearn import model_selection

## command_line options
parser = OptionParser(option_list=[
    make_option("--train-file",type="string",dest="train_file"),
    make_option("--max-entry",type="int",dest="max_entry",default=None),
    make_option("--out-dir",type="string",dest="out_dir"),
    make_option("--save-pickle",action="store_true",dest="save_pickle",default=False),

    make_option("--optimize",action='store_true',dest='optimize',default=False),
    make_option("--refit",action='store_true',dest='refit',default=False),
    make_option("--cluster",type='string',dest='cluster',default='default'),
    make_option("--cluster-nodes",type='string',dest='cluster_nodes',default=None),
    make_option("--kfolds",type="int",dest="kfolds",default=5),
    make_option("--grid",type="string",dest="grid",default=None),
    make_option("--niter",type="int",dest="niter",default=100),
    
    make_option("--features",type='string',dest='features',default=''),
    make_option("--balance-weights",action="store_true",dest="balance_weights",default=True),
    make_option("--no-balance-weights",action="store_false",dest="balance_weights"),
    make_option("--clf-params",type="string",dest="clf_params",default=None),
    make_option("--nthreads",type="int",dest="nthreads",default=4),
    
    make_option("--seed",type='int',dest='seed',default=123456),
])

## parse options
(options, args) = parser.parse_args()


# input features
default_features = [ u'pTZ', u'MVH', u'yZ', u'yH', u'etaZ', u'etaH', u'pTb1',
       u'pTb2', u'pTl1', u'pTl2', u'etab1', u'etab2', u'etal1', u'etal2',
       u'Mbb' ]

if options.features == '':
    features = default_features
    options.features = ','.join(features)
else:
    features = options.features.split(',')

# default clf parameters
clf_params = dict(n_estimators=600,subsample=0.8,max_depth=5,learning_rate=0.05,reg_lambda=1.,
                  nthread=options.nthreads)
if options.clf_params is not None:
    with open(options.grid,'r') as fin:
        options.clf_params = json.loads( fin.read() )
    clf_params.update( clf_params )
options.clf_params = clf_params
    
    
# hyper-parameters grid
default_grid = {
    "n_estimators" : [ 100, 300, 600 ], 
    "subsample" : [ 0.5, 0.8, 1. ],
    "max_depth" : [ 5, 10, 20 ],
    "reg_lambda" : [0., 0.5, 1. ],
    # "reg_alpha " : [0., 0.5, 1. ],
    "learning_rate" :  [0.05, 0.1, 0.025 ],
}
# reg_alpha
if options.grid is not None:
    with open(options.grid,'r') as fin:
        options.grid = json.loads( fin.read() )
else:
    options.grid = default_grid


print('--- options')
pprint(options.__dict__)
print('\n')

if not os.path.exists(options.out_dir):
    os.mkdir(options.out_dir)

print('--- loading training file ' + options.train_file + ' .... ' )
df = pd.read_hdf(options.train_file, columns=features+['label','wgt'])
print('\n')

# normalize weights
if options.balance_weights:
    ## labels = df['label'].unique()
    gb = df.groupby('label')
    ref = gb['wgt'].count().max()
    df['wgt'] = gb['wgt'].transform(lambda x: x * (ref / x.sum()) )
    

print('--- class weights' )
print(df.groupby('label')['wgt'].aggregate([np.sum,np.mean]) )
print('\n' )

# use ipython parallel for optimization
if options.optimize:
    print('--- setting up optimization cluster' )
    from sklearn.externals.joblib import Parallel, parallel_backend, register_parallel_backend
    
    import ipyparallel as ipp
    from ipyparallel import Client
    from ipyparallel.joblib import IPythonParallelBackend
    global joblib_rc,joblib_view,joblib_be
    joblib_rc = ipp.Client(profile=options.cluster)
    targets = None
    if options.cluster_nodes is not None:
        targets = [ int(x) for x in options.cluster_nodes.split(",") if x != "" ]
    joblib_view = joblib_rc.load_balanced_view(targets=targets)
    njobs = len(joblib_view)
    joblib_be = IPythonParallelBackend(view=joblib_view)
    
    register_parallel_backend('ipyparallel',lambda : joblib_be, make_default=True)
    print('will run %d jobs on %s (targets %s)' % ( njobs, options.cluster, targets ) )
    print('\n' )

    
# get features and target
X = df[features]#.values
y = df['label']#.values
w = df['wgt']#.values

    
# instantiate classifier
from xgboost import XGBClassifier
 
clf = XGBClassifier(**options.clf_params)

## print(clf.get_params())

if options.optimize:
    clf = model_selection.RandomizedSearchCV(clf,
                                             options.grid,
                                             n_iter=options.niter,
                                             cv=options.kfolds,
                                             n_jobs=njobs, verbose=100,
                                             refit=options.refit)

# train
clf.fit(X,y,sample_weight=w)

#save results
if options.optimize:
    with open('%s/best_params.json' % options.out_dir,'w+') as fout:
        fout.write( json.dumps( clf.best_params_ ) )
    pd.DataFrame( clf.cv_results_ ).to_hdf('%s/cv_results.hd5' % options.out_dir, key='cv_results')
    if options.refit:
        clf = clf.best_estimator_
else:
    with open('%s/best_params.json' % options.out_dir,'w+') as fout:
        fout.write( json.dumps( options.clf_params ) )

if not options.optimize or optimize.optimize and options.refit:
    if options.save_pickle:
        with gopen('%s/model.pkl.gz' % options.out_dir,'w+') as fout:
            pickle.dump(clf,fout)
            fout.close()
    try:
        model = clf.get_booster()
    except:
        model = clf.booster()
    model.save_model('%s/model.xgb' % options.out_dir)

## 
## 
## # train it
## clf.fit(X_train,y_train,w_train)
