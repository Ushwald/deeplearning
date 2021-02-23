ECHO OFF

FOR %%y IN (.001, .01, .05, .1) DO FOR %%x IN (0, .1, .5, 1) DO py -3.8 ./main.py --epochs 50 --sgdm=True --elu=True --crossvalidation=True --learningrate %%y --momentum %%x

FOR %%y IN (.001, .01, .05, .1) DO FOR %%x IN (0, .1, .5, 1) DO py -3.8 ./main.py --epochs 50 --crossvalidation=True --learningrate %%y --momentum %%x

FOR %%y IN (.001, .01, .05, .1) DO FOR %%x IN (0, .1, .5, 1) DO py -3.8 ./main.py --epochs 50 --sgdm=True --crossvalidation=True --learningrate %%y --momentum %%x

FOR %%y IN (.001, .01, .05, .1, .5) DO FOR %%x IN (0, .1, .5, 1) DO py -3.8 ./main.py --epochs 50 --crossvalidation=True --elu=True --learningrate %%y --momentum %%x
PAUSE