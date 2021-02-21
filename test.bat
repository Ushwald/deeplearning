ECHO OFF
 
FOR /L %%y IN (.03, .01, .06) Do FOR %%x IN (0, .00001. .0001, .001, .01, .1, .5, 1) DO py -3.8 ./main.py --epochs 50 --crossvalidation=True --learningrate %%y --momentum %%x


PAUSE