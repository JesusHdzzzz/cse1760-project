(cse176) jesushdzzzz@jesushdzzzz-SER8:~/Desktop/cse1760-project/Part2$ python ./src/xgb_mnist_pixels.py
Train features: (60000, 784)
Test features: (10000, 784)
Train size: 55000
Val   size: 5000

=== XGBoost Cross-Validation (Pixel Features) ===
n=50, depth=4 → CV Accuracy = 0.9293
n=50, depth=6 → CV Accuracy = 0.9541
n=50, depth=8 → CV Accuracy = 0.9609
n=100, depth=4 → CV Accuracy = 0.9522
n=100, depth=6 → CV Accuracy = 0.9661
n=100, depth=8 → CV Accuracy = 0.9688
n=200, depth=4 → CV Accuracy = 0.9659
n=200, depth=6 → CV Accuracy = 0.9728
n=200, depth=8 → CV Accuracy = 0.9729

Best: n_estimators=200, max_depth=8 (CV=0.9729)

Final Test Accuracy (pixel features): 0.9785

Classification Report:
              precision    recall  f1-score   support

           0     0.9894    0.9912    0.9903      1135
           1     0.9720    0.9758    0.9739      1032
           2     0.9744    0.9812    0.9778      1010
           3     0.9856    0.9735    0.9795       982
           4     0.9864    0.9787    0.9826       892
           5     0.9812    0.9812    0.9812       958
           6     0.9784    0.9698    0.9741      1028
           7     0.9713    0.9743    0.9728       974
           8     0.9683    0.9683    0.9683      1009
           9     0.9778    0.9898    0.9838       980

    accuracy                         0.9785     10000
   macro avg     0.9785    0.9784    0.9784     10000
weighted avg     0.9785    0.9785    0.9785     10000

Confusion Matrix:
[[1125    1    3    0    1    2    1    2    0    0]
 [   0 1007    8    3    0    0    6    4    1    3]
 [   0    3  991    0    2    0    7    4    2    1]
 [   0    3    1  956    0    4    0    3   15    0]
 [   0    1    5    0  873    6    2    2    1    2]
 [   3    0    1    2    3  940    0    5    0    4]
 [   3   15    2    1    0    0  997    2    7    1]
 [   1    4    1    2    3    3    1  949    5    5]
 [   5    2    5    6    1    0    4    3  977    6]
 [   0    0    0    0    2    3    1    3    1  970]]

===== TIMING SUMMARY (Pixels) =====
Data loading time:       0.34 sec
Cross-validation time:   1449.65 sec
Final training time:     155.90 sec
Evaluation time:         0.0393 sec
TOTAL runtime:           1606.24 sec
(cse176) jesushdzzzz@jesushdzzzz-SER8:~/Desktop/cse1760-project/Part2$ python ./src/xgb_mnist_lenet.py
Train features: (60000, 800)
Test features: (10000, 800)
Train size: 55000
Val   size: 5000

=== XGBoost Cross-Validation (LeNet5 Features) ===
n=50, depth=4 → CV Acc = 0.9754
n=50, depth=6 → CV Acc = 0.9805
n=50, depth=8 → CV Acc = 0.9819
n=100, depth=4 → CV Acc = 0.9842
n=100, depth=6 → CV Acc = 0.9859
n=100, depth=8 → CV Acc = 0.9853
n=200, depth=4 → CV Acc = 0.9885
n=200, depth=6 → CV Acc = 0.9880
n=200, depth=8 → CV Acc = 0.9868

Best: n_estimators=200, max_depth=4 (CV=0.9885)

Final Test Accuracy (LeNet5): 0.9901

Classification Report:
              precision    recall  f1-score   support

           0     0.9965    0.9947    0.9956      1135
           1     0.9875    0.9913    0.9894      1032
           2     0.9891    0.9911    0.9901      1010
           3     0.9929    0.9908    0.9918       982
           4     0.9943    0.9832    0.9887       892
           5     0.9937    0.9864    0.9900       958
           6     0.9826    0.9903    0.9864      1028
           7     0.9897    0.9887    0.9892       974
           8     0.9871    0.9871    0.9871      1009
           9     0.9879    0.9959    0.9919       980

    accuracy                         0.9901     10000
   macro avg     0.9901    0.9900    0.9900     10000
weighted avg     0.9901    0.9901    0.9901     10000

Confusion Matrix:
[[1129    1    3    0    0    1    1    0    0    0]
 [   1 1023    0    0    0    0    5    1    1    1]
 [   0    3 1001    0    3    0    2    1    0    0]
 [   0    2    0  973    0    3    0    0    4    0]
 [   0    0    5    0  877    1    2    2    3    2]
 [   2    0    1    1    1  945    0    3    0    5]
 [   1    5    0    0    0    0 1018    1    2    1]
 [   0    1    1    1    1    0    2  963    3    2]
 [   0    1    1    5    0    0    5    0  996    1]
 [   0    0    0    0    0    1    1    2    0  976]]

===== TIMING SUMMARY (LeNet5) =====
Data loading time:       1.37 sec
Cross-validation time:   2592.47 sec
Final training time:     190.27 sec
Evaluation time:         0.0365 sec
TOTAL runtime:           2784.42 sec