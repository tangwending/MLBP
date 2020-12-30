# -*- coding: utf-8 -*-
# @Author  : twd
# @FileName: test.py
# @Software: PyCharm


import os
from pathlib import Path
import keras
from keras.optimizers import Adam
from keras.models import model_from_json
from train import catch
from evaluation import scores, evaluate
import pickle
from keras.models import load_model


def predict(X_test, y_test, thred, para, weights, jsonFiles, h5_model, dir):

    # with open('test_true_label.pkl', 'wb') as f:
    #     pickle.dump(y_test, f)

    adam = Adam(lr=para['learning_rate']) # adam optimizer
    for ii in range(0, len(weights)):
        # 1.loading weight and structure (model)

        # json_file = open('BiGRU_base/' + jsonFiles[i], 'r')
        # model_json = json_file.read()
        # json_file.close()
        # load_my_model = model_from_json(model_json)
        # load_my_model.load_weights('BiGRU_base/' + weights[i])
        # load_my_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        h5_model_path = os.path.join(dir, h5_model[ii])
        load_my_model = load_model(h5_model_path)
        print("Prediction is in progress")

        # 2.predict
        score = load_my_model.predict(X_test)

        "========================================"
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] < thred:
                    score[i][j] = 0
                else:
                    score[i][j] = 1
        a, b, c, d, e = evaluate(score, y_test)
        print(a, b, c, d, e)
        "========================================"

        # 3.evaluation
        if ii == 0:
            score_label = score
        else:
            score_label += score

    score_label = score_label / len(h5_model)

    # data saving
    with open(os.path.join(dir, 'MLBP_prediction_prob.pkl'), 'wb') as f:
        pickle.dump(score_label, f)

    # getting prediction label
    for i in range(len(score_label)):
        for j in range(len(score_label[i])):
            if score_label[i][j] < thred: score_label[i][j] = 0
            else: score_label[i][j] = 1

    # data saving
    with open(os.path.join(dir, 'MLBP_prediction_label.pkl'), 'wb') as f:
        pickle.dump(score_label, f)

    # evaluation
    aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(score_label, y_test)

    print("Prediction is done")
    print('aiming:', aiming)
    print('coverage:', coverage)
    print('accuracy:', accuracy)
    print('absolute_true:', absolute_true)
    print('absolute_false:', absolute_false)
    print('\n')

    out = dir
    Path(out).mkdir(exist_ok=True, parents=True)
    out_path2 = os.path.join(out, 'result_test.txt')
    with open(out_path2, 'w') as fout:
        fout.write('aiming:{}\n'.format(aiming))
        fout.write('coverage:{}\n'.format(coverage))
        fout.write('accuracy:{}\n'.format(accuracy))
        fout.write('absolute_true:{}\n'.format(absolute_true))
        fout.write('absolute_false:{}\n'.format(absolute_false))
        fout.write('\n')



def test_my(test, para, model_num, dir):
    # step1: preprocessing
    test[1] = keras.utils.to_categorical(test[1])
    test[0], temp = catch(test[0], test[1])
    temp[temp > 1] = 1
    test[1] = temp

    # weight and json
    weights = []
    jsonFiles = []
    h5_model = []
    for i in range(1, model_num+1):
        weights.append('model{}.hdf5'.format(str(i)))
        jsonFiles.append('model{}.json'.format(str(i)))
        h5_model.append('model{}.h5'.format(str(i)))

    # step2:predict
    predict(test[0], test[1], test[2], para, weights, jsonFiles, h5_model, dir)