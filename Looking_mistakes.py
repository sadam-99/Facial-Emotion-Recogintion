def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    
    print "training fisher face classifier"
    print "size of training set is:", len(training_labels), "images"
    fishface.train(training_data, np.asarray(training_labels))

    print "predicting classification set"
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            cv2.imwrite("difficult\\%s_%s_%s.jpg" %(emotions[prediction_labels[cnt]], emotions[pred], cnt), image) #<-- this one is new
            incorrect += 1
            cnt += 1
    return ((100*correct)/(correct + incorrect))
