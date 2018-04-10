from flask import Flask, render_template
from datetime import datetime

import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as iread
import tensorflow as tf
from PIL import Image
import numpy as np
from random import shuffle
from random import randint
import urllib.request
from io import BytesIO

app = Flask(__name__)
# @app.route('/')
# def homepage():
#     a = trainnew.class1('asdad')
#     # a = trainnew.b()
#     return a
#     # the_time = datetime.now().strftime("%A, %d %b %Y %l:%M %p")

#     # return """
#     # <h1>Hello heroku</h1>
#     # <p>It is currently {time}.</p>

#     # <img src="http://loremflickr.com/600/400" />
#     # """.format(time=the_time)
# @app.route('/<test>')
# def homeepage(test):
#     return render_template('hello.html', hasil=trainnew.class1("asdasd"))

@app.route('/class/<cl>')
def homeepager(cl):
    def pa(cll):
        num_classes = 0
        training_steps = 1500
        ###########################################################################


        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        cwd = os.getcwd()

        # cwd = '/home/ivanrputra/mysite'
        # print(str(os.path.dirname(os.path.abspath(__file__))))
        image_path = os.path.join(cwd,'1.jpg')
        train_path = os.path.join(cwd,'new_train_data')
        save_path = os.path.join(cwd,'hog_saved_weights')
        hog_file_path = os.path.join(cwd,'hog_files')

        # image_path = '/home/ivanrputra/mysite/1.jpg'
        # train_path = '/home/ivanrputra/mysite/new_train_data'
        # save_path = '/home/ivanrputra/mysite/hog_saved_weights'
        # hog_file_path = '/home/ivanrputra/mysite/hog_files'
        cell = [8, 8]
        incr = [8,8]
        bin_num = 8
        im_size = [32,32]
        class_list = []
        train_list = []
        hog_list = []
        total_data = 0
        batch_size = 100
        class_data_count = []
        def create_array(image_path):
    
            image = Image.open(os.path.join(cwd,image_path)).convert('L')
            image_array = np.asarray(image,dtype=float)
            
            return image_array

        #uses a [-1 0 1 kernel]
        def create_grad_array(image_array):
            image_array = Image.fromarray(image_array)
            if not image_array.size == im_size:
                image_array = image_array.resize(im_size, resample=Image.BICUBIC)
            
            image_array = np.asarray(image_array,dtype=float)
            
            # gamma correction
            image_array = (image_array)**2.5

            # local contrast normalisation
            image_array = (image_array-np.mean(image_array))/np.std(image_array)
            max_h = 32
            max_w = 32

            grad = np.zeros([max_h, max_w])
            mag = np.zeros([max_h, max_w])
            for h,row in enumerate(image_array):
                for w, val in enumerate(row):
                    if h-1>=0 and w-1>=0 and h+1<max_h and w+1<max_w:
                        dy = image_array[h+1][w]-image_array[h-1][w]
                        dx = row[w+1]-row[w-1]+0.0001
                        grad[h][w] = np.arctan(dy/dx)*(180/np.pi)
                        if grad[h][w]<0:
                            grad[h][w] += 180
                        mag[h][w] = np.sqrt(dy*dy+dx*dx)
            
            return grad,mag

        def write_hog_file(filename, final_array):
            print('Saving '+filename+' ........\n')
            np.savetxt(filename,final_array)

        def read_hog_file(filename):
            return np.loadtxt(filename)

        def calculate_histogram(array,weights):
            bins_range = (0, 180)
            bins = bin_num
            hist,_ = np.histogram(array,bins=bins,range=bins_range,weights=weights)

            return hist

        def create_hog_features(grad_array,mag_array):
            max_h = int(((grad_array.shape[0]-cell[0])/incr[0])+1)
            max_w = int(((grad_array.shape[1]-cell[1])/incr[1])+1)
            cell_array = []
            w = 0
            h = 0
            i = 0
            j = 0

            #Creating 8X8 cells
            while i<max_h:
                w = 0
                j = 0

                while j<max_w:
                    for_hist = grad_array[h:h+cell[0],w:w+cell[1]]
                    for_wght = mag_array[h:h+cell[0],w:w+cell[1]]
                    
                    val = calculate_histogram(for_hist,for_wght)
                    cell_array.append(val)
                    j += 1
                    w += incr[1]

                i += 1
                h += incr[0]

            cell_array = np.reshape(cell_array,(max_h, max_w, bin_num))
            #normalising blocks of cells
            block = [2,2]
            #here increment is 1

            max_h = int((max_h-block[0])+1)
            max_w = int((max_w-block[1])+1)
            block_list = []
            w = 0
            h = 0
            i = 0
            j = 0

            while i<max_h:
                w = 0
                j = 0

                while j<max_w:
                    for_norm = cell_array[h:h+block[0],w:w+block[1]]
                    mag = np.linalg.norm(for_norm)
                    arr_list = (for_norm/mag).flatten().tolist()
                    block_list += arr_list
                    j += 1
                    w += 1

                i += 1
                h += 1

            #returns a vextor array list of 288 elements
            return block_list

        #image_array must be an array
        #returns a 288 features vector from image array
        def apply_hog(image_array):
            gradient,magnitude = create_grad_array(image_array)
            hog_features = create_hog_features(gradient,magnitude)
            hog_features = np.asarray(hog_features,dtype=float)
            hog_features = np.expand_dims(hog_features,axis=0)

            return hog_features

        #path must be image path
        #returns final features array from image_path
        def create_arrayurl(image_path):

            image = BytesIO(urllib.request.urlopen("http://www.jualbelisaham.esy.es/1.jpg").read())
            image2 = Image.open(image).convert('L')
            # image = Image.open(os.path.join(cwd,image_path)).convert('L')
            image_array = np.asarray(image2,dtype=float)
            
            return image_array

        def hog_from_pathurl(image_path):
            image_array = create_arrayurl(image_path)
            final_array = apply_hog(image_array)
            
            return final_array

        def create_arrayurlclass(image_path):

            url = "http://www.jualbelisaham.esy.es/"+image_path
            image = BytesIO(urllib.request.urlopen(url).read())
            image2 = Image.open(image).convert('L')
            # image = Image.open(os.path.join(cwd,image_path)).convert('L')
            image_array = np.asarray(image2,dtype=float)
            
            return image_array

        def hog_from_pathurlclass(image_path):
            image_array = create_arrayurlclass(image_path)
            final_array = apply_hog(image_array)
            
            return final_array

        def hog_from_path(image_path):
            image_arrayurl = create_array(image_path)
            final_array = apply_hog(image_array)
            
            return final_array

        #Creates hog files
        def create_hog_file(image_path,save_path):
            image_array = create_array(image_path)
            final_array = apply_hog(image_array)
            write_hog_file(save_path,final_array)

        ##########################################################################################
        #reads train data images and makes lists of paths
        def read_train_data_paths(num_classes,total_data):
            # if not os.path.exists(train_path):
            #     os.makedirs(train_path)
            # if not os.path.exists(hog_file_path):
            #   os.makedirs(hog_file_path)

            class_list.extend(os.listdir(train_path))
            num_classes += len(class_list)

            for folder,val in enumerate(class_list):
                class_path = os.path.join(train_path,val)
                hog_path = os.path.join(hog_file_path,val)

                # if not os.path.exists(hog_path):
                #   os.makedirs(hog_path)

                image_list = os.listdir(class_path)
                class_data_count.append(len(image_list))

                for i in image_list:
                    img_path = os.path.join(class_path,i)
                    train_list.append([img_path,folder])

                    #makes paths for cache
                    i = i.replace('.jpg','.txt')
                    i = i.replace('.JPG','.txt')
                    i = i.replace('.jpeg','.txt')
                    i = i.replace('.JPEG','.txt')

                    i = os.path.join(hog_path,i)
                    hog_list.append([i,folder])

            total_data += len(hog_list)

            return num_classes,total_data

        #creates cache in the form of .txt files
        def create_cache():
            for index,image in enumerate(train_list):
                if not os.path.exists(hog_list[index][0]):
                    #the following function is imported from hog file
                    create_hog_file(image[0],hog_list[index][0])
                else:
                    print('Found cache... '+hog_list[index][0])

        #Creates the variables for weights and biases
        def create_variables(num_classes):
            W = tf.Variable(tf.truncated_normal([288, num_classes]),name='weights')
            b = tf.Variable(tf.truncated_normal([1, num_classes]), name='biases')

            return W,b

        #creates labels; uses hog descriptors
        def create_labels(count, hog_list, total_data, batch_size):

            #labels are one-hot vectors. But 0 is replaced with -1
            point = count
            path = hog_list[count][0]
            lab = hog_list[count][1]
            y = np.zeros([1,num_classes])
            y[0][lab] = 1

            x = read_hog_file(path)
            x = np.expand_dims(x,axis=0)

            count += 1
            extra = np.min([batch_size,total_data-point])

            while count<point+extra and count<total_data:
                path = hog_list[count][0]
                lab = hog_list[count][1]

                y_new = np.zeros([1,num_classes])
                y_new[0][lab] = 1
                y = np.concatenate((y,y_new), axis=0)

                x_new = read_hog_file(path)
                x_new = np.expand_dims(x_new,axis=0)
                x = np.concatenate((x,x_new), axis=0)

                count+=1

            return x,y

        #evaluates accuracy
        def evaluate_accuracy(final,labels):
            prediction = tf.argmax(final,axis=1)
            ground_truth = tf.argmax(labels,axis=1)

            evaluate = tf.equal(prediction,ground_truth)
            accuracy = tf.reduce_mean(tf.cast(evaluate,dtype=tf.float32), axis=0)

            return accuracy*100

        #Creates a model for SOFTMAX
        def model(W,b,num_classes):
            x = tf.placeholder(tf.float32,[None, 288])
            y = tf.placeholder(tf.float32,[None, num_classes])

            logits = tf.add(tf.matmul(x,W),b)
            prediction = tf.nn.softmax(logits)

            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
            loss = tf.reduce_mean(loss)

            optimizer = tf.train.AdamOptimizer()
            train_step = optimizer.minimize(loss)
            accuracy = evaluate_accuracy(prediction,y)

            return train_step,accuracy,x,y

        #training in SOFTMAX Logistic mode
        def train_values():
            W,b = create_variables(num_classes)
            train_step, accuracy,x,y = model(W,b,num_classes)

            print('\n--------------------------------------------------------------------')
            print('ONE v/s ALL training - SOFTMAX LOGISTIC MULTICLASSIFIER')
            print('--------------------------------------------------------------------')

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(training_steps):
                    print('\nTraining step : '+str(epoch+1)+' .....................')
                    count = 0
                    while count<total_data:
                        X,Y = create_labels(count, hog_list, total_data, batch_size)

                        _,accu = sess.run([train_step,accuracy],feed_dict={x:X,y:Y})
                        print('Batch training Accuracy : '+str(accu)+' ...')

                        extra = np.min([batch_size,total_data-count])
                        count += extra

                #saving weights
                write_ckpt(W,sess,'weights','LOGIST')
                write_ckpt(b,sess,'biases','LOGIST')

                weight = sess.run(W)
                bias = sess.run(b)

            #Here, test data is randomly selected from the main data set
            k = int(0.1*(len(hog_list)))
            test = generate_random_test(k)
            X,Y = create_labels(0, test, k, k)
            _,pred = classify(X, weight.astype(dtype=np.float32), bias.astype(dtype=np.float32))
            accu = evaluate_accuracy(pred, Y)

            #Accuracy for test
            with tf.Session() as sess:
                print('\nTest Accuracy : '+str(sess.run(accu))+' % ....')

            return weight,bias

        #Classifying using Logistic function
        def classify(X,W,b):
            batch = X.shape[0]
            X = tf.convert_to_tensor(X,dtype=tf.float32)
            logits = tf.add(tf.matmul(X,W),b)
            y = tf.nn.softmax(logits)
            #score is the maximum probability obtained by the classifier
            score = tf.reduce_max(y, axis=1)

            with tf.Session() as sess:
                num = sess.run(tf.argmax(y,axis=1))
                score = sess.run(score)

            #creating label for calculating accuracy
            prediction = np.zeros([batch,num_classes])

            for i in range(batch):
                prediction[i][num[i]] = 1

            return score,prediction

        #Saves weights to file
        def write_ckpt(tensor, sess, name, mode):
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            #saves weights in the respective mode folder
            mode_path = os.path.join(save_path,mode)
            if not os.path.exists(mode_path):
                os.makedirs(mode_path)

            folder_path = os.path.join(mode_path,name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            #saves as a .ckpt file
            saver = tf.train.Saver({name:tensor})
            filename = name+'.ckpt'
            path = os.path.join(folder_path,filename)
            tensor_path = saver.save(sess, path)

            print("\nHog tensor saved at %s", tensor_path)

        #reads .ckpt file and restores variables
        #Variables must be created before calling this
        def read_ckpt(ckpt_path,name,tensor,sess):
            saver = tf.train.Saver({name:tensor})
            saver.restore(sess, ckpt_path)

        #Creating SVM labels
        #key for SVM is taken -1 here
        def create_svm_labels(count, hog_list, total_data, batch_size, class_num, key):
            point = count
            path = hog_list[count][0]
            lab = hog_list[count][1]

            y = np.array([[key]])
            if lab==class_num:
                y[0][0] = 1

            x = read_hog_file(path)
            x = np.expand_dims(x,axis=0)

            count += 1
            extra = np.min([batch_size,total_data-point])

            while count<point+extra and count<total_data:
                path = hog_list[count][0]
                lab = hog_list[count][1]

                y_new = np.array([[key]])
                if lab==class_num:
                    y_new[0][0] = 1

                y = np.concatenate((y,y_new), axis=0)

                x_new = read_hog_file(path)
                x_new = np.expand_dims(x_new,axis=0)
                x = np.concatenate((x,x_new), axis=0)

                count+=1

            return x,y

        #Creates Linear SVM Model
        def Linear_SVM_model(W,b):
            #W must be of shape [288,1]
            x = tf.placeholder(tf.float32,[None, 288])
            y = tf.placeholder(tf.float32,[None, 1])

            # Regularisation constant
            C = 1

            # Model is as follows:
            # hyperplane : hplane = W*x + b
            # cost = (1/n)*sum( max( 0, (1-y*hplane) ) ) + C*||W||^2
            h_plane = tf.add(tf.matmul(x,W),b)
            h_plane = 1.-tf.multiply(y,h_plane)
            cost = tf.maximum(0.,h_plane)
            cost = tf.reduce_mean(cost,axis=0)
            cost += C*tf.reduce_sum(tf.square(W), axis=1)

            optimizer = tf.train.AdamOptimizer()
            train_step = optimizer.minimize(cost)

            return train_step,x,y

        #Generates random test data from the main data list
        #num is the number of data
        def generate_random_test(num):
            test = []

            for i in range(num):
                s = randint(0,total_data)
                test.append(hog_list[s])

            return test

        #Trains SVM model
        #Training each class separately
        #One vs All classification
        def train_SVM():
            print('\n--------------------------------------------------------------------')
            print('ONE v/s ALL training - SVM MULTICLASSIFIER')
            print('--------------------------------------------------------------------')

            W_main = np.zeros([288,num_classes])
            b_main = np.zeros([1,num_classes])
            for i in range(num_classes):
                W = tf.Variable(tf.truncated_normal([288,1]))
                b = tf.Variable(tf.truncated_normal([1,1]))
                print('\nTraining SVM for Class '+str(i+1)+'/'+str(num_classes)+' : ' + class_list[i]+' .......................................\n')
                train_step,x,y = Linear_SVM_model(W,b)

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    for epoch in range(training_steps):
                        print('................ '+str(i+1)+'/'+str(num_classes)+' Training step : '+str(epoch+1)+' ................')
                        count = 0
                        while count<total_data:
                            print('Image: '+str(count+1)+'/'+str(total_data)+' ...')
                            X,Y = create_svm_labels(count, hog_list, total_data, batch_size, i, -1)
                            sess.run(train_step,feed_dict={x:X,y:Y})

                            extra = np.min([batch_size,total_data-count])
                            count += extra

                    #Weights for each class are added to the main matrix as columns
                    W_main[:,i] = (sess.run(W))[:,0]
                    b_main[:,i] = (sess.run(b))[:,0]

            #Generates Test data and tests the trained model
            k = int(0.1*(len(hog_list)))
            test = generate_random_test(k)
            X,Y = create_labels(0, test, k, k)
            _,_,pred = SVM_classify(X, W_main.astype(dtype=np.float32), b_main.astype(dtype=np.float32))
            accu = evaluate_accuracy(pred, Y)
            with tf.Session() as sess:
                print('\nTest Accuracy : '+str(sess.run(accu))+' % ....')

            #Creates weights and biases for saving
            W_final = tf.Variable(W_main.astype(dtype=np.float32),name='weights')
            b_final = tf.Variable(b_main.astype(dtype=np.float32),name='biases')
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                write_ckpt(W_final,sess,'weights','SVM')
                write_ckpt(b_final,sess,'biases','SVM')

            return W_main,b_main

        #Classifier for SVM Model
        def SVM_classify(X,W,b):
            batch = X.shape[0]
            X = tf.convert_to_tensor(X,dtype=tf.float32)
            h_plane = tf.add(tf.matmul(X,W),b)
            #score is the maximum positive distance from the hyperplane
            score = tf.reduce_max(h_plane, axis=1)

            with tf.Session() as sess:
                num = sess.run(tf.argmax(h_plane,axis=1))
                score = sess.run(score)
                plane = sess.run(h_plane)

            #Creating label vector for validating accuracy
            prediction = np.zeros([batch,num_classes])
            for i in range(batch):
                prediction[i][num[i]] = 1

            return score,plane, prediction

        ##################################################################################################
        a,b = read_train_data_paths(num_classes,total_data)
        num_classes += a
        total_data += b
        def class1(img):

            line = "SVM"
            W,b = create_variables(num_classes)
            mode_path = os.path.join(save_path, line)

            with tf.Session() as sess:
                read_ckpt(os.path.join(mode_path,'weights/weights.ckpt'),'weights',W,sess)
                read_ckpt(os.path.join(mode_path,'biases/biases.ckpt'),'biases',b,sess)
                
                W_array = sess.run(W)
                b_array = sess.run(b)

            W_array = tf.convert_to_tensor(W_array, dtype=tf.float32)
            b_array = tf.convert_to_tensor(b_array, dtype=tf.float32)

            image_path = os.path.join(cwd,'1.jpg')

            #Extracting Hog features
            X = hog_from_pathurl(image_path)
            
            #Classifying using mode
            if line=='SVM':
                _,_,prediction = SVM_classify(X,W_array,b_array)
            elif line=='LOGIST':
                _,prediction = classify(X,W_array,b_array)

            a =str(class_list[np.argmax(prediction)])
            
            return a

        def classi(img):
            aaa = "asas"
            line = "SVM"
            W,b = create_variables(num_classes)
            mode_path = os.path.join(save_path, line)

            with tf.Session() as sess:
                read_ckpt(os.path.join(mode_path,'weights/weights.ckpt'),'weights',W,sess)
                read_ckpt(os.path.join(mode_path,'biases/biases.ckpt'),'biases',b,sess)
                
                W_array = sess.run(W)
                b_array = sess.run(b)

            W_array = tf.convert_to_tensor(W_array, dtype=tf.float32)
            b_array = tf.convert_to_tensor(b_array, dtype=tf.float32)

            filename = img+".jpg"
            # image_path = os.path.join(cwd,filename)

            #Extracting Hog features
            X = hog_from_pathurlclass(filename)
            
            #Classifying using mode
            if line=='SVM':
                _,_,prediction = SVM_classify(X,W_array,b_array)
            elif line=='LOGIST':
                _,prediction = classify(X,W_array,b_array)

            a =str(class_list[np.argmax(prediction)])
            
            return a

        return classi(cll)

    return render_template('class.html', hasil=pa(str(cl)))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
