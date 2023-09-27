from django.shortcuts import render
from app.models import breast,lung,skin,RiskModel
from .forms import breastForm,lungForm,skinForm,riskForm,UserForm
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate,login,logout
from django.http import HttpResponseRedirect,HttpResponse



# Create your views here.
@login_required
def home(request):


    return render(request,'app/home.html',{})


def breast_photo_upload(request):
    if request.method == 'POST':
        form = breastForm(request.POST,request.FILES)
        if form.is_valid():
            form.save()
            return redirect('breast_cancer2')
    else:
        form = breastForm()


    return render(request, 'app/breast_photo_upload.html', {})


def feature_extraction():
    import math
    import numpy as np
    import cv2
    from PIL import Image
    import pandas as pd
    from scipy.stats import entropy
    from matplotlib import pyplot as plt
    import skimage.measure
    img = cv2.imread('images/breast/')
    p = cv2.calcHist([img], [0], None, [256], [0, 256])
    # p = cv2.imhist(img)
    size = 1024 * 1024
    p_norm = np.divide(p, size)
    mean = 0
    for z in range(0, 256):
        mean = mean + ((z - 1) * p_norm[z])
    var = 0
    for z in range(0, 256):
        var = var + (((z - 1) - mean) ** 2) * p_norm[z]
    std = math.sqrt(var)
    R = 1 - (1 / (1 + std ** 2))
    skew = 0
    for z in range(0, 256):
        skew = skew + ((((z - 1) - mean) ** 3) * p_norm[z])
    U = 0
    for z in range(0, 256):
        U = U + (p_norm[z] ** 2)
    E = skimage.measure.shannon_entropy(img)
    print(E)
    feature_vec = [mean[0], std, R, skew[0], U[0], E]
    fid = open('app/breast/features_input.txt', 'w');
    string = ' '.join([str(elem) for elem in feature_vec])
    print(string)
    fid.write(string);
    fid.close()

def breast_main():
    import numpy as np
    import csv
    from sklearn import svm
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn import tree
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn import metrics

    # feature_file = csv.reader(open('features.txt'), delimiter=" ")
    label_file = csv.reader(open('app/breast/labels.txt'), delimiter=" ")

    data_y = []

    data_X = np.loadtxt('app/breast/features.txt')
    print(data_X, data_X.shape)

    for row in label_file:
        data_y.append(row)

    data_y = np.array(data_y).astype(np.float)
    print(data_y.shape, data_y)

    train_X1 = data_X[0:50, :]
    train_X2 = data_X[81:, :]
    train_y1 = data_y[0:50, :]
    train_y2 = data_y[81:, :]
    X_train = np.concatenate((train_X1, train_X2), axis=0)
    y_train = np.concatenate((train_y1, train_y2), axis=0)
    y_train = y_train.reshape(-1)

    X_test = data_X[50:80, :]
    print(X_test[0])
    y_test = data_y[50:80, :]
    y_test = y_test.reshape(-1)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # clf = svm.NuSVC(nu=0.4) #63.3%
    clf = svm.SVC(C=2, kernel='rbf', degree=2, gamma=100)  # 76.6%
    # clf = LogisticRegression(C=1, penalty='l1')
    # clf = GradientBoostingClassifier()
    # clf = tree.DecisionTreeClassifier()
    # clf = AdaBoostClassifier(n_estimators = 100)
    # clf = svm.LinearSVC(C=100000)
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # input
    X_input = np.loadtxt('app/breast/features_input.txt')
    x = np.array(X_input)
    X = [x]
    X1 = np.array(X)
    Y = model.predict(X)
    if Y == 0:
        return ("The Result is Benign")
    else:
        return ("The Result is Malignant, Please Consult your Doctor")

@login_required
def breast_cancer(request):
    feature_extraction()
    x=breast_main()
    breast.objects.all().delete()
    return render(request, 'app/breast_cancer.html', {'x':x})

################### lung ###############
def lung_photo_upload(request):
    if request.method == 'POST':
        form = lungForm(request.POST,request.FILES)
        if form.is_valid():
            form.save()
            return redirect('lung_cancer2')
    else:
        form = lungForm()


    return render(request, 'app/lung_photo_upload.html', {})

def lung_main():
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.models import model_from_json
    from keras.models import load_model
    import numpy as np
    from PIL import Image
    import tensorflow as tf
    import os
    path = 'images/lung/input'
    json_file = open('app/lung/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("app/lung/model.h5")
    print("Loaded model from disk")

    validate_it = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(128, 128),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )
    val_pred_proba = loaded_model.predict(validate_it)
    val_pred = np.argmax(val_pred_proba, axis=1)
    if (val_pred == 0):
        return ('The result is Benign')
    elif (val_pred == 1):
        return ('The result is Malignant and it is adenocarcinoma,Please Consult your doctor')

    elif (val_pred == 2):
        return ('The result is Malignant and it is squamous cell carcinoma, Please Consult your doctor')


@login_required
def lung_cancer(request):
    x=lung_main()
    lung.objects.all().delete()
    return render(request, 'app/lung_cancer.html', {'x': x})



################## skin cancer ####################

def skin_photo_upload(request):
    if request.method == 'POST':
        form = skinForm(request.POST,request.FILES)
        if form.is_valid():
            form.save()
            return redirect('skin_cancer2')
    else:
        form = skinForm()


    return render(request, 'app/skin_photo_upload.html', {})


def skin_main():
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.models import model_from_json
    from keras.models import load_model
    import os
    import numpy as np
    from PIL import Image
    json_file = open('app/skin/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("app/skin/model.h5")
    print("Loaded model from disk")
    folder_input = 'images/skin/input'
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    print("done reading")
    ims_input = [read(os.path.join(folder_input, filename)) for filename in os.listdir(folder_input)]
    X_input = np.array(ims_input, dtype='uint8')
    X_input = X_input / 255.
    y_output = loaded_model.predict_classes(X_input)
    if y_output == 0:
        return ('The result is Benign')
    else:
        return ('The result is Malignant, Please consult your doctor')


@login_required
def skin_cancer(request):
    x = skin_main()
    skin.objects.all().delete()
    return render(request, 'app/skin_cancer.html', {'x': x})
@login_required
def risk_analysis_data_submit(request):
    if request.method == 'POST':
        form = riskForm(request.POST,request.FILES)
        if form.is_valid():
            form.save()
            return redirect('risk_analysis2')
        else:
            print(form.errors)
    else:

        form = riskForm()
    return render(request, 'app/Risk_analysis.html', {'form':form})


@login_required
def risk_analysis(request):
    import pandas as pd
    import pickle

    x = RiskModel.objects.all().last()
    lc = [[x.air, x.alchol, x.dust, x.occupation, x.genetic, x.chronic, x.diet, x.obesity, x.smoking, x.passive_smoke,
           x.chestPain, x.cough_blood, x.fatigue]]
    print(lc)
    bc = [[x.age, x.race, x.history, x.menarch, x.birth, x.biRads, x.harmone, x.menopause, x.bmi, x.biopsy, x.prior]]
    print(bc)
    sc = [x.skin_lesions]
    print(sc)


    filename1 = 'app/risk_analysis/lung_cancer.sav'
    loaded_model1 = pickle.load(open(filename1, 'rb'))
    result1 = loaded_model1.predict(lc)
    filename2 = 'app/risk_analysis/breast_cancer.sav'
    loaded_model2 = pickle.load(open(filename2, 'rb'))
    result2 = loaded_model2.predict(bc)
    final1=(result1[0])
    print(final1)
    final2=''
    final3=''
    final4=''
    if (result2 == 0):
        final2=('Low Risk')
    if (result2 == 1):
        final2=('High Risk')
    if (sc[0] == 0):
        final3=('Low Risk')
    if (sc[0] == 1):
        final3=('High Risk')
    if final1 == 'Low' and final2 == 'Low Risk' and final3 == 'Low Risk':
        final4="Low"
    print(final2)
    print(final3)
    return render(request, 'app/risk_analysis_result.html', {'final1':final1,'final2':final2,'final3':final3,'final4':final4})


def user_login(request):
    if request.method =='POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(username=username,password=password)
        if user:
            if user.is_active:
                login(request,user)

                return redirect('risk_analysis')
            else:
                return HttpResponse("Account not active")
        else:
            return HttpResponse('please enter login details correctly')
    else:
        return render(request,'app/login.html',{})

def register(request):
    registered = False

    if request.method == 'POST':
        user_form = UserForm(data=request.POST)

        if user_form.is_valid():
            user = user_form.save()
            user.set_password(user.password)
            user.save()

            registered = True
        else:
            print(user_form.errors)

    else:
        user_form = UserForm()

    return render(request,'app/registration.html',{'user_form':user_form,'registered':registered})

def user_logout(request):
    logout(request)
    print('hey')
    return redirect('user_login')

