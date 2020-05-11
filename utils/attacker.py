import keras
import numpy as np
import foolbox
from tqdm import tqdm

def attackFGSM(model, input_x, input_y, bounds):
    keras.backend.set_learning_phase(0)
    kerasmodel = foolbox.models.KerasModel(model, bounds=bounds)
    adv_x = []
    succ_case = []
    for i in tqdm(range(len(input_x))):
        attack = foolbox.attacks.FGSM(kerasmodel)
        img = attack(input_x[i], input_y[i])
        if not(img is None):
            succ_case.extend([i])
            adv_x.append(img)
        else:
            print("FGSM Attack Failed.")
        attack = None
    del input_x, input_y
    del kerasmodel,attack,model
    return np.asarray(adv_x),succ_case

def attackDeepFool(model, input_x, input_y, bounds):
    keras.backend.set_learning_phase(0)
    kerasmodel = foolbox.models.KerasModel(model, bounds=bounds)
    adv_x = []
    succ_case = []
    for i in tqdm(range(len(input_x))):
        attack = foolbox.attacks.DeepFoolAttack(kerasmodel)
        img = attack(input_x[i], input_y[i])
        if not(img is None):
            succ_case.extend([i])
            adv_x.append(img)
        attack = None
    advarray = np.asarray(adv_x)
    # org_y = input_y[succ_case]
    # sy=[]
    # for j in range(len(adv_x)):
    #     x = advarray[j]
    #     y1 = model.predict_classes(x[np.newaxis, ...])[0]
    #     sy.append(y1)
    #     y2 = model.predict_classes(adv_x[j][np.newaxis, ...])[0]
    #     if y1==org_y[j]:
    #         print("error values changes")
    #     if y2==org_y[j]:
    #         print("label values changes")
    # yy = model.predict_classes(advarray, batch_size=1)
    # print(np.sum(yy==org_y))
    # print(np.sum(yy == sy))
    # score = model.evaluate(advarray, to_categorical(input_y[succ_case]), batch_size=0, verbose=0)
    # print(score)
    del input_x, input_y
    del kerasmodel,attack,model
    return advarray, succ_case

def attackCWl2(model, input_x, input_y, bounds):
    keras.backend.set_learning_phase(0)

    adv_x = []
    succ_case = []
    for i in tqdm(range(len(input_x))):
        #print(input_x[i].shape)
        kerasmodel = foolbox.models.KerasModel(model, bounds=bounds)
        attack = foolbox.attacks.CarliniWagnerL2Attack(kerasmodel)
        img = attack(input_x[i], input_y[i])
        if not(img is None):
            succ_case.extend([i])
            adv_x.append(img)
        else:
            print("CW Attack Failed.")
        attack = None
    del input_x, input_y
    del kerasmodel,attack,model
    return np.asarray(adv_x), succ_case

def attackJSMA(model, input_x, input_y, bounds):
    keras.backend.set_learning_phase(0)
    adv_x = []
    kerasmodel = foolbox.models.KerasModel(model, bounds=bounds)
    succ_case = []
    for i in tqdm(range(len(input_x))):
        attack = foolbox.attacks.SaliencyMapAttack(kerasmodel)
        img = attack(input_x[i], input_y[i])
        if not(img is None):
            succ_case.extend([i])
            adv_x.append(img)
        else:
            print("JSMA Attack Failed.")
        attack = None
    del input_x, input_y
    del kerasmodel,attack,model
    return np.asarray(adv_x), succ_case

def attackBIM(model, input_x, input_y, bounds):
    keras.backend.set_learning_phase(0)
    kerasmodel = foolbox.models.KerasModel(model, bounds=bounds)
    adv_x = []
    succ_case = []
    for i in tqdm(range(len(input_x))):
        attack = foolbox.attacks.BIM(kerasmodel, distance=foolbox.distances.Linfinity)
        img = attack(input_x[i], input_y[i])
        if not(img is None):
            succ_case.extend([i])
            adv_x.append(img)
        else:
            print("BIM Attack Failed.")
        attack = None
    del input_x, input_y
    del kerasmodel,attack,model
    return np.asarray(adv_x), succ_case

def test_foobox():
    import keras
    import numpy as np
    import foolbox

    keras.backend.set_learning_phase(0)
    kmodel = keras.applications.resnet50.ResNet50(weights='imagenet')
    preprocessing = (np.array([104, 116, 123]), 1)
    model = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

    image, label = foolbox.utils.imagenet_example()
    # apply attack on source image
    attack = foolbox.attacks.CarliniWagnerL2Attack(model)
    adversarial = attack(image[:, :, ::-1], label)
#test_foobox()
