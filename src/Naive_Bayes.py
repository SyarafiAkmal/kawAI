import csv
import json

with open('data/dummy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    array_2d = [row for row in reader]


def calculateClassModel(cl):
    class_index = array_2d[0].index(cl)
    class_model = []
    for i in range(1, len(array_2d)):
        class_model.append(array_2d[i][class_index])

    classes = list(set(class_model))
    class_model = [[cls, 0] for cls in classes]
    for cls in class_model:
        for i in range(1, len(array_2d)):
            if cls[0] == array_2d[i][class_index]:
                cls[1] += 1
        cls.append(cls[1] / (len(array_2d) - 1))
    
    class_model.append(class_index)
    return class_model

def calculateAttributeModel(att, class_model):
    att_index = array_2d[0].index(att)
    class_index = class_model[len(class_model) - 1]

    att_model = []
    for i in range(1, len(array_2d)):
        att_model.append(array_2d[i][att_index])

    attributes = list(set(att_model))
    att_model = []
    for att in attributes:
        for i in range(len(class_model) - 1):
            att_model.append([att, 0, class_model[i][0], class_model[i][1]])

    for att in att_model:
        for i in range(1, len(array_2d)):
            if (att[0] == array_2d[i][att_index] and att[2] == array_2d[i][class_index]):
                att[1] += 1
        att.append(att[1]/ att[3])
    
    return att_model


def buildNBModel(filename):
    class_model = calculateClassModel('play')

    array_model = []
    for i in range(len(array_2d[0]) - 1):
        array_model.append(calculateAttributeModel(array_2d[0][i], class_model))

    json_model = {
        "yes": {},
        "no": {}
    }

    for attribute in array_model:
        for data_class in attribute:
            json_model[data_class[2]][data_class[0]] = {
                "freq": data_class[1],
                "p" : round(data_class[4], 2)
            }


    with open("model/"+ filename + ".json", "w") as file:
        json.dump(json_model, file, indent=4)

buildNBModel("dummy_NB_model")




