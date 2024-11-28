import csv, json

def calculateClassModel(cl, data_csv):
    class_index = data_csv[0].index(cl)
    class_model = []
    for i in range(1, len(data_csv)):
        class_model.append(data_csv[i][class_index])

    classes = list(set(class_model))
    class_model = [[cls, 0] for cls in classes]
    for cls in class_model:
        for i in range(1, len(data_csv)):
            if cls[0] == data_csv[i][class_index]:
                cls[1] += 1
        cls.append(cls[1] / (len(data_csv) - 1))
    
    class_model.append(class_index)
    return class_model

def calculateAttributeModel(att, class_model, data_csv):
    att_index = data_csv[0].index(att)
    class_index = class_model[len(class_model) - 1]

    att_model = []
    for i in range(1, len(data_csv)):
        att_model.append(data_csv[i][att_index])

    attributes = list(set(att_model))
    att_model = []
    for att in attributes:
        for i in range(len(class_model) - 1):
            att_model.append([att, 0, class_model[i][0], class_model[i][1]])

    for att in att_model:
        for i in range(1, len(data_csv)):
            if (att[0] == data_csv[i][att_index] and att[2] == data_csv[i][class_index]):
                att[1] += 1
        att.append(att[1]/ att[3])
    
    return att_model


def buildNBModel(filename):
    with open('../data/'+ filename +'.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data_csv = [row for row in reader]
    
    class_model = calculateClassModel('play', data_csv)

    array_model = []
    for i in range(len(data_csv[0]) - 1):
        array_model.append(calculateAttributeModel(data_csv[0][i], class_model, data_csv))

    json_model = {}
    for i in range(len(class_model) - 1):
        json_model[class_model[i][0]] = {
            "freq": class_model[i][1],
            "p": round(class_model[i][2], 2)
        }

    for attribute in array_model:
        for data_class in attribute:
            json_model[data_class[2]][data_class[0]] = {
                "freq": data_class[1],
                "p" : round(data_class[4], 2)
            }

    return json_model

def featNB(model):
    pass

print(json.dumps(buildNBModel("dummy"), indent = 4))




