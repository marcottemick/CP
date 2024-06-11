import cv2
import json
import joblib

encodeur_OD = joblib.load(
    r'code\models_encoders\encodeur_OD.joblib')
encodeur_OG = joblib.load(
    r'code\models_encoders\encodeur_OG.joblib')

def preprocess_img(img,new_dim=(240,320)):
    new_img=cv2.resize(img, (new_dim[1],new_dim[0]), interpolation = cv2.INTER_AREA)
    new_img=new_img / 255.0
    return new_img

def extract_data_json(index: int):
    with open('./employees_info.json', 'r', encoding='utf-8') as fichier:
        datas_employes = json.load(fichier)
        info_employe = datas_employes[str(index)]
        nom_prenom =  info_employe['nom'].split(' ')
        prenom = nom_prenom[0]
        nom = nom_prenom[1]
        annee_embauche = info_employe['annee_embauche']
        genre = info_employe['genre']
        poste = info_employe['poste']

        return prenom, nom, annee_embauche, genre, poste

def best_values(data: list, labelencoder):
    data_list = data.tolist()
    data_sort = data_list.copy()
    data_sort.sort(reverse=True)
    two_best_values = data_sort[0: 2]

    index1 = labelencoder.inverse_transform([data_list.index(two_best_values[0])])
    index2 = labelencoder.inverse_transform([data_list.index(two_best_values[1])])

    prenom1, nom1, annee1, genre1, poste1 = extract_data_json(int(index1))
    prenom2, nom2, annee2, genre2, poste2 = extract_data_json(int(index2))

    labels = [str(index1[0]) + ":\n" + prenom1 + ' ' + nom1, str(index2[0]) + ":\n" + prenom2 + ' ' + nom2]

    return two_best_values, labels

