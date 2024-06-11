import tkinter as tk
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import joblib
import time

from PIL import Image, ImageTk
from keras import regularizers
from keras.utils import to_categorical
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import utils


class ImagePredictionApp:
    """class gérant une interface tkinter permettant de faire le prognostic d'un(e) salarié(e) à partir de l'un de ses yeux."""
    def __init__(self, root):
        # chargement des modèles
        self.load_model_two_eyes()
        self.load_model_left_eye()
        self.load_model_right_eye() 
        # chargement des encodeurs
        self.encodeur_OD = joblib.load(
            'code\models_encoders\encodeur_OD.joblib')
        self.encodeur_OG = joblib.load(
            'code\models_encoders\encodeur_OG.joblib')
        self.encodeur_2O = joblib.load(
            'code\models_encoders\encodeur_2O.joblib')

        # paramètrage des var d'affichage
        self.root = root
        self.root.geometry("1200x500")
        self.root.configure(bg='black')

        self.root.title("Application d'Authentification")

        default_button_style = {
            "bg": 'black', "fg": "green", "highlightbackground": "green", "font": ('Consolas', 15)
        }

        default_label_style = {
            "bg": 'black', "fg": "green", "font": ('Consolas', 15)
        }

        default_cadre_style = {
            "bg": "black", "highlightbackground": "green"
        }

        # création de la grille
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_rowconfigure(3, weight=0)
        self.root.grid_rowconfigure(4, weight=0)
        self.root.grid_rowconfigure(5, weight=0)
        self.root.grid_rowconfigure(6, weight=0)

        self.root.grid_columnconfigure(0, minsize=350, weight=350)
        self.root.grid_columnconfigure(1, minsize=1, weight=1)
        self.root.grid_columnconfigure(2, minsize=500, weight=500)
        self.root.grid_columnconfigure(3, minsize=1, weight=1)
        self.root.grid_columnconfigure(4, minsize=350, weight=350)

        #séparation entre les différentes colonnes
        self.separation_1 = tk.Canvas(
            root, width=1, height=500, **default_cadre_style)
        self.separation_1.grid(column=1, row=0, rowspan=7)
        self.separation_2 = tk.Canvas(
            root, width=1, height=500, **default_cadre_style)
        self.separation_2.grid(column=3, row=0, rowspan=7)
        self.separation_3 = tk.Canvas(root, width=350, height=1, **default_cadre_style)
        self.separation_3.grid(column=4, row=5)

        #contenu première colonne => identité du salarié(e)
        self.cadre_employe = tk.Canvas(
            root, width=150, height=200, **default_cadre_style)
        self.cadre_employe.grid(column=0, row=0, rowspan=3)
        self.infos = tk.Label(
            root, text="prénom:\nnom:\nannée d'embauche:     \ngenre:\nposte:", **default_label_style, justify="left")
        self.infos.grid(column=0, row=3, rowspan=6)

        #contenu deuxième colonne => affichage de l'oeil + btn de lancement de l'analyse
        self.select_button = tk.Button(
            root, text="Sélectionner une image", command=self.load_image, **default_button_style)
        self.select_button.grid(column=2, row=0)
        self.cadre_img = tk.Canvas(
            root, width=330, height=300, **default_cadre_style)
        self.cadre_img.grid(column=2, row=1, rowspan=3)
        self.predict_button = tk.Button(
            root, text="Lancer la Prédiction", command=self.predict_image, **default_button_style)
        self.predict_button.grid(column=2, row=5)
        self.imagepath_label = tk.Label(root)
        self.imagepath_label.grid(column=2, row=6)

        # contenu troisième colonnne => camenberts de confiance des prédictions (oeil + salarié(e)) + temps d'exécution
        self.graph_created = False  # var pour suivre si la méthode bar_graph a été créée
        self.graph_frame = tk.Canvas(root)
        self.graph_frame.grid(column=4, row=0, rowspan=5)
        self.prediction_time = tk.Label(root, text='temps: 0.00s', **default_label_style)
        self.prediction_time.grid(column=4, row=6)
        self.graph()

    def load_model_two_eyes(self):
        """chargement du modèle de détermination de l'oeil"""
        self.model_two_eyes = tf.keras.models.load_model(
            r'code\models_encoders\vgg16_side_O2ID_classif')
        self.model_two_eyes.trainable = False

    def load_model_left_eye(self):
        """chargement du modèle de prédiction de l'employé en fonction de son oeil gauche"""
        self.model_left_eye = tf.keras.models.load_model(
            r'code\models_encoders\vgg16_side_OG2ID_classif')
        self.model_left_eye.trainable = False

    def load_model_right_eye(self):
        """chargement du modèle de prédiction de l'employé en fonction de son oeil droit"""
        self.model_right_eye = tf.keras.models.load_model(
            r'code\models_encoders\vgg16_side_OD2ID_classif')
        self.model_right_eye.trainable = False

    def load_image(self):
        """chargement de l'image à analyser"""
        file_path = tk.filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            # Redimensionnez l'image pour l'affichage
            self.image_prep = utils.preprocess_img(self.image)
            self.imagepath_label.config(text=f"Image : {file_path}")
            self.photo = ImageTk.PhotoImage(Image.open(file_path))
            self.cadre_img.create_image(7, 2, image=self.photo, anchor="nw")
            self.reinitilise_predict()
            self.delete_avatar()
  
    def load_avatar(self, genre: str):
        """affichage de l'avatar suivant le sexe du salarié(e)"""
        if genre == 'Femme':
            avatar_load = Image.open(r'code\avatar\matrix_trinity.jpg')
        else:
            avatar_load = Image.open(r'code\avatar\matrix_neo.jpg')
        self.avatar = ImageTk.PhotoImage(avatar_load.resize((150, 200), Image.LANCZOS))
        self.avatar_load = self.cadre_employe.create_image(7, 2, image=self.avatar, anchor='nw')

    def delete_avatar(self):
        '''suppression de l'avatar'''
        self.cadre_employe.delete(self.load_avatar)

    def reinitilise_predict(self):
        """réinitialisation des données de l'employé(e), des camemberts et du temps de réalisation des prédictions"""
        self.prediction_time.config(text= 'temps: 0.00s')
        self.infos.config(
            text="prénom:\nnom:\nannée d'embauche:     \ngenre:\nposte:")
        if self.graph_created:
            self.graph()

    def predict_image(self):
        """prédiction de l'oeil puis prédiction de l'employé-e en fonction de l'oeil gauche ou droit"""
        if hasattr(self, 'image'):
            debut = time.time()
            prediction_side_probs = self.model_two_eyes.predict(
                np.array([self.image_prep]))
            prediction_side_result = np.argmax(prediction_side_probs)
          
            if prediction_side_result == 0:
                self.prediction_probs = self.model_left_eye.predict(
                    np.array([self.image_prep]))
                print('argmax', np.argmax(self.prediction_probs))
                print('prediction_probs', self.prediction_probs)
                self.prediction = self.encodeur_OG.inverse_transform(                    
                    [np.argmax(self.prediction_probs)])
                self.encodeur_select = self.encodeur_OG                

            elif prediction_side_result == 1:
                self.prediction_probs = self.model_right_eye.predict(
                    np.array([self.image_prep]))
                self.prediction = self.encodeur_OD.inverse_transform(
                    [np.argmax(self.prediction_probs)])
                self.encodeur_select = self.encodeur_OD
            
            prenom, nom, annee_embauche, genre, poste = utils.extract_data_json(int(self.prediction))
            self.infos.config(
                text=f"prénom: {prenom}\nnom: {nom}\nannée d'embauche: {annee_embauche}\ngenre: {genre}\nposte: {poste}")
            
            two_best_values, labels = utils.best_values(self.prediction_probs[0], self.encodeur_select)
            
            self.graph(["Gauche", "Droit"], labels, prediction_side_probs[0], two_best_values)

            self.load_avatar(genre)
            
            fin = time.time()
            self.prediction_time.config(text = f'temps: {round(fin - debut, 2)}s')

        else:
            self.cadre_img.create_text(170, 100, text="Aucune image sélectionnée", fill='green', font=("Consolas"))

    def graph(self, labels1: list=['??', '??'], labels2: list =['??', '??'], predictions1 : list =[100, 0], predictions2 : list= [100, 0]):
        """affichage des camemberts rapportants le degré de confiance dans la prédiction pour l'oeil et pour l'employé-e"""
        if self.graph_created:
            self.destroy_graph()

        font = {'family': 'Consolas',
                'size': 10}
        plt.rc('font', **font)
        plt.style.use('dark_background')
        self.fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(4, 4))
        self.fig.subplots_adjust(hspace=0.5)  # Ajustez l'espacement vertical

        ax1.set_facecolor('black')
        ax2.set_facecolor('black')

        # Créez les camemberts pour chaque sous-graphique
        ax1.pie([val * 100 for val in predictions1], labels=labels1,
                autopct='%1.1f%%', startangle=0, colors=['#008000', '#00FF00'])
        ax2.pie([val * 100 for val in predictions2], labels=labels2,
                autopct='%1.1f%%', startangle=0, colors=['#008000', '#00FF00'])

        ax1.set_title('Prédiction oeil', color='green')
        ax2.set_title('Prédiction employé-e', color='green')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill='x', expand=True)

        self.canvas_widget.configure(bg='black')
        self.graph_frame.configure(bg='black', highlightbackground='green')

        self.graph_created = True

    def destroy_graph(self):
        '''Détruit les camemberts'''
        self.fig.clf()
        plt.close(self.fig)
        self.canvas_widget.destroy()
        self.bar_graph_created = False


if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePredictionApp(root)
    root.mainloop()
