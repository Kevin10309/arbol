import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tabulate import tabulate

class SpamDetector:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.stopwords = {'el', 'y', 'es', 'en', 'para', 'de', 'a', 'por', 'con', 'esto'}
        self.spam_keywords = {'whatsapp', 'dinero', 'gratis', 'oportunidad', 'iphone'}

    def preprocess_text(self):
        self.data.drop_duplicates(subset='contenido', inplace=True)
        self.data['contenido'] = self.data['contenido'].str.lower().str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        self.data['contenido'] = self.data['contenido'].str.split()
        self.data['contenido'] = self.data['contenido'].apply(lambda x: [word for word in x if word not in self.stopwords])

    def vectorize_text(self):
        self.data['contenido'] = self.data['contenido'].str.join(' ')
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(self.data['contenido'])
        return features, vectorizer.get_feature_names_out()

    def apply_rules(self):
        self.data['etiqueta'] = 'not_spam'
        for index, row in self.data.iterrows():
            contenido = set(row['contenido'])
            if any(keyword in contenido for keyword in self.spam_keywords):
                self.data.at[index, 'etiqueta'] = 'spam'
            elif any(link in str(row['enlaces']) for link in ['spamlink.com', 'spamiphone.com']):
                self.data.at[index, 'etiqueta'] = 'spam'
            elif row['remitente'] not in {'SPAM1@mail.com', 'amigo@mail.com', 'SPAM2@mail.com', 'boss@mail.com'}:
                self.data.at[index, 'etiqueta'] = 'spam'
            elif any(keyword in row['asunto'] for keyword in {'dinero', 'iphone'}):
                self.data.at[index, 'etiqueta'] = 'spam'

    def calculate_probabilities(self):
        total_correos = len(self.data)
        correos_spam = (self.data['etiqueta'] == 'spam').sum()
        correos_no_spam = total_correos - correos_spam
        
        P_spam = correos_spam / total_correos if total_correos > 0 else 0
        P_no_spam = 1 - P_spam
        
        features, feature_names = self.vectorize_text()
        df_features = pd.DataFrame(features.toarray(), columns=feature_names)
        df_features['etiqueta'] = self.data['etiqueta']

        spam_features = df_features[df_features['etiqueta'] == 'spam'].drop(columns=['etiqueta'])
        non_spam_features = df_features[df_features['etiqueta'] == 'not_spam'].drop(columns=['etiqueta'])

        total_spam_words = spam_features.sum().sum()
        total_non_spam_words = non_spam_features.sum().sum()

        P_caracteristicas_spam = spam_features.sum() / total_spam_words if total_spam_words > 0 else 0
        P_caracteristicas_no_spam = non_spam_features.sum() / total_non_spam_words if total_non_spam_words > 0 else 0

        return P_spam, P_no_spam, P_caracteristicas_spam, P_caracteristicas_no_spam

    def evaluate_model(self):
        clasificaciones = self.data['etiqueta'].values
        etiquetas_reales = self.data['etiqueta'].values
        
        precision = np.mean(clasificaciones == etiquetas_reales) 
        tp = np.sum((clasificaciones == 'spam') & (etiquetas_reales == 'spam'))
        fn = np.sum((clasificaciones == 'not_spam') & (etiquetas_reales == 'spam'))
        recuperacion = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return precision, recuperacion
    
    def get_processed_data(self):
        return self.data

# Uso de la clase
spam_detector = SpamDetector('Correo.csv')
spam_detector.preprocess_text()
spam_detector.apply_rules()

data = spam_detector.get_processed_data()
print("\nDatos procesados:")
print(tabulate(data.head(), headers='keys', tablefmt='pretty'))

P_spam, P_no_spam, P_caracteristicas_spam, P_caracteristicas_no_spam = spam_detector.calculate_probabilities()
precision, recuperacion = spam_detector.evaluate_model()

print("\nResultados del análisis de spam:")
print(f'Probabilidad previa de spam: {P_spam:.4f}')
print(f'Probabilidad de características dado spam:\n{P_caracteristicas_spam.to_string()}')
print(f'Probabilidad de características dado no spam:\n{P_caracteristicas_no_spam.to_string()}')
print(f'Precisión: {precision:.4f}, Recuperación: {recuperacion:.4f}')
