import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class SpamDetector:
    def __init__(self, file_path):
        """Inicializa la clase SpamDetector con el archivo CSV."""
        self.file_path = file_path
        self.data = pd.read_csv(file_path)  # Cargar el archivo CSV
        self.stopwords = set(['el', 'y', 'es', 'en', 'para', 'de', 'a', 'por', 'con', 'esto'])  # Lista ejemplo de stopwords
        self.spam_keywords = ['whatsapp', 'dinero', 'gratis', 'oportunidad', 'iphone']  # Palabras clave de spam

    def preprocess_text(self):
        """Realiza el preprocesamiento del texto: eliminación de duplicados, minúsculas, caracteres especiales, etc."""
        # Eliminar correos duplicados
        self.data = self.data.drop_duplicates(subset='contenido')
        
        # Convertir todo el texto a minúsculas
        self.data['contenido'] = self.data['contenido'].str.lower()  # Minusculas
        
        # Eliminar caracteres especiales
        self.data['contenido'] = self.data['contenido'].str.replace(r'[^a-zA-Z0-9\s]', '')  # Eliminar caracteres especiales
        
        # Dividir el texto en palabras (tokenización)
        self.data['contenido'] = self.data['contenido'].str.split()  # Dividir en palabras
        
        # Eliminar las stopwords
        self.data['contenido'] = self.data['contenido'].apply(lambda x: [word for word in x if word not in self.stopwords])  # Eliminar stopwords

    def vectorize_text(self):
        """Vectoriza el texto utilizando TfidfVectorizer."""
        self.data['contenido'] = self.data['contenido'].str.join(' ')  # Juntar las palabras nuevamente en una cadena
        vectorizer = TfidfVectorizer(stop_words='english')  # Crear vectorizador
        features = vectorizer.fit_transform(self.data['contenido'])  # Vectorizar el texto
        return features, vectorizer.get_feature_names_out()

    def apply_rules(self):
        """Aplica las reglas de clasificación para identificar los correos spam."""
        for index, row in self.data.iterrows():
            # Regla 1: Si el correo electrónico contiene la palabra clave "Whatsapp", entonces es Spam.
            if 'whatsapp' in row['contenido']:
                self.data.at[index, 'etiqueta'] = 'spam'
            # Regla 2: Si el correo electrónico contiene un enlace a un sitio web conocido por distribuir malware, entonces es spam.
            elif 'spamlink.com' in row['enlaces'] or 'spamiphone.com' in row['enlaces']:
                self.data.at[index, 'etiqueta'] = 'spam'
            # Regla 3: Si el correo electrónico tiene un remitente desconocido, entonces es Spam.
            elif row['remitente'] not in ['SPAM1@mail.com', 'amigo@mail.com', 'SPAM2@mail.com', 'boss@mail.com']:  # Remitentes conocidos
                self.data.at[index, 'etiqueta'] = 'spam'
            # Regla 4: Si el correo electrónico tiene un asunto que es demasiado bueno para ser verdad, entonces es Spam.
            elif 'dinero' in row['asunto'] or 'iphone' in row['asunto']:
                self.data.at[index, 'etiqueta'] = 'spam'
            # Regla 5: Si el correo electrónico está mal escrito o tiene errores gramaticales, entonces es Spam.
            # (Aquí, por simplicidad, consideramos un ejemplo básico de errores gramaticales)
            elif 'haz sido' in row['contenido'] or 'gratis' in row['contenido']:
                self.data.at[index, 'etiqueta'] = 'spam'

      def calculate_probabilities(self):
        """Calcula la probabilidad previa de spam y las probabilidades de las características en spam y no spam."""
        total_correos = len(self.data)  # Total de correos
        correos_spam = len(self.data[self.data['etiqueta'] == 'spam'])  # Número de correos spam
        correos_no_spam = total_correos - correos_spam  # Número de correos no spam
        
        P_spam = correos_spam / total_correos  # Probabilidad de spam
        P_no_spam = 1 - P_spam  # Probabilidad de no spam
        
        features, feature_names = self.vectorize_text()
        features_array = features.toarray()
        df_features = pd.DataFrame(features_array, columns=feature_names)
        df_features['etiqueta'] = self.data['etiqueta']
        
        spam_features = df_features[df_features['etiqueta'] == 'spam'].drop(columns=['etiqueta'])
        non_spam_features = df_features[df_features['etiqueta'] == 'not_spam'].drop(columns=['etiqueta'])
        
        P_caracteristicas_spam = spam_features.sum() / spam_features.sum().sum()  # Probabilidad de características en spam
        P_caracteristicas_no_spam = non_spam_features.sum() / non_spam_features.sum().sum()  # Probabilidad de características en no spam
        
        return P_spam, P_no_spam, P_caracteristicas_spam, P_caracteristicas_no_spam
    
    def classify_emails(self):
        """Clasifica los correos electrónicos como spam o no spam usando el Teorema de Bayes."""
        P_spam, P_no_spam, P_caracteristicas_spam, P_caracteristicas_no_spam = self.calculate_probabilities()
        features, feature_names = self.vectorize_text()
        features_array = features.toarray()
        df_features = pd.DataFrame(features_array, columns=feature_names)
        
        # Aplicación de la fórmula de Bayes
        P_spam_given_features = (P_spam * P_caracteristicas_spam) / (
            P_spam * P_caracteristicas_spam + P_no_spam * P_caracteristicas_no_spam
        )
        
        # Clasificación basada en la probabilidad calculada
        self.data['prediccion'] = np.where(P_spam_given_features.sum(axis=1) > 0.5, 'spam', 'not_spam')
    
    def evaluate_model(self):
        """Evalúa el modelo calculando precisión y recuperación."""
        clasificaciones = self.data['prediccion'].values
        etiquetas_reales = self.data['etiqueta'].values
        
        precision = np.sum(clasificaciones == etiquetas_reales) / len(clasificaciones)  # Precisión
        recuperacion = np.sum((clasificaciones == 'spam') & (etiquetas_reales == 'spam')) / np.sum(etiquetas_reales == 'spam')  # Recuperación
        
        return precision, recuperacion
    
# Uso de la clase SpamDetector
spam_detector = SpamDetector('Correo.csv')
spam_detector.preprocess_text()  # Preprocesar el texto
spam_detector.apply_rules()  # Aplicar reglas de detección de spam
spam_detector.classify_emails()  # Clasificar los correos usando el modelo basado en Bayes
precision, recuperacion = spam_detector.evaluate_model()  # Evaluar el modelo

# Mostrar resultados
print(f'Precisión del modelo: {precision:.4f}')
print(f'Recuperación del modelo: {recuperacion:.4f}')
