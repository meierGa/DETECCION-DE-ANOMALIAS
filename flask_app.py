from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.cluster import KMeans

app = Flask(__name__)

# Variable global para almacenar los montos
montos = []

@app.route('/', methods=['GET', 'POST'])
def index():
    global montos

    if request.method == 'POST':
        try:
            if 'monto' in request.form:
                # Obtener el monto del formulario
                nuevo_monto = request.form.get('monto', type=float)
                if nuevo_monto is not None:
                    montos.append(nuevo_monto)  # Agregar el nuevo monto a la lista
            elif 'reiniciar' in request.form:
                # Reiniciar la lista de montos
                montos = []  # Vaciar la lista de montos

            # Crear un DataFrame con los montos a retirar
            df = pd.DataFrame({'Monto': montos})

            # Ajustar el modelo KMeans solo si hay datos
            if not df.empty:
                kmeans = KMeans(n_clusters=2)  # Usamos 2 clusters
                df['Cluster'] = kmeans.fit_predict(df[['Monto']])

                # Encontrar el centroide de los clusters
                centroides = kmeans.cluster_centers_

                # Definir un umbral para identificar anomalías
                umbral = 5000

                # Marcar como anomalía si el monto excede el umbral
                df['Anomalia'] = df['Monto'] > umbral

                # Graficar los resultados
                plt.figure(figsize=(10, 6))
                plt.scatter(df['Monto'], np.zeros_like(df['Monto']), c=df['Cluster'], s=100, label='Datos')
                plt.scatter(centroides[:, 0], np.zeros_like(centroides[:, 0]), c='red', s=200, label='Centroides')
                plt.axvline(x=umbral, color='orange', linestyle='--', label='Umbral de Anomalía')
                plt.title('Detección de Anomalías en Montos a Retirar')
                plt.xlabel('Montos')
                plt.yticks([])
                plt.legend()
                plt.grid()

                # Guardar la figura en un objeto BytesIO
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
            else:
                plot_url = None  # No hay datos para graficar

            # Mostrar resultados en la página
            return render_template('index.html', plot_url=plot_url, df=df.to_html(classes='data', header="true"))

        except Exception as e:
            # Mostrar el error en la consola
            print(f"Error: {e}")
            return render_template('index.html', plot_url=None, df=None)

    # Código para el método GET
    return render_template('index.html', plot_url=None, df=None)