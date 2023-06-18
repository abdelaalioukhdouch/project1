# import json
# from flask import Flask, jsonify, request
# import pandas as pd
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go


# app = Flask(__name__)

# books = [
#     {
#         'id': 1,
#         'title': "Title 1",
#     },
#     {
#         'id': 2,
#         'title': "Title 2",
#     },
#     {
#         'id': 3,
#         'title': "Title 3",
#     }
# ]

# @app.route('/books/all', methods=['GET'])

# def api_all():
#     return jsonify(books)

# @app.route('/api/v1/resources/books', methods=['GET'])
# def api_id():
# #Vérifiez si un ID a été fourni dans l'URL.
# #Si un ID est fourni, assignez-le à une variable.
# #Si aucun ID n'est fourni, affichez une erreur dans le navigateur.

#  if 'id' in request.args: #'id' est présent dans les arguments de la requête(request.args)
#    id = int(request.args['id']) # le code convertit sa valeur en entier et l'assigne
#                                 #à une variable appelée id
#     #ids = int(request.args.get('id').split(','))                           
#  else:
#   return "Error: No id field provided. Please specify an id."
# # Create an empty list for our results
#  results = []
# # Parcourez les données et faites correspondre les résultats qui correspondent à l'ID
# # demandé.
# # IDs are unique, but other fields might return many results
 
#  for book in books:
#   if book['id'] == id:
#     results.append(book)
# # Use the jsonify function from Flask to convert our list of
# # Python dictionaries to the JSON format.
#  return jsonify(results)  




# import matplotlib.pyplot as plt
# import plotly.graph_objects as go



# # Diagramme à barres horizontales avec Matplotlib
# df =  pd.read_excel("database/Full Dataset.xlsx")
# def bar_h(df, X):
#     counts = df[X].value_counts()
#     plt.barh(counts.index, counts.values)
#     plt.xlabel('Nombre de vêtements')
#     plt.ylabel(f'{X}')
#     plt.title(f'Nombre de vêtements par {X}')
#     # Sauvegarder le graphique au format HTML
#     plt.savefig(f'images/{X}_barplot.png')
#     ##plt.savefig(f, format='png')

# #bar_h(df, 'Type')
# bar_h(df, 'Washing_instruction')
# bar_h(df, 'Drying_instruction')

# # Diagramme en secteurs avec Plotly
# labels = df['EI'].value_counts().index
# values = df['EI'].value_counts().values

# fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
# fig.update_layout(title='Distribution du label EI (Environmental Impact)',
#                   title_font_size=15)
# # Sauvegarder le graphique au format HTML
# fig.write_html("environmental_impact_piechart.html")


# if __name__ == '__main__':
#     #app.run( ) # permet d’exécuter l’application.
#     app.run(host='0.0.0.0', port=8001, debug=True)