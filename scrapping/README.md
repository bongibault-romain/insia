## Indéxage des pages

- Commencer par une liste des sites web (par exemple le site de l'insa)

- Lancer un algorithme qui va chercher tout les liens dans cette page et les mettre dans une liste

- On répéte ça sur tous les liens trouvés.
Ensuite il suffit de récupérer le contenu de chacunes des pages et le mettre en forme. On peut utiliser les spécificités du langage HTML (récuprer les \<articles\>, \<p\>, etc.)

**Limitations**: 

- Il est impossible de trouver des liens dans des pages utilisant réact ou d'autres framework dont le rendu se fait dynamiquement.

- Certaines pages sont protégés par des systèmes d'authentification et seront donc inaccessibles.