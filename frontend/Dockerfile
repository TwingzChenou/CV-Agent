# Utilise une image Node stable
FROM node:20-alpine

WORKDIR /app

# Copie les fichiers de dépendances
COPY package*.json ./

# Installe les dépendances
RUN npm install

# Copie le reste du code
COPY . .

# Attribuer la propriété de tous les fichiers du dossier /app à l'utilisateur 'node'
RUN chown -R node:node /app

# Basculer sur l'utilisateur non-root par défaut 'node'
USER node

# Expose le port 3000
EXPOSE 3000

# Commande de lancement (mode dev pour tes tests)
CMD ["npm", "run", "dev"]