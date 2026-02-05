# Utilise une image Node stable
FROM node:20-alpine

WORKDIR /app

# Copie les fichiers de dépendances
COPY package*.json ./

# Installe les dépendances
RUN npm install

# Copie le reste du code
COPY . .

# Build l'application Next.js (si tu es en production)
# RUN npm run build

# Expose le port 3000
EXPOSE 3000

# Commande de lancement (mode dev pour tes tests)
CMD ["npm", "run", "dev"]