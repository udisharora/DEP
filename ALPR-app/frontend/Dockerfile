# ==========================================
# STAGE 1: Build the React Application
# ==========================================
FROM node:18-alpine AS build

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Build the app
COPY . .
RUN npm run build

# ==========================================
# STAGE 2: Serve Pre-Built Static Files
# ==========================================
FROM nginx:alpine

# Copy the built output from STAGE 1 into Nginx
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]