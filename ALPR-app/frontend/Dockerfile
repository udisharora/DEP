# ==========================================
# STAGE 1: Build the React/Vite Application
# ==========================================
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package config and install dependencies
COPY package.json ./
RUN npm install

# Copy source code and build the static files
COPY . .
RUN npm run build

# ==========================================
# STAGE 2: Serve the Static Files with Nginx
# ==========================================
FROM nginx:alpine

# Copy ONLY the compiled files from the builder stage to Nginx's default public folder
# Note: Vite outputs to 'dist' by default. If using Create React App, change 'dist' to 'build'
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port 80 for the internal Docker network
EXPOSE 80

# Start Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]