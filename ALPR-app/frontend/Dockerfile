# ==========================================
# STAGE 1: Serve Pre-Built Static Files
# ==========================================
# The React/Vite app is built locally with `npm run build` (or `npx vite build`)
# which outputs to the `dist/` folder. We simply copy that into nginx.
FROM nginx:alpine

COPY dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]