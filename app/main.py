# Point d'entrée API (initialisation de FastApi, inclusion des routes)

#Importation des bibliothèques
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


# Création de l'application FastAPI
app = FastAPI()

# Configuration de Jinja2 pour les templates
templates = Jinja2Templates(directory="app/templates")  # Modifie ici pour indiquer le bon chemin

# Mount pour servir les fichiers statiques depuis le dossier "static"
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Route principale qui renvoie une page HTML avec Jinja2
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "app_name": "4B INDUSTRY"})

@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "app_name": "4B INDUSTRY"})

# Route des mentions légales
@app.get("/mentions-legales", response_class=HTMLResponse)
async def mentions_legales(request: Request):
    return templates.TemplateResponse("mention.html", {"request": request, "app_name": "4B INDUSTRY"})

@app.get("/error")
async def error_page():
    return HTMLResponse(content=open("erreur.html").read(), status_code=404)

# Gestion des erreurs 404
@app.exception_handler(404)
async def not_found(request: Request, exc: Exception):
    return templates.TemplateResponse("erreur.html", {"request": request, "app_name": "4B INDUSTRY"})

# Route pour afficher la page d'affichage avec le pseudo et les menus
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "app_name": "4B INDUSTRY", "pseudo": "Utilisateur"})

# Route pour afficher la page de prédiction avec le pseudo et les menus
@app.get("/prediction", response_class=HTMLResponse)
async def prediction(request: Request):
    return templates.TemplateResponse("prediction.html", {"request": request, "app_name": "4B INDUSTRY", "pseudo": "Utilisateur"})

# Route pour la page des résultats
@app.get("/resultat", response_class=HTMLResponse)
async def resultats(request: Request):
    return templates.TemplateResponse("resultat.html", {"request": request, "app_name": "4B INDUSTRY", "pseudo": "Utilisateur"})

# Lancer l'application si ce fichier est exécuté directement
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
