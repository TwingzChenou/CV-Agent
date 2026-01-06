from github import Github
import os

def get_github_activity(owner: str = "TwingzChenou", repo: str = None) -> str:
    """
    Récupère INSTANTANÉMENT le README d'un dépôt spécifique via l'API directe.
    Plus de scan de dossiers, plus de lenteur.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return "Erreur: Token manquant."
    
    if not repo:
        return "Erreur: Nom du repo manquant."

    try:
        # 1. Connexion
        g = Github(token)
        
        # 2. Ciblage direct du repo
        # L'API ne cherche pas, elle va directement à l'adresse
        repo_obj = g.get_repo(f"{owner}/{repo}")
        
        # 3. Demande spécifique du README
        # C'est une fonction spéciale de l'API GitHub qui trouve le README 
        # peu importe son nom exact (README.md, readme.txt, etc.)
        readme = repo_obj.get_readme()

        print(readme)
        
        # 4. Décodage (Le contenu arrive encodé, il faut le traduire en texte)
        content = readme.decoded_content.decode("utf-8")
        
        print(f"✅ README récupéré pour {repo} (Taille: {len(content)} caractères)")
        
        return f"README Content for {repo}:\n{content[:5000]}"

    except Exception as e:
        # Gestion des erreurs (ex: repo introuvable ou pas de README)
        print(f"❌ Erreur : {e}")
        return f"Impossible de lire le README pour {repo}. Vérifiez que le dépôt existe et est public."

# --- ZONE DE TEST ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test avec ton repo
    print(get_github_activity(repo="ai-cv"))
