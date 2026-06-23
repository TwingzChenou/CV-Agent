import os
import sys
import logging
import datetime
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Setup path and load environment
current_file = Path(__file__).resolve()
backend_root = current_file.parent.parent.parent
sys.path.append(str(backend_root))

load_dotenv()
logger = logging.getLogger(__name__)

CACHE_DISPLAY_NAME = "cv_agent_context_cache"
MODEL_NAME = "gemini-2.5-flash"

def compile_context() -> str:
    """
    Compiles all the static data (CV, Profile, GitHub repo lists, READMEs)
    into a single structured text string.
    """
    # 1. CV (PDF to markdown)
    import pymupdf4llm
    from app.engine.loader import load_documents
    cv_filepath = load_documents("Quentin_Forget_CV.pdf")
    try:
        cv_text = pymupdf4llm.to_markdown(str(cv_filepath)).strip()
    except Exception as e:
        logger.error(f"Error reading CV PDF: {e}")
        cv_text = "CV indisponible."
        
    # 2. Profil MD
    profil_filepath = load_documents("profil_quentin.md")
    try:
        with open(profil_filepath, "r", encoding="utf-8") as f:
            profil_text = f.read().strip()
    except Exception as e:
        logger.error(f"Error reading profil.md: {e}")
        profil_text = "Profil complémentaire indisponible."

    # 3. GitHub repositories and READMEs
    from app.engine.tools import list_github_projects, get_github_activity
    try:
        github_projects = list_github_projects()
    except Exception as e:
        logger.error(f"Error listing github projects: {e}")
        github_projects = "Aucun projet public trouvé."
        
    readmes = ""
    for repo in ["CV-Agent", "Momentum_AI"]:
        try:
            readmes += f"\n\n--- README: {repo} ---\n" + get_github_activity(repo)
        except Exception as e:
            logger.error(f"Error reading README for {repo}: {e}")

    # Compile the final structured text
    context_text = f"""
Voici les documents de référence concernant Quentin Forget (Monsieur Forget) :

=== CV PROFESSIONNEL ===
{cv_text}

=== PROFIL COMPLÉMENTAIRE ===
{profil_text}

=== LISTE DES PROJETS GITHUB ===
{github_projects}

=== CONTENUS DES READMES DE SES PROJETS ===
{readmes}
"""
    return context_text.strip()

def get_or_create_cv_cache(client: genai.Client) -> str:
    """
    Checks if a valid cache for the CV already exists. If yes, extends its TTL.
    Otherwise, compiles the context and creates a new cache.
    Returns the cache resource name (e.g., 'cachedContents/...').
    """
    logger.info("Checking for existing Gemini context caches...")
    try:
        caches = list(client.caches.list())
    except Exception as e:
        logger.error(f"Error listing Gemini caches: {e}")
        caches = []
        
    for c in caches:
        if c.display_name == CACHE_DISPLAY_NAME and MODEL_NAME in c.model:
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            if c.expire_time > now_utc:
                try:
                    logger.info(f"Reusing existing cache: {c.name}. Extending TTL...")
                    client.caches.update(
                        name=c.name,
                        config=types.UpdateCachedContentConfig(ttl="3600s")
                    )
                    return c.name
                except Exception as e:
                    logger.error(f"Failed to update cache TTL: {e}. Recreating...")
                    
    # Cache doesn't exist or is invalid, create a new one
    logger.info("Compiling context data for new cache...")
    context_text = compile_context()
    
    # Import SYSTEM_PROMPT late to avoid circular dependencies
    from app.engine.generate import SYSTEM_PROMPT
    
    logger.info("Creating new context cache in Gemini...")
    try:
        cache = client.caches.create(
            model=MODEL_NAME,
            config=types.CreateCachedContentConfig(
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=context_text)]
                    )
                ],
                system_instruction=SYSTEM_PROMPT.strip(),
                display_name=CACHE_DISPLAY_NAME,
                ttl="3600s"
            )
        )
        logger.info(f"Successfully created context cache: {cache.name} (Expires: {cache.expire_time})")
        return cache.name
    except Exception as e:
        logger.error(f"Failed to create context cache: {e}")
        raise e
