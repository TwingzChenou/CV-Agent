import pytest
from app.engine.generate import parse_intent, estimate_tokens, calculate_cost

def test_parse_intent():
    """Teste le parsing d'intentions avec ou sans paramètres."""
    # Sans arguments
    assert parse_intent("chitchat") == ("chitchat", None)
    assert parse_intent("list_all_projects") == ("list_all_projects", None)
    
    # Avec arguments
    assert parse_intent("cv_query_engine(contacts)") == ("cv_query_engine", "contacts")
    assert parse_intent("read_project_readme(CV-Agent)") == ("read_project_readme", "CV-Agent")
    
    # Cas limites
    assert parse_intent("read_project_readme()") == ("read_project_readme", None)
    assert parse_intent("read_project_readme(   )") == ("read_project_readme", None)

def test_estimate_tokens():
    """Teste l'estimation du nombre de tokens."""
    # Cas vides
    assert estimate_tokens(None) == 0
    assert estimate_tokens("") == 0
    
    # Chaînes courtes et moyennes
    assert estimate_tokens("hello") == 1       # max(1, 5 // 4)
    assert estimate_tokens("hello world") == 2 # max(1, 11 // 4)
    assert estimate_tokens("a" * 100) == 25

def test_calculate_cost():
    """Teste le calcul des coûts en USD sur la base de la grille de tarifs Gemini."""
    # Tarifs par million : prompt = 0.30$, completion = 2.50$, embedding = 0.15$
    cost = calculate_cost(1000, 100, 10000)
    expected_cost = (1000 * 0.30 + 100 * 2.50 + 10000 * 0.15) / 1_000_000
    assert cost == expected_cost
