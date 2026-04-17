import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import spacy
import google.generativeai as genai
from datetime import datetime

from bert_scorer import BertScorer

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Kickstarter Blurb Optimizer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model artifacts on startup ──────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data") + "/"

nlp = spacy.load("en_core_web_md")
sentiment_analyzer = SentimentIntensityAnalyzer()

# Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your environment (see .env.example).")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Load feature lists
with open(f"{DATA_PATH}features_scale.txt") as f:
    features_scale = [line.strip() for line in f.readlines()]
with open(f"{DATA_PATH}features_no_scale.txt") as f:
    features_no_scale = [line.strip() for line in f.readlines()]

# Load joint BERT model from HuggingFace Hub (Filipe's best: ROC-AUC 0.857)
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "RoseCymbler/kickstarter-joint-bert")
print(f"Loading joint BERT model from {HF_MODEL_REPO}...")
bert_scorer = BertScorer(
    hf_repo_id=HF_MODEL_REPO,
    train_csv_path=f"{DATA_PATH}train.csv",
    features_scale=features_scale,
    features_no_scale=features_no_scale,
)
print("BERT model ready.")

# ── In-memory leaderboard ────────────────────────────────────────────────────
leaderboard = []

# ── Schemas ──────────────────────────────────────────────────────────────────
class ProjectInput(BaseModel):
    project_name: str
    blurb: str
    category: str
    country: str
    goal_usd: float
    duration: int

class SubmitInput(BaseModel):
    project_name: str
    blurb: str
    category: str
    country: str
    goal_usd: float
    duration: int
    original_score: float
    rewritten_blurb: str
    rewritten_score: float

# ── Helper functions ─────────────────────────────────────────────────────────
CATEGORIES = [
    "Art", "Comics", "Crafts", "Dance", "Design", "Fashion",
    "Film & Video", "Food", "Games", "Journalism", "Music",
    "Photography", "Publishing", "Technology", "Theater"
]
COUNTRIES = ["AU", "CA", "GB", "NZ", "US"]

def extract_features(name: str, blurb: str) -> dict:
    blurb_str = str(blurb)
    name_str = str(name)
    doc_name = nlp(name_str)
    doc_blurb = nlp(blurb_str)
    return {
        "blurb_length": len(blurb_str.split()),
        "sentiment_score": sentiment_analyzer.polarity_scores(blurb_str)["compound"],
        "readability_score": textstat.flesch_reading_ease(blurb_str),
        "name_blurb_similarity": doc_name.similarity(doc_blurb),
    }

def score_blurb(name, blurb, category, country, goal_usd, duration):
    features = extract_features(name, blurb)
    prob = bert_scorer.score(
        name=name,
        blurb=blurb,
        category=category,
        country=country,
        goal_usd=goal_usd,
        duration=duration,
        blurb_length=features["blurb_length"],
        sentiment_score=features["sentiment_score"],
        readability_score=features["readability_score"],
        name_blurb_similarity=features["name_blurb_similarity"],
    )
    return prob, features

def diagnose(features: dict):
    strengths, recs = [], []

    sent = features["sentiment_score"]
    if sent < 0.1:
        recs.append("Use more positive, enthusiastic language.")
    else:
        strengths.append(f"Positive sentiment ({sent:.2f})")

    read = features["readability_score"]
    if read < 30:
        recs.append("Simplify your language — shorter sentences, common words.")
    elif read > 80:
        recs.append("Add slightly more sophisticated vocabulary.")
    else:
        strengths.append(f"Good readability ({read:.0f})")

    length = features["blurb_length"]
    if length < 15:
        recs.append("Too short — expand to 20-30 words with more detail.")
    elif length > 35:
        recs.append("Too long — tighten to 20-30 words.")
    else:
        strengths.append(f"Good length ({length} words)")

    sim = features["name_blurb_similarity"]
    if sim < 0.5:
        recs.append("Make the connection between your blurb and project name stronger.")
    else:
        strengths.append(f"Good name-blurb coherence ({sim:.2f})")

    return strengths, recs

REWRITE_PROMPT = """You are an expert crowdfunding copywriter. Rewrite the blurb below to maximize the campaign's chance of getting funded.

Follow these evidence-based rules (from analysis of 110,000+ Kickstarter campaigns):

1. SENTIMENT: Use positive, enthusiastic language, warm tone but stay natural. Target a VADER compound score between 0.3 and 0.5. Avoid sounding overly promotional or desperate.
2. READABILITY: Use simple, everyday words. Short sentences. Target a Flesch Reading Ease score above 60. Avoid marketing jargon, complex vocabulary, and fancy adjectives.
3. CALL-TO-ACTION: Include 2-3 action verbs: help, join, support, discover, create, build, bring, launch, share, explore.
4. LENGTH: Keep between 20-30 words. Use the simplest phrasing possible. Every word must earn its place.
5. COHERENCE: The blurb must clearly relate to the project name.
6. SPECIFICITY: Be concrete about what the project delivers.
7. EMOTIONAL APPEAL: Convey passion without desperation. "Help us bring X to life" > "We need money for X".
8. SIMPLICITY: Write like you're explaining the project to a friend, not writing an ad.

KEYWORD OPTIMIZATION: Based on the project Category, prioritize the "Top 10 Success Words" and strictly avoid the "Top 10 Failure Words" listed in the reference data below:
- Film & Video: Success (documentary, webseries, stretch, thesis, short, web, complete, blu, refugee, misadventures) | Failure (screenplay, educate, based, want, television, entertainment, tv, anime, experiment, destiny)
- Publishing: Success (children, dragon, magical, featuring, kids, holmes, picture, memoir, hardcover, bestselling) | Failure (poetic, knowledge, podcast, poetry, current, com, sports, magazine, adult, creating)
- Art: Success (coloring, residency, mural, pins, 2013, deck, enamel, book, public, dragons) | Failure (youtube, peace, wood, opportunities, canvas, express, com, want, difference, game)
- Music: Success (folk, heading, rock, collaborative, new, returns, nashville, stretch, bluegrass, hands) | Failure (edm, latin, artist, gospel, equipment, rap, hiphop, im, electronic, showcasing)
- Technology: Success (clock, arduino, auto, raspberry, compact, compatible, waterproof, mac, pen, portable) | Failure (creating, products, app, website, site, platform, owners, application, members, able)

Here are 2 examples of what a blurb in Film & Video category with a high probability of success looks like:
1. "A post-apocalyptic love story with all practical FX. Winner of two Best Feature awards. Get your name in the credits and a signed copy!"
2. "A narrative film following a woman's journey from her home in San Francisco into her family's past at the shores of the Salton Sea."

Here are 2 examples of what a blurb in Film & Video category with a high probability of failure looks like:
1. "This cartoon is about the main character 'PG' Porky/Glenn who tells stories about his life experiences as a child and adult."
2. "the story of in a movie of scam and pozie dealings of a man who was to think that america roads were paved with gold"

Here are 2 examples of what a blurb in Publishing category with a high probability of success looks like:
1. "A zine focused on the stories of magical girls (or boys, etc) fighting against social issues in our dystopian world."
2. "A group of diverse creators bring quests for social change to the magical girl genre and resources to help you join the fight."

Here are 2 examples of what a blurb in Publishing category with a high probability of failure looks like:
1. "Wanting to finish Publishing the last 3 of the book series. Also to have it adapted for Tv mini series."
2. "We aim to make high quality independent music accessible to everyone. Help support emerging artists by giving them a place to be heard"

Here are 2 examples of what a blurb in Art category with a high probability of success looks like:
1. "Time for Art builds into a series of pieces influenced by you, the backers. You control content & duration. This month: cityscape"
2. "Time for Art will build into a series of pieces influenced by you, the backers. You control content & duration. This month: parkland"

Here are 2 examples of what a blurb in Art category with a high probability of failure looks like:
1. "Opening an art Studio with today's technology and current art students for tutoring to every person to access for a small fee."
2. "My name is Mitchell and I make thick or thin framed canvases. I am looking for help to get equipment of my own such as a printer."

Here are 2 examples of what a blurb in Music category with a high probability of success looks like:
1. "Noveria plans to be recording our 2nd album, 'In Silence', this fall. We need money, to do that. Any help would be greatly appreciated."
2. "I've been working with a group of film makers in New York to create a short film for my single 'Know Your Worth'."

Here are 2 examples of what a blurb in Music category with a high probability of failure looks like:
1. "I am creating an internet radio station that caters to those who love and crave the music from the late 80's through mid 2000's."
2. "I wanted to hold a festival with the vendors and day time fun of a off road overlanding expo, with the amazing live rock/edm music."

Here are 2 examples of what a blurb in Technology category with a high probability of success looks like:
1. "Award-Winning Audio Design Experts Voix are back with their latest product. The amazing mi8| Retro Duo Wireless Stereo Sound System."
2. "The perfect holiday gift for your friend that has suddenly become fascinated with flat earth cartography. Or just get one for yourself."

Here are 2 examples of what a blurb in Technology category with a high probability of failure looks like:
1. "I want this city to advance the future in an old world city. and to breathe new life into it."
2. "It is a smart car diagnostic station for each car owner who wants to check his car before going to Dealer garage for repairs."

Output ONLY the rewritten blurb. No explanations, no quotes.

Project Name: {name}
Category: {category}
Original Blurb: {blurb}

Rewritten Blurb:"""

def rewrite_blurb(name, blurb, category):
    prompt = REWRITE_PROMPT.format(name=name, blurb=blurb, category=category)
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip().strip('"').strip("'")
    except Exception as e:
        return f"Error: {e}"

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/score")
def score_endpoint(project: ProjectInput):
    prob, features = score_blurb(
        project.project_name, project.blurb, project.category,
        project.country, project.goal_usd, project.duration
    )
    strengths, recs = diagnose(features)
    return {
        "probability": prob,
        "features": features,
        "strengths": strengths,
        "recommendations": recs,
        "similar_projects": [],
    }

@app.post("/rewrite")
def rewrite_endpoint(project: ProjectInput):
    # Score original
    prob_orig, features_orig = score_blurb(
        project.project_name, project.blurb, project.category,
        project.country, project.goal_usd, project.duration
    )
    # Rewrite
    rewritten = rewrite_blurb(project.project_name, project.blurb, project.category)
    # Score rewritten
    prob_rewr, features_rewr = score_blurb(
        project.project_name, rewritten, project.category,
        project.country, project.goal_usd, project.duration
    )
    return {
        "original_blurb": project.blurb,
        "rewritten_blurb": rewritten,
        "original_probability": prob_orig,
        "rewritten_probability": prob_rewr,
        "lift": prob_rewr - prob_orig,
        "original_features": features_orig,
        "rewritten_features": features_rewr,
    }

@app.post("/submit")
def submit_endpoint(entry: SubmitInput):
    leaderboard.append({
        "project_name": entry.project_name,
        "blurb": entry.blurb,
        "category": entry.category,
        "country": entry.country,
        "goal_usd": entry.goal_usd,
        "original_score": entry.original_score,
        "rewritten_blurb": entry.rewritten_blurb,
        "rewritten_score": entry.rewritten_score,
        "submitted_at": datetime.now().isoformat(),
    })
    return {"status": "ok", "rank": _get_rank(entry.rewritten_score)}

@app.get("/leaderboard")
def leaderboard_endpoint():
    sorted_lb = sorted(leaderboard, key=lambda x: x["rewritten_score"], reverse=True)
    return {"leaderboard": sorted_lb[:10], "total": len(leaderboard)}

@app.post("/leaderboard/reset")
def reset_leaderboard():
    leaderboard.clear()
    return {"status": "cleared"}

def _get_rank(score):
    sorted_scores = sorted([e["rewritten_score"] for e in leaderboard], reverse=True)
    return sorted_scores.index(score) + 1 if score in sorted_scores else len(sorted_scores) + 1
