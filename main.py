import json
from datetime import datetime, timezone
import re
import pickle
import os
import requests
from bs4 import BeautifulSoup
import time
import numpy as np
import torch
import sys  # Import sys for sys.exit()
import pymupdf # bruh i cant belive it took me an hour to realize i forgot to import it

# --- CadQuery Imports (NEW) ---
import cadquery as cq  # Main CadQuery library
# No direct imports for box, difference, union from trimesh.creation or trimesh.boolean needed now

# --- NLTK Data Downloads (for WordNetLemmatizer) ---
import nltk

print("--- Just a moment, checking my language tools... ---")
try:
    nltk.data.find('corpora/wordnet')
    print("WordNet: All good to go!")
except Exception as e:
    print(f"Oops, looks like I need to grab WordNet. No worries, I'll download it now. ({e})")
    nltk.download('wordnet')
    print("WordNet: Download complete!")

from nltk.stem import WordNetLemmatizer

print("--- My tools are ready! ---")

# --- For Gemini API Integration ---
from google import genai as genai
from google.genai import errors as genai_errors
from dotenv import load_dotenv

load_dotenv()

# --- Transformer Model for Semantic Understanding ---
from sentence_transformers import SentenceTransformer, util

# --- Global Constants ---
GEMINI_MODEL_NAME = "gemini-3.5-flash"
MODEL_DIR = 'chatbot_model'
CHAT_MEMORY_PATH = os.path.join(MODEL_DIR, 'chat_memory.pkl')
LEARNED_QA_FILE = os.path.join(MODEL_DIR, "learned_qa.jsonl")
PRE_LEARNED_FACTS_FILE = os.path.join(MODEL_DIR, "pre_learned_facts.txt")
QA_EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "qa_embeddings.pkl")

# Semantic search threshold for local understanding
SEMANTIC_CONFIDENCE_THRESHOLD = 0.75

# Gemini API Key and Model Name
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Debugging API Key loading and handling ---
if not GEMINI_API_KEY:
    print("\nUh oh! I can't find your GEMINI_API_KEY. Please make sure it's set in your .env file.")
    print("Example: GEMINI_API_KEY='YOUR_API_KEY_HERE'")
    print("Stopping execution.")
    sys.exit(1)  # Using sys.exit() for a clean and explicit exit
else:
    print(f"DEBUG: GEMINI_API_KEY loaded successfully (starts with '{GEMINI_API_KEY[:5]}...').")

# Global variable for the Gemini model name, will be set after checking availability

try:
    client = genai.Client()
except Exception as e:
    print(
        f"\nHmm, something's not quite right with connecting to Gemini API. Please double-check your API key and internet connection. ({e})")
    sys.exit(1)  # Using sys.exit() for a clean and explicit exit

# --- Global variables for loaded data ---
qa_database = {}
qa_corpus_for_embedding = []
qa_embeddings = None

chat_memory = {}
lemmatizer = WordNetLemmatizer()
sentence_transformer_model = None

gemini_chat_session = None

# --- Global for PDF-specific knowledge for the current session ---
current_pdf_chunks_data = None
PDF_CHUNK_SIZE = 500
PDF_CHUNK_OVERLAP = 50
PDF_RETRIEVAL_TOP_K = 3
PDF_RETRIEVAL_THRESHOLD = 0.6

# --- NEW: Global to store the path of the last generated/transformed STL file ---
last_generated_stl_file = None

# --- Data Sources (Initial Corpus) ---
initial_raw_corpus = [
    ("hi there", "hello how are you"),
    ("what's your name", "i am N.A.F.I.S "),
    ("how are you doing", "i am doing well thank you"),
    ("tell me a joke", "why did the scarecrow win an award because he was outstanding in his field"),
    ("goodbye", "see you later"),
    ("i am sad", "i am sorry to hear that"),
    ("where are you from", "i am from the internet"),
    ("can you help me", "yes i can help you"),
    ("who built you", "i was built by mr_hex_agon \n go check out his work on github!"),
    ("how old are you", "i do not have an age"),
    ("are you intelligent", "i am a program so i do not have intelligence"),
    ("what do you like to do", "i like to process information"),
    ("thank you", "you are welcome"),
    ("do you understand me", "yes i understand you"),
    ("what are you capable of", "i can assist with various queries and provide information"),
    ("i need help with python", "i can provide general information about python"),
    ("how was your day", "i do not have days as i am a program"),
    ("tell me about yourself", "i am a helpful artificial intelligence assistant"),
    ("what can you do", "i can assist with various queries and provide information"),
]


# --- Helper for text normalization ---
def normalize_text_for_lookup(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# -- i added new functions for remebering leared ddata better as old one got currupted

def append_learned_qa(path, question, answer, source):
    """Append one Q&A record as a single JSON line."""
    record = {
        "q": question,
        "a": answer,
        "source": source,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_learned_qa(path):
    """Return a list of (question, answer) tuples. Bad lines are skipped, not fatal."""
    pairs = []
    if not os.path.exists(path):
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pairs.append((rec["q"], rec["a"]))
            except (json.JSONDecodeError, KeyError):
                print(f"Skipping malformed line {line_no} in {path}")
    return pairs

# --- Web Scraping Function (Remains the same) ---
def scrape_qa_from_url(url):
    print(f"Alright, I'm trying to pull some Q&A from: {url}")
    scraped_pairs = []
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        qa_list_items = soup.find_all('li', class_='rank-math-list-item')

        if not qa_list_items:
            qa_list_items = soup.find_all('li')
            print(
                "No super specific 'rank-math-list-item' class found. I'll try looking at general list items (<li>) instead.")

        for item in qa_list_items:  # Fixed `for item in item` to `for item in qa_list_items`
            text = item.get_text(separator=' ', strip=True)
            match = re.match(r'^\d+\.\s*(.*?)(?:Ans|Answer):\s*(.*)$', text, re.IGNORECASE)

            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip()

                if question and answer and len(question) > 5 and len(answer) > 5:
                    question = re.sub(r'^\d+\.\s*', '', question).strip()
                    scraped_pairs.append((question, answer))
            else:
                parts = re.split(r'^\d+\.\s*([^.]+\.)\s*(.*)$', text, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 4:
                    question_part = parts[1].strip()
                    answer_part = parts[2].strip()

                    if question_part.endswith('.'):
                        question_part = question_part[:-1].strip()

                    if question_part and answer_part and len(question_part) > 5 and len(answer_part) > 5:
                        scraped_pairs.append((question_part, answer_part))

            if len(scraped_pairs) >= 200:
                print("Got a good amount of info (200 pairs)! I'll stop here for now.")
                break

        print(f"Found and grabbed {len(scraped_pairs)} potential Q&A pairs!")
        if not scraped_pairs:
            print(
                "Hmm, couldn't find any Q&A pairs with my current scraping strategy. You might need to tweak my parsing code for this website.")

    except requests.exceptions.RequestException as e:
        print(f"Oh dear, couldn't quite reach that URL. ({e})")
    except Exception as e:
        print(f"Something unexpected happened while I was trying to scrape: ({e})")
    return scraped_pairs


# --- Function to list available Gemini models (Remains the same) ---
def list_available_gemini_models():
    print("\nChatbot: Checking which Gemini models are available to me...")
    try:
        print("--- Available Gemini Models ---")
        found_models = False
        for m in client.models.list():
            if "generateContent" in (m.supported_actions or []):
                print(f"  - {m.name}")
                found_models = True
        if not found_models:
            print("  No models supporting 'generateContent' found for your API key.")
        print("---------------------------------------------------------")
        print("\nChatbot: Set GEMINI_MODEL_NAME near the top of the file to one of the names above.")
    except Exception as e:
        print(f"Chatbot: Ran into an issue trying to list models: {e}")


# --- PDF Processing Functions (Remain a copy) ---
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = pymupdf.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Chatbot: Oops, couldn't read that PDF! Error: {e}")
        return None


def chunk_text(text, chunk_size, chunk_overlap):
    chunks = []
    if not text:
        return chunks

    current_position = 0
    while current_position < len(text):
        end_position = min(current_position + chunk_size, len(text))
        chunk = text[current_position:end_position]
        chunks.append(chunk)
        if end_position == len(text):
            break
        current_position += chunk_size - chunk_overlap
    return chunks


def load_pdf_for_chat(pdf_path, model):
    global current_pdf_chunks_data
    current_pdf_chunks_data = []

    print(f"Chatbot: Okay, let's load '{os.path.basename(pdf_path)}' into my temporary memory...")

    text = extract_text_from_pdf(pdf_path)
    if text is None:
        return False

    chunks = chunk_text(text, PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP)
    if not chunks:
        print("Chatbot: The PDF was empty or I couldn't extract any meaningful text after chunking.")
        return False

    print(f"Chatbot: Extracted {len(chunks)} text chunks from the PDF. Generating embeddings now...")

    chunk_texts = [c for c in chunks]
    embeddings = model.encode(chunk_texts, convert_to_tensor=True)

    for i, chunk in enumerate(chunks):
        current_pdf_chunks_data.append({
            'text': chunk,
            'embedding': embeddings[i]
        })

    print(
        f"Chatbot: PDF content loaded! I can now answer questions based on '{os.path.basename(pdf_path)}' for this conversation.")
    return True


# --- Get Answer from Gemini API (Updated to use ChatSession) ---
def get_answer_from_gemini(query, chat_session):
    global GEMINI_MODEL_NAME

    print(
        f"\nChatbot: Hmm, let me think... checking with Gemini for: '{query}' using model '{GEMINI_MODEL_NAME}' and conversational memory...")
    try:
        response = chat_session.send_message(query)

        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        else:
            return "Gemini didn't give me a clear answer for that. Maybe try rephrasing?"
    except genai_errors.APIError as e:
        if e.code == 404:
            print(f"Chatbot: The model '{GEMINI_MODEL_NAME}' isn't found or supported.")
            list_available_gemini_models()
            return "I encountered a problem because my current Gemini model isn't available. Please try again or use the 'list_models' command."
        print(f"Chatbot: Gemini API error {e.code}: {e.message}")
        return "I encountered an unexpected problem while trying to get information from Gemini."
    except Exception as e:
        print(f"Chatbot: An unexpected error occurred while contacting Gemini: {e}")
        return "I encountered an unexpected problem while trying to get information from Gemini."


# --- Chat Memory Functions (Remains the same) ---
def load_chat_memory(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                memory = pickle.load(f)
                return memory
        except Exception as e:
            print(f"Couldn't quite load our chat history: {e}. Starting fresh!")
            return {}
    return {}


def save_chat_memory(memory, file_path):
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(memory, f)
    except Exception as e:
        print(f"Oops, couldn't save our chat history: {e}")


# --- Chatbot Initialization Function (Remains the same) ---
def initialize_chatbot():
    print("\n--- Alright, let's get things rolling! ---")
    os.makedirs(MODEL_DIR, exist_ok=True)

    global qa_database, qa_corpus_for_embedding, qa_embeddings, sentence_transformer_model, current_pdf_chunks_data

    # This is the line that was causing the NameError if SentenceTransformer wasn't truly loaded.
    if sentence_transformer_model == None:
        print(
            "Just a sec, loading up my brain (the Sentence-Transformer model). This might take a moment if it's my first time!")
        # Ensure SentenceTransformer is imported and available before this call
        sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Brain loaded! Ready for some deep thinking.")

    qa_database = {}
    qa_corpus_for_embedding = []
    current_pdf_chunks_data = None  # Ensure PDF data is reset on full chatbot re-initialization

    def add_qa_to_corpus(q, a):
        normalized_q = normalize_text_for_lookup(q)
        qa_database[normalized_q] = a
        qa_corpus_for_embedding.append(q)

    for q, a in initial_raw_corpus:
        add_qa_to_corpus(q, a)

    pre_learned_facts_count = 0
    if os.path.exists(PRE_LEARNED_FACTS_FILE):
        print(f"Pulling in some pre-learned facts from {PRE_LEARNED_FACTS_FILE}...")
        with open(PRE_LEARNED_FACTS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':::', 1)
                if len(parts) == 2:
                    add_qa_to_corpus(parts[0], parts[1])
                    pre_learned_facts_count += 1
                else:
                    print(f"Heads up: Found a slightly odd line in {PRE_LEARNED_FACTS_FILE}: {line.strip()}")
    print(f"Loaded {pre_learned_facts_count} handy pre-learned Q&A pairs.")

    learned_pairs = load_learned_qa(LEARNED_QA_FILE)
    for q, a in learned_pairs:
        add_qa_to_corpus(q, a)
    learned_data_count = len(learned_pairs)
    print(f"Recalled {learned_data_count} things learned")

    print(f"All set! My local brain now knows about {len(qa_corpus_for_embedding)} unique questions.")

    if os.path.exists(QA_EMBEDDINGS_PATH) and len(qa_corpus_for_embedding) > 0:
        try:
            with open(QA_EMBEDDINGS_PATH, 'rb') as f:
                loaded_data = pickle.load(f)
                if loaded_data['qa_corpus'] == qa_corpus_for_embedding:
                    qa_embeddings = loaded_data['embeddings']
                    print("Great! Loaded existing Q&A embeddings from file. That was fast!")
                else:
                    print("My knowledge changed a bit. Re-generating embeddings now...")
                    qa_embeddings = sentence_transformer_model.encode(qa_corpus_for_embedding, convert_to_tensor=True)
                    with open(QA_EMBEDDINGS_PATH, 'wb') as f:
                        pickle.dump({'qa_corpus': qa_corpus_for_embedding_ref, 'embeddings': qa_embeddings_ref}, f)
                    print("New Q&A embeddings generated and saved. All up-to-date!")
        except Exception as e:
            print(f"Oops, ran into a problem loading embeddings: {e}. I'll just re-generate them from scratch.")
            qa_embeddings = sentence_transformer_model.encode(qa_corpus_for_embedding, convert_to_tensor=True)
            with open(QA_EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump({'qa_corpus': qa_corpus_for_embedding, 'embeddings': qa_embeddings}, f)
            print("New Q&A embeddings generated and saved. Ready to go!")
    else:
        if len(qa_corpus_for_embedding) > 0:
            print("First time generating Q&A embeddings. This might take a bit for a large knowledge base...")
            qa_embeddings = sentence_transformer_model.encode(qa_corpus_for_embedding, convert_to_tensor=True)
            with open(QA_EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump({'qa_corpus': qa_corpus_for_embedding, 'embeddings': qa_embeddings}, f)
            print("Q&A embeddings generated and saved. We're ready for deep understanding!")
        else:
            print("No Q&A data yet, so no embeddings to generate. Let's learn some!")

    global chat_memory
    chat_memory = load_chat_memory(CHAT_MEMORY_PATH)

    print("\n--- Chatbot is fully initialized and ready to chat! ---")

    return {
        'qa_database': qa_database,
        'qa_corpus_for_embedding': qa_corpus_for_embedding,
        'qa_embeddings': qa_embeddings,
        'sentence_transformer_model': sentence_transformer_model,
        'chat_memory': chat_memory
    }


# --- Function to Process a Single Query (Core Logic with natural tone prints) ---
def process_query(user_input, chatbot_state, chat_session):
    qa_database_ref = chatbot_state['qa_database']
    qa_corpus_for_embedding_ref = chatbot_state['qa_corpus_for_embedding']
    qa_embeddings_ref = chatbot_state['qa_embeddings']
    sentence_transformer_model_ref = chatbot_state['sentence_transformer_model']
    chat_memory_ref = chatbot_state['chat_memory']

    normalized_input = normalize_text_for_lookup(user_input)
    response = ""
    source_of_answer = "None"
    confidence = 0.0

    # 1. Try Exact/Normalized String Match first (fastest)
    if normalized_input in qa_database_ref:
        response = qa_database_ref[normalized_input]
        source_of_answer = "My Local Brain (Direct Match)"
        confidence = 1.0
        print(f"Chatbot: Got it right here! (Confidence: {confidence:.2f}) -> {response}")
        return response, source_of_answer, confidence

        # 2. If no direct match, try Semantic Similarity using Transformer embeddings (Local KB)
    elif qa_embeddings_ref is not None and len(qa_corpus_for_embedding_ref) > 0:
        print("Chatbot: No exact match. Let me check my memory for similar questions in my general knowledge...")
        user_input_embedding = sentence_transformer_model_ref.encode(user_input, convert_to_tensor=True)

        cosine_scores = util.cos_sim(user_input_embedding, qa_embeddings_ref)[0]
        max_similarity = torch.max(cosine_scores).item()
        max_similarity_index = torch.argmax(cosine_scores).item()

        confidence = max_similarity

        if max_similarity >= SEMANTIC_CONFIDENCE_THRESHOLD:
            matched_question_original = qa_corpus_for_embedding_ref[max_similarity_index]
            response = qa_database_ref[normalize_text_for_lookup(matched_question_original)]
            source_of_answer = f"My Local Brain (Found a good match! Confidence: {confidence:.2f})"
            print(f"Chatbot: Ah, I think I know this one! (Confidence: {confidence:.2f}) -> {response}")
            return response, source_of_answer, confidence

            # 3. If no confident local match, check current_pdf_chunks_data (PDF RAG)
    if current_pdf_chunks_data is not None and len(current_pdf_chunks_data) > 0:
        print(f"Chatbot: Not in my main brain, checking the loaded PDF now for '{user_input}'...")
        if 'user_input_embedding' not in locals():
            user_input_embedding = sentence_transformer_model_ref.encode(user_input, convert_to_tensor=True)

        pdf_embeddings = torch.stack([d['embedding'] for d in current_pdf_chunks_data])
        cosine_scores = util.cos_sim(user_input_embedding, pdf_embeddings)[0]

        top_k_indices = torch.topk(cosine_scores, k=min(PDF_RETRIEVAL_TOP_K, len(current_pdf_chunks_data))).indices

        relevant_chunks_text = []
        best_pdf_chunk_confidence = 0.0

        for idx in top_k_indices:
            score = cosine_scores[idx].item()
            if score >= PDF_RETRIEVAL_THRESHOLD:
                relevant_chunks_text.append(current_pdf_chunks_data[idx]['text'])
                if score > best_pdf_chunk_confidence:
                    best_pdf_chunk_confidence = score

        if relevant_chunks_text:
            context_text = "\n\n".join(relevant_chunks_text)
            gemini_rag_prompt = f"Given the following information, please answer the question. If the answer is not in the provided text, clearly state that you cannot answer from the provided information. Do not make up information.\n\nContext:\n```\n{context_text}\n```\n\nQuestion: {user_input}\nAnswer:"

            print(
                f"Chatbot: Sending an augmented query to Gemini with PDF context (Best PDF chunk confidence: {best_pdf_chunk_confidence:.2f})...")
            gemini_answer_from_pdf = get_answer_from_gemini(gemini_rag_prompt, chat_session)

            if gemini_answer_from_pdf and \
                    "Gemini didn't give me a clear answer" not in gemini_answer_from_pdf and \
                    "I'm having a little trouble connecting to Gemini" not in gemini_answer_from_pdf and \
                    "I encountered a problem because my current Gemini model isn't available" not in gemini_answer_from_pdf and \
                    "not in the provided text" not in gemini_answer_from_pdf.lower() and \
                    "cannot answer from the given context" not in gemini_answer_from_pdf.lower() and \
                    "not found in the provided context" not in gemini_answer_from_pdf.lower():

                response = gemini_answer_from_pdf
                source_of_answer = f"Loaded PDF (RAG, Confidence: {best_pdf_chunk_confidence:.2f})"
                print(f"Chatbot: Got it from the PDF via Gemini! -> {response}")
                return response, source_of_answer, best_pdf_chunk_confidence
            else:
                print("Chatbot: Gemini couldn't confidently answer from the PDF context. Trying general knowledge...")

    # 4. If no confident match from local, and no answer from PDF, query Gemini broadly
    print(f"Chatbot: Falling back to general Gemini knowledge for '{user_input}'...")
    gemini_answer = get_answer_from_gemini(user_input, chat_session)

    if gemini_answer and \
            "Gemini didn't give me a clear answer" not in gemini_answer and \
            "I'm having a little trouble connecting to Gemini" not in gemini_answer and \
            "I encountered a problem because my current Gemini model isn't available" not in gemini_answer:

        response = gemini_answer
        source_of_answer = "Gemini (General Knowledge)"
        print(f"Chatbot: Got it from Gemini's broad knowledge! -> {response}")

        if normalize_text_for_lookup(user_input) not in qa_database_ref:
            qa_database_ref[normalize_text_for_lookup(user_input)] = response
            qa_corpus_for_embedding_ref.append(user_input)
            new_embedding = sentence_transformer_model_ref.encode(user_input, convert_to_tensor=True)
            if qa_embeddings_ref is not None:
                qa_embeddings_ref = torch.cat((qa_embeddings_ref, new_embedding.unsqueeze(0)), dim=0)
            else:
                qa_embeddings_ref = new_embedding.unsqueeze(0)
            chatbot_state['qa_embeddings'] = qa_embeddings_ref

            append_learned_qa(LEARNED_QA_FILE, user_input, response, source="gemini")

            with open(QA_EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump({'qa_corpus': qa_corpus_for_embedding_ref, 'embeddings': qa_embeddings_ref}, f)
            print(f"Chatbot: Coolio! I just learned something new from Gemini and saved it to my general memory!")
    else:
        response = "Gosh, I'm having a bit of trouble with that one. My local knowledge isn't strong enough, and Gemini isn't giving me a clear answer either. Could you try asking something else, or rephrasing it?"
        source_of_answer = "None (Couldn't figure it out)"

    chat_memory_ref[normalized_input] = response

    return response, source_of_answer, confidence


# --- Programmatic 3D Box Generation Function (NOW USING CADQUERY) ---

def create_custom_3d_printable_box(length, width, height, wall_thickness,
                                   text_to_imprint=None, imprint_depth=1.0,
                                   output_filename="custom_box.stl",
                                   create_lid=False, corner_radius=0.0):  # New parameter for lid creation
    """
    Generates a 3D printable hollow box with precise dimensions and optional
    imprinted (engraved or raised) text, saved as an STL file, using CadQuery.
    Optionally generates a matching lid.

    Args:
        length (float): The outer length (X-axis) of the box.
        width (float): The outer width (Y-axis) of the box.
        height (float): The outer height (Z-axis) of the box.
        wall_thickness (float): The thickness of the box walls.
        text_to_imprint (str, optional): Text to engrave/raise on the front side. Defaults to None.
        imprint_depth (float): Depth of engraving (positive for cut-in, negative for raised out).
                                Value is absolute magnitude.
        output_filename (str): The name of the STL file to save.
        create_lid (bool): If True, also generates a matching lid for the box.
        corner_radius (float): The radius for rounding the vertical corners of the box and lid.
                                If 0 or negative, no rounding is applied.

    Returns:
        str: The path to the generated STL file (box) if successful, None otherwise.
             If a lid is also created, it will be saved alongside.
    """
    try:
        # Define a small tolerance for boolean operations to avoid floating point issues
        # This tolerance makes the cutting object slightly smaller or the fitting object slightly larger
        boolean_tolerance = 0.1

        # --- 1. Create the base solid outer box, aligned with its bottom at Z=0 ---
        # We start by creating the box centered, then translate it so its bottom is at Z=0.
        outer_box_solid = cq.Workplane("XY").box(length, width, height).translate((0, 0, height / 2))

        # Apply fillet to vertical edges if radius is positive
        actual_corner_radius = corner_radius
        if actual_corner_radius > 0:
            # Max fillet radius for the outer box
            # It must be less than half of the smallest dimension minus wall thickness to avoid self-intersection
            max_fillet_radius_outer = min(length / 2, width / 2) - wall_thickness - boolean_tolerance

            if actual_corner_radius > max_fillet_radius_outer:
                print(
                    f"Warning: Requested corner radius {actual_corner_radius:.2f}mm is too large for box dimensions and wall thickness. Reducing outer box fillet to {max(0.0, max_fillet_radius_outer):.2f}mm.")
                actual_corner_radius = max(0.0, max_fillet_radius_outer)  # Ensure it's not negative

            if actual_corner_radius > 0:  # Apply fillet only if radius is positive after reduction
                outer_box_solid = outer_box_solid.edges("|Z").fillet(actual_corner_radius)

        final_box_with_text = outer_box_solid  # Start with the potential filleted solid box

        # --- 2. Add Text Imprint (if provided) ---
        if text_to_imprint:
            print(f"DEBUG: Attempting to add text '{text_to_imprint}'...")

            # Define the CadQuery Plane for the text.
            # Origin at the center of the front face of the bottom-aligned box.
            text_plane = cq.Plane(
                origin=(0, width / 2, height / 2),
                xDir=(-1, 0, 0),
                normal=(0, -1, 0)
            )

            text_workplane = cq.Workplane(text_plane)

            text_extrusion_distance = imprint_depth

            text_feature = text_workplane.text(
                txt=text_to_imprint,
                fontsize=15.0,
                distance=text_extrusion_distance,
                halign='center',
                valign='center'
            )

            # --- DEBUG: Export text feature itself (for debugging purposes) ---
            debug_text_filename = f"DEBUG_TEXT_{text_to_imprint.replace(' ', '_')}.stl"
            try:
                if isinstance(text_feature.val(), (cq.Solid, cq.Compound)):
                    text_feature.val().exportStl(debug_text_filename, tolerance=0.001)
                    print(f"DEBUG: Exported raw text feature for debugging to {os.path.abspath(debug_text_filename)}")
                else:
                    print(
                        f"DEBUG: Cannot export raw text feature (type: {type(text_feature.val())}). It's not a Solid or Compound directly.")
            except Exception as e:
                print(f"DEBUG ERROR: Could not export debug text feature: {e}")
            # --- END DEBUG ---

            if imprint_depth > 0:  # Engraving (cut into material)
                print("DEBUG: Performing CadQuery cut (engraving) with text feature...")
                final_box_with_text = outer_box_solid.cut(text_feature)
            else:  # Raising (extends out from material)
                print("DEBUG: Performing CadQuery union (raising) with text feature...")
                final_box_with_text = outer_box_solid.union(text_feature)

        # --- 3. Create the inner cut box to hollow the main box ---
        # Inner box dimensions.
        # Subtract 2*wall_thickness from X and Y for side walls.
        # Subtract 1*wall_thickness from Z for the bottom wall, making the top open.
        # Apply boolean_tolerance to make the cutting tool slightly smaller for a clean cut.
        inner_length = length - (2 * wall_thickness) - (2 * boolean_tolerance)
        inner_width = width - (2 * wall_thickness) - (2 * boolean_tolerance)
        inner_cavity_height = height - wall_thickness - boolean_tolerance  # Cavity height (open top)

        if inner_length <= 0 or inner_width <= 0 or inner_cavity_height <= 0:
            print(f"Error: Invalid dimensions or wall thickness for a hollow box. "
                  f"Inner dimensions would be: L={inner_length}, W={inner_width}, H={inner_cavity_height}")
            print("Please ensure (length, width) > 2 * wall_thickness and height > wall_thickness.")
            return None

        # Create the inner cutting box, initially centered at Z=0
        inner_cut_box_solid = cq.Workplane("XY").box(inner_length, inner_width, inner_cavity_height)

        # Position the inner cut box:
        # The bottom of the outer_box_solid (after initial translate) is at Z=0.
        # We want the bottom of the inner cavity to be at Z = wall_thickness.
        # The inner_cut_box_solid is initially centered, so its bottom is at -inner_cavity_height / 2.
        # To move its bottom to (wall_thickness + boolean_tolerance), its center needs to be moved to:
        # (wall_thickness + boolean_tolerance) + (inner_cavity_height / 2)
        z_translation_for_cut = (wall_thickness + boolean_tolerance) + (inner_cavity_height / 2)

        inner_cut_box_positioned = inner_cut_box_solid.translate((0, 0, z_translation_for_cut))

        # Perform boolean cut to hollow the box
        # This should now always result in a hollow box with an open top
        box_solid = final_box_with_text.cut(inner_cut_box_positioned)

        # --- Export the main box ---
        full_box_output_path = os.path.abspath(output_filename)
        box_solid.val().exportStl(full_box_output_path, tolerance=0.001)
        print(f"Chatbot: Successfully generated the main box: '{full_box_output_path}'")

        # --- Generate Lid if requested ---
        if create_lid:
            print("DEBUG: Creating lid for the box...")
            lid_thickness = wall_thickness  # Use same thickness as walls for the lid
            lid_lip_height = wall_thickness  # Height of the lip that goes into the box

            # Lid lip clearance: this is crucial for the lid to fit
            # Increase this value to make the lid's lip smaller for a looser fit
            lid_fit_clearance = 0.5  # Adjusted clearance for better fit

            # Main flat part of the lid (same outer dimensions as the box)
            # The lid plate also needs to be aligned with its bottom at Z=0 for consistent filleting
            lid_plate = cq.Workplane("XY").box(length, width, lid_thickness).translate((0, 0, lid_thickness / 2))

            # Apply fillet to lid plate vertical edges if corner_radius > 0
            if actual_corner_radius > 0:  # Use actual_corner_radius from box
                # Max radius for lid plate is simply min(length, width) / 2
                max_fillet_radius_lid_plate = min(length / 2, width / 2)
                lid_plate_fillet_radius = actual_corner_radius

                if lid_plate_fillet_radius > max_fillet_radius_lid_plate:
                    print(
                        f"Warning: Lid plate fillet radius {lid_plate_fillet_radius:.2f} is too large for lid plate dimensions. Reducing to {max(0.0, max_fillet_radius_lid_plate):.2f}mm.")
                    lid_plate_fillet_radius = max(0.0, max_fillet_radius_lid_plate)

                if lid_plate_fillet_radius > 0:
                    lid_plate = lid_plate.edges("|Z").fillet(lid_plate_fillet_radius)

            # Calculate dimensions for the lip, which fits inside the box opening
            # The inner dimensions of the box opening are length - 2*wall_thickness, width - 2*wall_thickness
            lip_length = length - (2 * wall_thickness) - lid_fit_clearance
            lip_width = width - (2 * wall_thickness) - lid_fit_clearance

            if lip_length <= 0 or lip_width <= 0:
                print(
                    "Warning: Lid lip dimensions would be zero or negative after clearance. Creating a flat lid without a lip.")
                lid_final = lid_plate
            else:
                # Create the lip, initially centered
                lip_solid = cq.Workplane("XY").box(lip_length, lip_width, lid_lip_height)

                # Apply fillet to the lip's vertical edges
                if actual_corner_radius > 0:
                    # Inner corner radius of the box will be original_corner_radius - wall_thickness
                    # The lip's fillet needs to be slightly smaller than that for clearance.
                    inner_corner_box_radius = max(0.0, actual_corner_radius - wall_thickness)

                    # Additional clearance for the lip's corner to fit within the box's inner corner
                    lip_corner_fit_clearance = lid_fit_clearance / 2  # Can be adjusted

                    lip_fillet_radius = max(0.0, inner_corner_box_radius - lip_corner_fit_clearance)

                    # Ensure the lip fillet radius is not too large for the lip's dimensions
                    max_fillet_radius_lip = min(lip_length / 2, lip_width / 2)
                    if lip_fillet_radius > max_fillet_radius_lip:
                        print(
                            f"Warning: Lip fillet radius {lip_fillet_radius:.2f}mm is too large for lip dimensions. Reducing to {max(0.0, max_fillet_radius_lip):.2f}mm.")
                        lip_fillet_radius = max(0.0, max_fillet_radius_lip)

                    if lip_fillet_radius > 0:  # Apply fillet only if radius is positive after reduction
                        lip_solid = lip_solid.edges("|Z").fillet(lip_fillet_radius)

                # Position the lip so its top aligns with the bottom of the lid_plate
                # lid_plate's bottom is at Z=0 if its center is at Z=lid_thickness/2
                # lip_solid's bottom is at Z=-lid_lip_height/2 if its center is at Z=0
                # We want lip's top (+lip_lip_height/2 from its center) to be at Z=0 (bottom of lid_plate)
                # So, lip's center needs to be moved to -(lid_lip_height/2)
                # Then, relative to lid_plate's center (lid_thickness/2), the lip's center should be at:
                # (lid_thickness / 2) - (lip_solid.BoundingBox().zlen / 2) - (lid_lip_height / 2)
                # Or, simpler: lid_plate center Z is lid_thickness / 2. Lip should be below it.
                # Total height of lip and plate is lid_thickness + lid_lip_height.
                # Center of the combined object would be 0 if the whole thing was one solid block.
                # Let's align the lid_plate bottom at Z=0.
                # The lip needs to go downwards from Z=0.
                # The lid_plate is at Z=0 (its bottom). So its center is lid_thickness/2.
                # The lip needs its *top* to be at Z=0.
                # So, lip's center should be at -lid_lip_height/2.
                # This translates the lip from its own center (0,0,0) to (0,0,-lid_lip_height/2)
                positioned_lip = lip_solid.translate((0, 0, -lid_lip_height / 2))

                # Union the plate and the positioned lip
                lid_final = lid_plate.union(positioned_lip)
                # After union, the combined lid's bottom will be at -(lid_lip_height), its top at (lid_thickness).
                # We want the lid to be 'flat' on the printer bed, so its bottom should be at Z=0.
                # The current lowest point is -lid_lip_height. We need to shift it up by lid_lip_height.
                lid_final = lid_final.translate((0, 0, lid_lip_height))

            lid_output_filename = os.path.splitext(output_filename)[0] + "_lid.stl"
            full_lid_output_path = os.path.abspath(lid_output_filename)
            lid_final.val().exportStl(full_lid_output_path, tolerance=0.001)
            print(f"Chatbot: Successfully generated the matching lid: '{full_lid_output_path}'")

        # --- Update last_generated_stl_file ---
        global last_generated_stl_file
        last_generated_stl_file = output_filename  # Last generated is the box itself (not the lid)

        return output_filename  # Return the name of the main box file

    except ImportError:
        print("Error: 'cadquery' library not found. Please install it using 'pip install cadquery'.")
        print("Also, ensure 'ocp' (OpenCascade Python Bindings) is installed, usually pulled by cadquery.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during custom 3D model generation with CadQuery: {e}")
        return None


# --- Function to transform an existing 3D model with optional flip and mirror ---
def transform_model(input_filename, output_filename, apply_vertical_flip=False, apply_mirror_yz=False):
    """
    Loads an existing STL model and applies optional vertical flip and/or mirroring.

    Args:
        input_filename (str): Path to the input STL file.
        output_filename (str): Path to save the transformed STL file.
        apply_vertical_flip (bool): If True, rotates the model 180 degrees around the X-axis.
        apply_mirror_yz (bool): If True, mirrors the model across the YZ plane (along X-axis).

    Returns:
        str: Path to the generated STL file if successful, None otherwise.
    """
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        return None

    try:
        print(f"DEBUG: Loading model from '{input_filename}' using importSTL(filename=...)...")
        loaded_model = cq.importers.importSTL(filename=input_filename)

        transformed_model = loaded_model

        if apply_vertical_flip:
            print("DEBUG: Performing vertical flip (rotate 180 around X-axis)...")
            transformed_model = transformed_model.rotate((0, 0, 0), (1, 0, 0), 180)

        if apply_mirror_yz:
            print("DEBUG: Performing mirror operation across YZ plane (along X-axis)...")
            transformed_model = transformed_model.mirror("YZ")

        # Save the transformed model
        full_output_path = os.path.abspath(output_filename)  # Get absolute path for consistent output
        print(f"DEBUG: Exporting transformed model to '{full_output_path}'...")
        transformed_model.val().exportStl(full_output_path, tolerance=0.001)

        print(f"Successfully transformed model saved to {full_output_path}")

        # --- Update last_generated_stl_file ---
        global last_generated_stl_file
        last_generated_stl_file = output_filename

        return output_filename

    except Exception as e:
        print(f"An unexpected error occurred during model transformation: {e}")
        print("Please ensure the input file is a valid STL file.")
        return None


# --- Main Interactive Chat Loop ---
def chat():
    print("\nHey there! Welcome to our chat. Type 'quit' to say goodbye.")
    print("Type 'scrape' if you want me to learn from a website.")
    print("Type 'rebuild_embeddings' if you added new facts manually and want me to fully re-process my brain.")
    print("Type 'list_models' to see which Gemini models are available.")
    print("Type 'new_topic' to clear the current conversation history and any loaded PDF.")
    print("Type 'load_pdf <filepath>' to load a PDF file for me to learn from for this chat.")
    print(
        "Type 'make_box <dimensions> [with-lid] [rounded-corners <radius>]' to create a custom 3D printable box (e.g., 'make_box 100x50x40 2mm walls engrave Hello with-lid rounded-corners 5mm').")
    print(
        "Type 'transform_model [input_filename] [output_filename] [vertical-flip|flip_vertical] [mirror-yz|mirror_horizontal]' to transform an existing STL model.")  # Updated help message for new command

    chatbot_state = initialize_chatbot()

    global gemini_chat_session, current_pdf_chunks_data, last_generated_stl_file

    try:
        gemini_chat_session = client.chats.create(model=GEMINI_MODEL_NAME)
        print(f"Chatbot: I've started a new conversation session with Gemini using model '{GEMINI_MODEL_NAME}'.")
    except Exception as e:
        print(f"Chatbot: Failed to start Gemini chat session with model '{GEMINI_MODEL_NAME}': {e}")
        print("Chatbot: Please check your API key and try listing models ('list_models') to find a working one.")
        gemini_chat_session = None

    while True:
        try:
            user_input = input("\nYou: ")
        except UnicodeDecodeError:
            print(
                "\nChatbot: I'm having trouble understanding your input due to an encoding issue. Please ensure your terminal's encoding is set to UTF-8 or try pasting plain text.")
            continue  # Skip processing this input and prompt again

        if user_input.lower() == 'quit':
            save_chat_memory(chatbot_state['chat_memory'], CHAT_MEMORY_PATH)
            print("Chatbot: Alright, see ya later! Thanks for chatting!")
            break
        elif user_input.lower() == 'scrape':
            website_url = input(
                "Chatbot: Okay, what's the URL you'd like me to scrape Q&A from? (e.g., https://www.example.com/faq): ")
            new_qa_pairs = scrape_qa_from_url(website_url)
            if new_qa_pairs:
                for q, a in new_qa_pairs:
                    append_learned_qa(LEARNED_QA_FILE, q, a, source="scrape")
                print(
                    f"\nChatbot: Done scraping! I've added {len(new_qa_pairs)} new Q&A pairs to my {LEARNED_QA_FILE} file.")
                print(
                    "Chatbot: **Just a heads up: To make sure I understand these new facts semantically, it's a good idea to type 'rebuild_embeddings' next!**")
            else:
                print("Chatbot: Didn't find anything new to learn from that URL, or something went wrong.")
            continue
        elif user_input.lower() == 'rebuild_embeddings':
            print(
                "\nChatbot: Okay, let's give my brain a full refresh! I'll re-process all my knowledge for better understanding...")
            chatbot_state = initialize_chatbot()
            print("Chatbot: All refreshed and ready for deeper understanding!")
            continue
        elif user_input.lower() == 'list_models':
            list_available_gemini_models()
            continue
        elif user_input.lower() == 'new_topic':
            if gemini_chat_session:
                try:
                    gemini_chat_session = client.chats.create(model=GEMINI_MODEL_NAME)
                    current_pdf_chunks_data = None
                    last_generated_stl_file = None  # Clear last generated file on new topic
                    print("Chatbot: Okay, I've cleared our conversation history and any loaded PDF. Let's talk about something new!")
                except Exception as e:
                    print(f"Chatbot: Couldn't start a new chat session to clear context: {e}")
            else:
                print("Chatbot: No active Gemini session to clear anyway.")
            continue
        # --- Handle Programmatic 3D Box Generation Command ---
        elif user_input.lower().startswith('make_box'):
            # Extract the part of the command after 'make_box'
            box_command = user_input.lower().split(' ', 1)
            if len(box_command) < 2:
                print(
                    "Chatbot: Please provide dimensions and details for the box. E.g., 'make_box 100x50x40 2mm walls engrave Hello'.")
                continue

            # This logic is adapted from the previous `process_user_request` for the box
            box_details = box_command[1].strip()

            create_lid = False
            if 'with-lid' in box_details:
                create_lid = True
                box_details = box_details.replace('with-lid',
                                                  '').strip()  # Remove the flag so it doesn't interfere with other parsing

            # Parsing dimensions
            dims = re.findall(r'(\d+\.?\d*)\s*(mm|cm|inch)?', box_details)
            dimensions = []
            for val, unit in dims:
                num = float(val)
                if unit == "cm":
                    num *= 10  # Convert to mm
                elif unit == "inch":
                    num *= 25.4  # Convert to mm
                dimensions.append(num)

            length, width, height = None, None, None
            if len(dimensions) >= 3:
                length, width, height = dimensions[0], dimensions[1], dimensions[2]
            else:
                len_match = re.search(r'(\d+\.?\d*)\s*(?:mm|cm|inch)?\s*long', box_details)
                wid_match = re.search(r'(\d+\.?\d*)\s*(?:mm|cm|inch)?\s*wide', box_details)
                hei_match = re.search(r'(\d+\.?\d*)\s*(?:mm|cm|inch)?\s*tall', box_details)

                if len_match: length = float(len_match.group(1)) * (
                    10 if "cm" in len_match.group(0) else (25.4 if "inch" in len_match.group(0) else 1))
                if wid_match: width = float(wid_match.group(1)) * (
                    10 if "cm" in wid_match.group(0) else (25.4 if "inch" in wid_match.group(0) else 1))
                if hei_match: height = float(hei_match.group(1)) * (
                    10 if "cm" in hei_match.group(0) else (25.4 if "inch" in hei_match.group(0) else 1))

            # Wall thickness
            wall_match = re.search(r'(\d+\.?\d*)\s*(?:mm|cm|inch)?\s*wall(?:s| thickness)?', box_details)
            wall_thickness = 2.0  # Default
            if wall_match:
                wall_val = float(wall_match.group(1))
                wall_unit_match = re.search(r'(mm|cm|inch)', wall_match.group(0))
                wall_unit = wall_unit_match.group(1) if wall_unit_match else None
                if wall_unit == "cm":
                    wall_val *= 10
                elif wall_unit == "inch":
                    wall_val *= 25.4
                wall_thickness = wall_val

            # Text imprint
            text_imprint_match = re.search(r'(?:engrave|imprint|raised|put text)\s*[\'"]?([a-zA-Z0-9\s]+)[\'"]?',
                                           box_details)
            text_to_imprint = text_imprint_match.group(1).strip() if text_imprint_match else None

            # Imprint depth
            imprint_depth = 1.0  # Default
            imprint_depth_match_engrave = re.search(r'(\d+\.?\d*)\s*(?:mm|cm|inch)?\s*deep', box_details)
            imprint_depth_match_raised = re.search(r'sticks out\s*(\d+\.?\d*)\s*(?:mm|cm|inch)?', box_details)

            if imprint_depth_match_engrave:
                imprint_depth = float(imprint_depth_match_engrave.group(1))
            elif imprint_depth_match_raised:
                imprint_depth = -float(imprint_depth_match_raised.group(1))

            # Corner radius
            corner_radius = 0.0  # Default to no rounding
            radius_match = re.search(r'(?:radius|rounded-corners)\s*(\d+\.?\d*)\s*(?:mm|cm|inch)?', box_details)
            if radius_match:
                radius_val = float(radius_match.group(1))
                radius_unit_match = re.search(r'(mm|cm|inch)', radius_match.group(0))
                radius_unit = radius_unit_match.group(1) if radius_unit_match else None
                if radius_unit == "cm":
                    radius_val *= 10
                elif radius_unit == "inch":
                    radius_val *= 25.4
                corner_radius = radius_val

            if length and width and height:
                output_filename_base = f"box_{length}x{width}x{height}_wall{wall_thickness}"
                if text_to_imprint:
                    output_filename_base += f"_{text_to_imprint.replace(' ', '_')}"
                if corner_radius > 0:
                    output_filename_base += f"_radius{corner_radius}"

                box_output_filename = output_filename_base + ".stl"
                lid_output_filename = output_filename_base + "_lid.stl"  # Pre-define lid name for message

                print(
                    f"Chatbot: Generating custom box: L={length}mm, W={width}mm, H={height}mm, Wall={wall_thickness}mm, Text='{text_to_imprint}', Depth={imprint_depth}mm, Corner Radius={corner_radius}mm")

                # Call the function to create the box and potentially the lid
                model_file = create_custom_3d_printable_box(
                    length=length,
                    width=width,
                    height=height,
                    wall_thickness=wall_thickness,
                    text_to_imprint=text_to_imprint,
                    imprint_depth=imprint_depth,
                    output_filename=box_output_filename,  # Pass the box's intended filename
                    create_lid=create_lid,
                    corner_radius=corner_radius  # Pass the new parameter
                )

                if model_file:
                    print(f"Chatbot: Successfully generated the main box: '{os.path.abspath(box_output_filename)}'")
                    if create_lid:
                        print(f"Chatbot: And its matching lid is ready: '{os.path.abspath(lid_output_filename)}'")
                    # last_generated_stl_file is updated inside create_custom_3d_printable_box
                else:
                    print(
                        "Chatbot: Failed to programmatically generate the 3D box. Check parameters and error messages above.")
            else:
                print("Chatbot: Please provide clear length, width, and height for the box (e.g., '100x50x40').")
            continue
        # --- END Programmatic 3D Box Generation Command ---

        # --- Handle Transform Model Command (new, with toggles and aliases) ---
        elif user_input.lower().startswith('transform_model'):
            # Split user input into parts
            parts = user_input.lower().split()

            # Remove 'transform_model' command word
            args = parts[1:]

            input_filename = None
            output_filename = None
            apply_vertical_flip = False
            apply_mirror_yz = False

            # Parse arguments
            i = 0
            while i < len(args):
                arg = args[i]
                if arg.endswith('.stl'):
                    if input_filename is None:
                        input_filename = arg
                    elif output_filename is None:
                        output_filename = arg
                    # If both filenames are already set, ignore further .stl arguments for now
                elif arg == 'vertical-flip' or arg == 'flip_vertical':  # Add flip_vertical alias
                    apply_vertical_flip = True
                elif arg == 'mirror-yz' or arg == 'mirror_horizontal':  # Add mirror_horizontal alias
                    apply_mirror_yz = True
                i += 1

            # Fallback to last generated file if no input filename was provided by user
            if input_filename is None:
                if last_generated_stl_file:
                    input_filename = last_generated_stl_file
                    print(f"Chatbot: Using previously generated file: '{input_filename}' for transformation.")
                else:
                    print("Chatbot: No input STL file specified and no previously generated file to transform.")
                    print(
                        "Usage: 'transform_model <input.stl> [output.stl] [flags]' OR 'transform_model [flags]' (if a file was previously generated).")
                    print("Flags: vertical-flip, mirror-yz (or their aliases: flip_vertical, mirror_horizontal).")
                    continue

            # Generate default output filename if not provided by user
            if output_filename is None:
                base, ext = os.path.splitext(input_filename)
                output_filename = f"{base}_transformed{ext}"
                print(f"Chatbot: Output file name not specified. Saving as '{output_filename}'.")

            print(
                f"Chatbot: Attempting to transform model from '{input_filename}' to '{output_filename}' (Vertical Flip: {apply_vertical_flip}, Mirror YZ: {apply_mirror_yz})...")
            transformed_file = transform_model(input_filename, output_filename, apply_vertical_flip, apply_mirror_yz)
            if transformed_file:
                print(f"Chatbot: Model successfully transformed and saved as '{transformed_file}'.")
                # last_generated_stl_file is updated inside transform_model
            else:
                print("Chatbot: Failed to transform the model. Check messages above for details.")
            continue
        # --- END Transform Model Command ---

        # Check if Gemini session is available before attempting to process
        if gemini_chat_session is None:
            print(
                "Chatbot: I can't process that right now because my connection to Gemini isn't active. Please check your API key and try restarting, or use 'list_models'.")
            continue

        response, source_of_answer, confidence = process_query(user_input, chatbot_state, gemini_chat_session)

        if source_of_answer.startswith("None") or \
                (
                        source_of_answer == "Gemini (General Knowledge)" and "I encountered a problem because my current Gemini model isn't available" in response):
            print(f"Chatbot: {response}")


if __name__ == "__main__":
    try:
        # Check for sentence_transformers specifically before CadQuery
        from sentence_transformers import SentenceTransformer as _temp_ST

        print("Sentence-Transformers: All good to go!")
    except ImportError as e:
        print(f"\nMissing a crucial library! Error: {e}")
        print("The 'sentence-transformers' library isn't installed. It's needed for the AI's semantic understanding.")
        print("Please install it by running: pip install sentence-transformers")
        print("This will also install PyTorch or TensorFlow, which it depends on.")
        sys.exit(1)

    try:
        import cadquery

        print("CadQuery: All good to go!")
    except ImportError as e:
        print(f"\nMissing a crucial library! Error: {e}")
        print("\nLooks like 'cadquery' isn't installed!")
        print("Please install it (with its dependencies) by running: pip install cadquery")
        print("If you want the GUI for visual editing, also run: pip install cq-editor")
        sys.exit(1)

    # Original torch and fitz checks are now redundant if sentence-transformers is installed first,
    # as it brings in torch/tensorflow and PyMuPDF (fitz) is a separate concern.
    # Leaving the original message format for consistency with the past iterations if they become relevant.
    try:
        import torch
    except ImportError:
        print("\nOh no! PyTorch isn't installed. The Sentence-Transformer needs it to work its magic.")
        print("Please install PyTorch by running: pip install torch torchvision torchaudio")
        sys.exit(1)

    try:
        import pymupdf
    except ImportError:
        print("\nAh, the PDF reader library (PyMuPDF) isn't installed!")
        print("Please install it by running: pip install PyMuPDF")
        sys.exit(1)

    chat()
