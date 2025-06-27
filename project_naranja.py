import streamlit as st
import requests
import textwrap
import os
import json
import time
import spacy
import re
from dotenv import load_dotenv
from qbreader import Sync
from qbreader import Difficulty, Category, Tossup, Bonus, AnswerJudgement, Packet

# --- Page and State Configuration ---

st.set_page_config(
    page_title="Project Naranja - Knowledge Acquisition Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables from a .env file (for local development)
load_dotenv()

# --- Custom CSS for Styling ---
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    /* General Body and Font */
    body, .stApp {
        background-color: #f8f9fa; /* Off-white background */
        font-family: 'Poppins', sans-serif;
    }

    /* Main Title */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #212529 !important;
        text-align: center;
        padding-bottom: 1rem;
    }

    /* Subtitle/description below main title */
    .st-emotion-cache-1avp8dj p {
        text-align: center;
        color: #6c757d; 
    }

    /* Tab navigation */
    .st-emotion-cache-1r4qj8v {
        background-color: #e9ecef;
        border-radius: 0.75rem;
        padding: 0.5rem;
        margin-bottom: 2rem;
    }
    .st-emotion-cache-1r4qj8v label {
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        color: #495057;
    }
    /* FIX: More subtle highlight for selected tab */
    .st-emotion-cache-1r4qj8v input:checked + div {
        background-color: transparent !important; /* Make background transparent */
        color: #007bff !important;
        font-weight: 600;
        border: none !important; 
    }
    
    /* FIX: Remove highlight from sidebar radio buttons */
    div[data-testid="stRadio"] label {
        background-color: transparent !important;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }

    /* Main Content Containers */
    .st-emotion-cache-1r6slb0 {
        background: #ffffff;
        border-radius: 1rem;
        border: 1px solid #dee2e6;
        padding: 2rem !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Headings inside containers */
    .st-emotion-cache-1r6slb0 h2, .st-emotion-cache-1r6slb0 h3 {
         color: #343a40 !important;
    }
    
    /* Button Styles - Solid, professional look */
    .stButton > button {
        border-radius: 0.5rem !important;
        font-weight: 600 !important;
        color: white !important;
        background-color: #007bff !important;
        border: 1px solid #007bff !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        background-color: #0056b3 !important;
        border-color: #0056b3 !important;
    }
    .stButton > button:active {
        background-color: #004085 !important;
        border-color: #004085 !important;
    }

    /* Expander styles */
    .st-emotion-cache-b7hhl6 {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
    }
    
    /* Text Input style */
    .stTextInput input {
        border: 1px solid #ced4da !important;
        border-radius: 0.5rem !important;
        transition: border-color .15s ease-in-out, box-shadow .15s ease-in-out !important;
    }
    .stTextInput input:focus {
        border-color: #80bdff !important;
        box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25) !important;
    }

    /* Checkbox color */
    div[data-testid="stCheckbox"] svg {
        fill: #495057 !important; /* Professional dark gray */
    }
    
    div[data-testid="stCheckbox"] label p {
         color: #212529 !important;
    }

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# Initialize qbreader and spaCy clients
@st.cache_resource
def get_qbr_client():
    return Sync()

@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model and returns the nlp object."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy English model not found. Please run 'python -m spacy download en_core_web_sm' in your terminal.", icon="🚨")
        return None

qbr = get_qbr_client()
nlp = load_spacy_model()
if nlp is None:
    st.stop()


# --- Functions ---

@st.cache_data(show_spinner=False)
def get_ai_structured_explanation(prompt: str) -> dict:
    """
    Sends a prompt to the Gemini API and requests a structured JSON response.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Error: GEMINI_API_KEY not found. Please set it in your .env file.", icon="🚨")
        return {}

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    schema = {
        "type": "OBJECT", "properties": {
            "explanation": {"type": "STRING", "description": "The detailed explanation of the topic."},
            "image_search_query": {"type": "STRING", "description": "A concise, 2-3 word search term for the subject that is very likely to match a Wikipedia article title."},
            "recommended_reading": {
                "type": "ARRAY", "items": {
                    "type": "OBJECT", "properties": {"title": {"type": "STRING"}, "url": {"type": "STRING"}}, "required": ["title", "url"]
                }
            }
        }, "required": ["explanation", "image_search_query", "recommended_reading"]
    }
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema}}
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        if "candidates" in result:
            json_string = result["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(json_string)
        else:
            return {"explanation": "Sorry, the AI could not provide a structured explanation.", "image_search_query": "error", "recommended_reading": []}
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        st.error(f"An error occurred while getting the AI explanation: {e}", icon="🌐")
        return {}

def strip_html(text: str) -> str:
    """Removes HTML tags from a string."""
    if not text: return ""
    return re.sub('<[^<]+?>', '', text)

@st.cache_data
def get_set_list():
    """Fetches and caches the list of tournament sets."""
    return qbr.set_list()


def display_explanation_section(prompt):
    """Fetches and displays the AI explanation in a container."""
    with st.spinner("Generating explanation..."):
        data = get_ai_structured_explanation(prompt)
        if data:
            with st.container(border=True):
                query = data.get('image_search_query', 'book')
                google_image_search_url = f"https://www.google.com/search?tbm=isch&q={query.replace(' ', '+')}"
                st.link_button(f"🖼️ Search for images of '{query}'", google_image_search_url)

                st.markdown(data.get("explanation", "No explanation available."))
                if data.get("recommended_reading"):
                    st.markdown("**Further Reading:**")
                    for link in data["recommended_reading"]:
                        st.markdown(f"- [{link['title']}]({link['url']})")

# Define maps here to be accessible by get_new_question
difficulty_map = {"Middle School": Difficulty.MS, "High School (Easy)": Difficulty.HS_EASY, "High School (Regular)": Difficulty.HS_REGS, "High School (Hard)": Difficulty.HS_HARD, "High School (Nationals)": Difficulty.HS_NATS, "College (Easy)": Difficulty.ONE_DOT, "College (Medium)": Difficulty.TWO_DOT, "College (Hard)": Difficulty.THREE_DOT, "College (Hardest)": Difficulty.FOUR_DOT}
category_map = {"Literature": Category.LITERATURE, "History": Category.HISTORY, "Science": Category.SCIENCE, "Fine Arts": Category.FINE_ARTS, "Religion": Category.RELIGION, "Mythology": Category.MYTHOLOGY, "Philosophy": Category.PHILOSOPHY, "Social Science": Category.SOCIAL_SCIENCE, "Geography": Category.GEOGRAPHY, "Other Academic": Category.OTHER_ACADEMIC, "Pop Culture / Trash": Category.TRASH}

def get_new_question():
    """Fetches a new question and resets state based on selected checkboxes."""
    with st.spinner("Fetching a new question..."):
        try:
            q_type = st.session_state.q_type

            # Build list of selected difficulties from checkboxes
            selected_difficulties = [diff_enum for name, diff_enum in difficulty_map.items() if st.session_state.get(f"diff_{name}")]
            
            # Build list of selected categories from checkboxes
            selected_categories = [cat_enum for name, cat_enum in category_map.items() if st.session_state.get(f"cat_{name}")]

            if not selected_difficulties or not selected_categories:
                st.error("Please select at least one difficulty and one category.")
                return

            if q_type == "Tossup":
                question = qbr.random_tossup(difficulties=selected_difficulties, categories=selected_categories)[0]
                st.session_state.words = question.question_sanitized.split()
                st.session_state.tossup_state = 'waiting'
            else: # Bonus
                question = qbr.random_bonus(difficulties=selected_difficulties, categories=selected_categories, three_part_bonuses=True)[0]
                st.session_state.bonus_answers = []
            
            st.session_state.question = question
            st.session_state.word_index = -1
            st.session_state.feedback = None
        except IndexError:
             st.error("No questions found with the selected criteria. Please broaden your selection.")
             st.session_state.question = None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.question = None

if 'question' not in st.session_state: st.session_state.question = None
if 'search_results' not in st.session_state: st.session_state.search_results = None
if 'search_page' not in st.session_state: st.session_state.search_page = 1
if 'packet_data' not in st.session_state: st.session_state.packet_data = None
if 'packet_set_name' not in st.session_state: st.session_state.packet_set_name = None

# --- UI Layout ---
st.title("Project Naranja - Knowledge Acquisition Tool")

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Trainer"

tabs = ["Trainer", "Question Search", "Packet Study"]
active_tab = st.radio(
    "Navigation", 
    tabs, 
    key="active_tab", 
    horizontal=True, 
    label_visibility="collapsed"
)

# ==============================================================================
# --- TRAINER TAB ---
# ==============================================================================
if active_tab == "Trainer":
    with st.sidebar:
        st.header("Trainer Settings")
        st.radio("Question Type", ["Tossup", "Bonus"], key="q_type")
        with st.expander("Difficulties", expanded=True):
            for name in difficulty_map:
                st.checkbox(name, key=f"diff_{name}", value=(name == "High School (Regular)"))
        with st.expander("Categories", expanded=True):
            for name in category_map:
                st.checkbox(name, key=f"cat_{name}", value=True)
        st.button("Get New Question", on_click=get_new_question, use_container_width=True)

    if not st.session_state.question:
        st.info("Click 'Get New Question' in the sidebar to start practicing!")
    else:
        question = st.session_state.question
        is_tossup = isinstance(question, Tossup)
        with st.container(border=True):
            st.subheader(f"{st.session_state.q_type} Question"); st.markdown(f"**Set:** {question.set.name} | **Category:** {question.category.value}")
            if is_tossup:
                tossup_state = st.session_state.get('tossup_state', 'waiting')
                if tossup_state == 'waiting':
                    st.markdown("_(Ready to read tossup)_")
                    if st.button("▶️ Start Reading", use_container_width=True):
                        st.session_state.tossup_state = 'reading'; st.session_state.word_index = 0; st.rerun()
                elif tossup_state == 'reading':
                    revealed_text = " ".join(st.session_state.words[:st.session_state.word_index + 1])
                    st.markdown(revealed_text)
                    if st.button("⚡ BUZZ IN!", use_container_width=True):
                        st.session_state.tossup_state = 'buzzed'; st.rerun()
                    if st.session_state.word_index < len(st.session_state.words) - 1:
                        st.session_state.word_index += 1; time.sleep(0.15); st.rerun()
                    else:
                        st.session_state.tossup_state = 'grace_period'; st.session_state.grace_period_end = time.time() + 5; st.rerun()
                elif tossup_state == 'grace_period':
                    st.markdown(question.question_sanitized)
                    remaining_time = st.session_state.grace_period_end - time.time()
                    if remaining_time > 0:
                        st.info(f"Time to buzz: {remaining_time:.1f}s")
                        if st.button("⚡ BUZZ IN!", use_container_width=True):
                            st.session_state.tossup_state = 'buzzed'; st.rerun()
                        time.sleep(0.1); st.rerun()
                    else:
                        st.session_state.tossup_state = 'over'; st.rerun()
                elif tossup_state == 'buzzed':
                    revealed_text = " ".join(st.session_state.words[:st.session_state.word_index + 1]) if st.session_state.word_index > -1 else ""
                    st.markdown(revealed_text)
                    with st.form("answer_form"):
                        user_answer = st.text_input("Enter your answer", key="user_answer_input", label_visibility="collapsed")
                        if st.form_submit_button("Submit Answer", use_container_width=True):
                            judgement = question.check_answer_sync(user_answer)
                            st.session_state.feedback = ("success", "Correct!") if judgement.correct() else ("error", "Incorrect.")
                            st.session_state.tossup_state = 'over'; st.rerun()
                elif tossup_state == 'over':
                    st.markdown(question.question_sanitized)
                    if st.session_state.feedback:
                        getattr(st, st.session_state.feedback[0])(st.session_state.feedback[1], icon="✅" if st.session_state.feedback[0] == "success" else "❌")
                    else:
                        st.warning("Time's up!")
            else: # Bonus
                st.markdown(f"**Leadin:** {question.leadin}"); st.markdown("---")
                for i in range(len(question.parts)):
                    st.markdown(f"**Part {i+1}:** {question.parts[i]}")
                    if len(st.session_state.get('bonus_answers', [])) > i:
                        is_correct, user_ans = st.session_state.bonus_answers[i]
                        if is_correct: st.success(f"You answered: '{user_ans}' (Correct)", icon="✔️")
                        else: 
                            st.error(f"You answered: '{user_ans}' (Incorrect)", icon="✖️")
                            st.markdown(f"💡 **Correct Answer:** {question.answers[i]}", unsafe_allow_html=True)
                    else:
                        with st.form(f"bonus_part_{i}"):
                            bonus_answer = st.text_input("Answer", key=f"bonus_input_{i}", label_visibility="collapsed")
                            if st.form_submit_button("Submit Part Answer"):
                                judgement = qbr.check_answer(question.answers[i], bonus_answer)
                                if 'bonus_answers' not in st.session_state: st.session_state.bonus_answers = []
                                st.session_state.bonus_answers.append((judgement.correct(), bonus_answer)); st.rerun()
                        break
                if len(st.session_state.get('bonus_answers', [])) == len(question.parts):
                    st.session_state.tossup_state = 'over'
        
        is_game_over = st.session_state.get('tossup_state') == 'over'
        if is_game_over:
            st.divider(); st.header("🕵️‍♀️ Review and Analyze")
            if is_tossup:
                st.markdown(f"**Correct Answer:** {question.answer}", unsafe_allow_html=True)
                with st.expander("Show Overall Answer Summary"):
                    prompt = f'Act as a subject matter expert. Provide a detailed, in-depth encyclopedic summary of "{question.answer_sanitized}". Use Markdown bolding to highlight key terms. Also provide a concise, 2-3 word search term for a relevant image and 2-3 links for further reading. Prioritize links from Wikipedia and Encyclopedia Britannica.'
                    display_explanation_section(prompt)
                st.subheader("Clue-by-Clue Breakdown")
                doc = nlp(question.question_sanitized)
                review_sentences = [sent.text for sent in doc.sents]
                for i, sentence in enumerate(review_sentences):
                    if not sentence: continue
                    with st.expander(f"**Clue {i+1}:** *{sentence.strip()}*"):
                        prompt = f'The overall answer to a quizbowl question is "{question.answer_sanitized}". Your role is a subject-matter expert. Your task is to provide a detailed, in-depth explanation of the specific names, places, or concepts within this single clue: "{sentence.strip()}". Explain how they connect to the main answer. Crucially, use Markdown bolding (**text**) to highlight the most important key terms. Do NOT repeat general information about the main answer. Provide a search query and reading links specific to this clue\'s specific content. Prioritize links from Wikipedia and Encyclopedia Britannica.'
                        display_explanation_section(prompt)
            else: # Bonus
                with st.expander("Show Overall Bonus Topic Summary"):
                    prompt = f'Act as a subject matter expert. The lead-in to a bonus is: "{question.leadin}". Provide a detailed, in-depth summary of the likely overall topic. Use Markdown bolding to highlight key terms. Also provide a concise, 2-3 word search term for a relevant image and reading links. Prioritize links from Wikipedia and Encyclopedia Britannica.'
                    display_explanation_section(prompt)
                st.subheader("Part-by-Part Breakdown")
                for i in range(len(question.parts)):
                    with st.expander(f"**Part {i+1} Review**"):
                        st.markdown(f"**Question:** {question.parts[i]}"); st.markdown(f"**Correct Answer:** {question.answers[i]}", unsafe_allow_html=True)
                        prompt = f'Act as a subject matter expert. A quizbowl question asks: "{question.parts[i]}". The correct answer is "{question.answers[i]}". Provide a very detailed, in-depth, encyclopedic explanation of the answer in the context of the question. Use Markdown bolding to highlight key terms. Do not repeat the question itself in your explanation. Also provide a search query and reading links for "{question.answers[i]}". Prioritize links from Wikipedia and Encyclopedia Britannica.'
                        display_explanation_section(prompt)
            
            if st.button("Next Question →", on_click=get_new_question, use_container_width=True):
                get_new_question()
                st.rerun()

# ==============================================================================
# --- QUESTION SEARCH TAB ---
# ==============================================================================
elif active_tab == "Question Search":
    st.header("🔎 Question Database Search")

    with st.form("search_form"):
        st.text_input("Search Query", key="search_query")
        st.selectbox("Search In", options=["Question", "Answer", "Both"], key="search_type")
        
        cols = st.columns(2)
        with cols[0]:
            with st.expander("Difficulty Filters", expanded=True):
                for name in difficulty_map:
                    st.checkbox(name, key=f"search_diff_{name}", value=True)
        with cols[1]:
             with st.expander("Category Filters", expanded=True):
                for name in category_map:
                    st.checkbox(name, key=f"search_cat_{name}", value=True)

        with st.expander("Question Type Filters", expanded=True):
            st.checkbox("Tossup", key=f"search_q_type_Tossup", value=True)
            st.checkbox("Bonus", key=f"search_q_type_Bonus", value=True)
        
        submitted = st.form_submit_button("Search", use_container_width=True)

        if submitted:
            st.session_state.search_page = 1 
            st.session_state.search_submitted = True


    if st.session_state.get('search_submitted'):
        with st.spinner("Searching..."):
            search_q_types = [q_type for q_type in ["Tossup", "Bonus"] if st.session_state.get(f"search_q_type_{q_type}")]
            if "Tossup" in search_q_types and "Bonus" in search_q_types: q_type_param = "all"
            elif "Tossup" in search_q_types: q_type_param = "tossup"
            elif "Bonus" in search_q_types: q_type_param = "bonus"
            else: q_type_param = "all"

            search_difficulties_values = [diff_enum.value for name, diff_enum in difficulty_map.items() if st.session_state.get(f"search_diff_{name}")]
            search_categories_values = [cat_enum.value for name, cat_enum in category_map.items() if st.session_state.get(f"search_cat_{name}")]

            API_BASE_URL = "https://www.qbreader.org/api/query"
            params = {
                'queryString': st.session_state.search_query, 
                'searchType': st.session_state.search_type.lower(),
                'difficulties': ",".join([str(d) for d in search_difficulties_values]),
                'categories': ",".join(search_categories_values),
                'questionType': q_type_param, 'maxReturnLength': 10,
                'tossupPagination': st.session_state.search_page,
                'bonusPagination': st.session_state.search_page
            }
            response = requests.get(API_BASE_URL, params=params)
            response.raise_for_status()
            json_data = response.json()
            st.session_state.search_results = (json_data, st.session_state.search_query)

    if st.session_state.get('search_results'):
        results, query = st.session_state.search_results
        st.divider()
        st.subheader(f"Search Results for '{query}'")
        
        tossups_data = results.get('tossups', {})
        tossup_array = tossups_data.get('questionArray', [])
        tossup_count = tossups_data.get('count', 0)
        
        if tossup_array:
            st.subheader(f"Tossups ({tossup_count} found)")
            for tossup_data in tossup_array:
                with st.container(border=True):
                    st.markdown(f"**Set:** {tossup_data.get('set', {}).get('name', 'N/A')}")
                    st.markdown(f"{strip_html(tossup_data.get('question', ''))}")
                    
                    with st.expander("Answer & Analysis"):
                        st.markdown(f"**Answer:** {tossup_data.get('answer', 'N/A')}", unsafe_allow_html=True)
                        if st.button("Analyze Full Answer", key=f"search_ans_{tossup_data.get('_id')}"):
                            prompt = f'Act as a subject matter expert. Provide a detailed, in-depth encyclopedic summary of "{strip_html(tossup_data.get("answer", ""))}". Use Markdown bolding to highlight key terms. Also provide a concise, 2-3 word search term for a relevant image and 2-3 links for further reading. Prioritize links from Wikipedia and Encyclopedia Britannica.'
                            display_explanation_section(prompt)
                        st.markdown("---")
                        st.markdown("**Clue-by-Clue Breakdown**")
                        doc = nlp(strip_html(tossup_data.get('question', '')))
                        sentences = [sent.text for sent in doc.sents]
                        for i, sentence in enumerate(sentences):
                            st.markdown(f"*{sentence.strip()}*")
                            if st.button("Analyze Clue", key=f"search_tossup_{tossup_data.get('_id')}_clue_{i}"):
                                prompt = f'The overall answer to a quizbowl question is "{strip_html(tossup_data.get("answer", ""))}". Your role is a subject-matter expert... (prompt continues)'
                                display_explanation_section(prompt)
        
        bonuses_data = results.get('bonuses', {})
        bonus_array = bonuses_data.get('questionArray', [])
        bonus_count = bonuses_data.get('count', 0)
        if bonus_array:
            st.subheader(f"Bonuses ({bonus_count} found)")
            for bonus_data in bonus_array:
                 with st.container(border=True):
                    st.markdown(f"**Set:** {bonus_data.get('set', {}).get('name', 'N/A')}")
                    st.markdown(f"**Leadin:** {strip_html(bonus_data.get('leadin', ''))}")
                    st.markdown("---")
                    for i, part in enumerate(bonus_data.get('parts', [])):
                        st.markdown(f"**Part {i+1}:** {strip_html(part)}")
                        st.markdown(f"**Answer:** {bonus_data.get('answers', ['N/A'])[i]}", unsafe_allow_html=True)
                    
                    with st.expander("Analyze Bonus with AI"):
                        st.markdown(f"**Leadin:** *{strip_html(bonus_data.get('leadin', ''))}*")
                        if st.button("Analyze Leadin", key=f"search_leadin_{bonus_data.get('_id')}"):
                            prompt = f'Act as a subject matter expert. The lead-in to a bonus is: "{strip_html(bonus_data.get("leadin", ""))}". Provide a detailed, in-depth summary... (prompt continues)'
                            display_explanation_section(prompt)

                        for i, part in enumerate(bonus_data.get('parts', [])):
                            st.markdown("---")
                            ans_text = strip_html(bonus_data.get('answers', ['N/A'])[i])
                            st.markdown(f"**Part {i+1}:** *{strip_html(part)}*")
                            st.markdown(f"**Answer:** *{ans_text}*")
                            if st.button("Analyze Part", key=f"search_bonus_{bonus_data.get('_id')}_part_{i}"):
                                prompt = f'Act as a subject matter expert. A quizbowl question asks: "{strip_html(part)}"... (prompt continues)'
                                display_explanation_section(prompt)

        total_results = max(tossup_count, bonus_count)
        if total_results > 10:
            total_pages = (total_results // 10) + (1 if total_results % 10 > 0 else 0)
            
            page_cols = st.columns([1, 1, 1])
            if page_cols[0].button("⬅️ Previous Page", disabled=(st.session_state.search_page <= 1)):
                st.session_state.search_page -= 1; st.rerun()
            
            page_cols[1].markdown(f"<div style='text-align: center; margin-top: 0.5rem;'>Page {st.session_state.search_page} of {total_pages}</div>", unsafe_allow_html=True)

            if page_cols[2].button("Next Page ➡️", disabled=(st.session_state.search_page >= total_pages)):
                st.session_state.search_page += 1; st.rerun()

# ==============================================================================
# --- PACKET STUDY TAB ---
# ==============================================================================
elif active_tab == "Packet Study":
    st.header("📖 Packet Study")
    set_list = get_set_list()
    if set_list:
        set_name = st.selectbox("Select a Tournament Set", options=set_list, key="packet_set_name_selector")
        
        if set_name:
            num_packets = qbr.num_packets(set_name)
            if num_packets > 0:
                packet_number = st.number_input("Select Packet Number", min_value=1, max_value=num_packets, value=1)
                if st.button("Load Packet", use_container_width=True):
                    with st.spinner("Loading packet..."):
                        API_BASE_URL = "https://www.qbreader.org/api/packet"
                        params = {'setName': set_name, 'packetNumber': packet_number}
                        response = requests.get(API_BASE_URL, params=params)
                        response.raise_for_status()
                        json_data = response.json()
                        
                        st.session_state.packet_data = json_data 
                        st.session_state.packet_set_name = set_name
            else:
                st.warning("This set does not appear to have any packets.")

    if st.session_state.get('packet_data'):
        st.divider()
        st.subheader(f"Viewing: {st.session_state.packet_set_name} - Packet {st.session_state.packet_data.get('number', '')}")
        
        # Display Tossups
        tossups = st.session_state.packet_data.get('tossups', [])
        if tossups:
            st.header("Tossups")
            for i, tossup in enumerate(tossups):
                with st.container(border=True):
                    st.markdown(f"**Tossup {i+1}**")
                    st.markdown(strip_html(tossup.get('question', '')))
                    st.markdown(f"**Answer:** {tossup.get('answer', '')}", unsafe_allow_html=True)
                    with st.expander("Analyze Clue-by-Clue"):
                        doc = nlp(strip_html(tossup.get('question', '')))
                        sentences = [sent.text for sent in doc.sents]
                        for j, sentence in enumerate(sentences):
                            st.markdown(f"*{sentence.strip()}*")
                            if st.button("Analyze Clue", key=f"packet_tossup_{i}_clue_{j}"):
                                display_explanation_section(f'The answer is "{strip_html(tossup.get("answer", ""))}". Your role is a subject-matter expert. Your task is to provide a detailed, in-depth explanation of the specific names, places, or concepts within this single clue: "{sentence.strip()}". Explain how they connect to the main answer. Crucially, use Markdown bolding (**text**) to highlight the most important key terms. Do NOT repeat general information about the main answer. Provide a search query and reading links specific to this clue\'s specific content. Prioritize links from Wikipedia and Encyclopedia Britannica.')
        
        # Display Bonuses
        bonuses = st.session_state.packet_data.get('bonuses', [])
        if bonuses:
            st.header("Bonuses")
            for i, bonus in enumerate(bonuses):
                with st.container(border=True):
                    st.markdown(f"**Bonus {i+1}**")
                    st.markdown(f"**Leadin:** {strip_html(bonus.get('leadin', ''))}")
                    st.markdown("---")
                    for j, part in enumerate(bonus.get('parts', [])):
                        st.markdown(f"**Part {j+1}:** {strip_html(part)}")
                        st.markdown(f"**Answer:** {bonus.get('answers', ['N/A'])[j]}", unsafe_allow_html=True)
                    with st.expander("Analyze Part-by-Part"):
                        for j, part in enumerate(bonus.get('parts', [])):
                            st.markdown("---")
                            st.markdown(f"**Part {j+1}:** *{strip_html(part)}*")
                            st.markdown(f"**Answer:** {bonus.get('answers', ['N/A'])[j]}", unsafe_allow_html=True)
                            if st.button("Analyze Part", key=f"packet_bonus_{i}_part_{j}"):
                                display_explanation_section(f'Act as a subject matter expert. A quizbowl question asks: "{strip_html(part)}". The correct answer is "{bonus.get("answers", ["N/A"])[j]}". Provide a very detailed, in-depth, encyclopedic explanation of the answer in the context of the question. Use Markdown bolding to highlight key terms. Do not repeat the question itself in your explanation. Also provide a search query and reading links for "{bonus.get("answers", ["N/A"])[j]}". Prioritize links from Wikipedia and Encyclopedia Britannica.')

