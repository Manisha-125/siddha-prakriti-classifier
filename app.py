from flask import Flask, render_template, request, jsonify, send_file
from gtts import gTTS
import google.generativeai as genai
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import numpy as np
import joblib
import os
import time
import pandas as pd
import json
import datetime
from gemini_answer_mapper import map_answer_with_gemini, map_clicked_option, build_feature_vector
from consultation_engine import run_consultation
from question_feature_mapping import MODEL_FEATURES
app = Flask(__name__)

# ============================================================
# ⚙️ MODEL ENGINE CORES & CONFIGURATION
# ============================================================
model = joblib.load('prakriti_model.pkl')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    SELECTED_FEATURES = joblib.load('selected_features.pkl')
except Exception:
    SELECTED_FEATURES = MODEL_FEATURES

DECODING_MAP = {0: "Vata", 1: "Pitta", 2: "Kapha"}
EXCEL_FILE_PATH = "Siddha_Prakriti_Database_Optimized.xlsx"
last_diagnostic_report = {}

OFFLINE_MULTILINGUAL_MAP = {
    "Minimal sweating, may have dry body odor": 0, "குறைந்த வியர்வை, வறண்ட உடல் நாற்றம் இருக்கலாம்": 0, "വളരെ കുറഞ്ഞ വേർപ്പ്, വരണ്ട ശരീര ഗന്ധം ഉണ്ടായേക്കാം": 0,
    "Moderate, may sweat a lot, strong body odor": 1, "மிதமான அல்லது அதிக வியர்வை, கடுமையான உடல் நாற்றம்": 1, "മിതമായ അല്ലെങ്കിൽ അമിതമായി വേർക്കുന്നു, കടുത്ത ശരീര ഗന്ധം": 1,
    "Sweats less but can feel damp in humid weather": 2, "குறைவாக வியர்க்கும் ஆனால் ஈரப்பதமான காலநிலையில் வியர்த்து கொட்டும்": 2, "വേർപ്പ് കുറവാണ് എങ്കിലും ഈർപ്പമുള്ള കാലാവസ്ഥയിൽ നനവ് അനുഭവപ്പെടുന്നു": 2,
    "Narrow, dry, bony, may have cracks": 0, "குறுகலான, வறண்ட, எலும்புகள் தெரியும் பாதம், வெடிப்புகள் இருக்கலாம்": 0, "ഇടുങ്ങിയത്, വരണ്ടത്, അസ്ഥികൾ കാണാവുന്നത്, വിണ്ടുകീറൽ ഉണ്ടായേക്കാം": 0,
    "Medium, warm, well-shaped": 1, "நடுத்தர அளவு, வெப்பமான, நல்ல வடிவமைப்பு கொண்ட பாதம்": 1, "മിതമായ വലുപ്പം, ചൂടുള്ളത്, നല്ല ആകൃതിയുള്ളത്": 1,
    "Large, thick, soft, strong": 2, "பெரிய, தடிமனான, மென்மையான மற்றும் வலுவான பாதம்": 2, "വലുത്, തടിച്ചത്, മൃദുവായത്, ശക്തമായ പാദങ്ങൾ": 2,
    "Narrow, small, sometimes bony": 0, "குறுகலான, சிறிய, சில நேரங்களில் எலும்புகள் தெரியும் தோள்கள்": 0, "ഇടുങ്ങിയത്, ചെറുത്, ചിലപ്പോൾ അസ്ഥികൾ ദൃശ്യമാകുന്നത്": 0,
    "Medium, well-developed, defined": 1, "மிதமான, நன்கு வளர்ந்த, தெளிவான வடிவமைப்பு": 1, "മിത വലുപ്പം, നന്നായി വികസിച്ചത്, വ്യക്തമായ ആകൃതി": 1,
    "Broad, rounded, strong but relaxed": 2, "அகன்ற, உருண்டையான, வலுவான மற்றும் தளர்த்தியான தோள்கள்": 2, "വീതിയേറിയത്, ഉരുണ്ടത്, ശക്തമെങ്കിലും അയഞ്ഞ തോളുകൾ": 2,
    "Dry, rough, cool, thin, prone to cracks and flakiness, dark complexion": 0, "வறண்ட, சொரசொரப்பான, குளிர்ந்த, மெல்லிய தோல், வெடிப்புகள் மற்றும் செதில்கள் வரலாம், கறுத்த நிறம்": 0, "വരണ്ടത്, പരുപരുത്തത്, തണുത്തത്, മെലിഞ്ഞത്, വിണ്ടുകീരാനും തൊലി ഇളകാനും സാധ്യതയുള്ളത്, ഇരുണ്ട നിറം": 0,
    "Warm, soft, reddish, sensitive, prone to rashes, acne, and inflammation": 1, "வெப்பமான, மென்மையான, சிவந்த தோல், அலர்ஜி, முகப்பரு மற்றும் அழற்சி வரக்கூடியது": 1, "ചൂടുള്ളത്, മൃദുവായത്, ചുവപ്പുനിറം, തടിപ്പുകൾ, മുഖക്കുരു, വീക്കം എന്നിവയ്ക്ക് സാധ്യതയുള്ളത്": 1,
    "Thick, smooth, oily, pale, cool to touch": 2, "தடிமனான, மென்மையான, எண்ணெய் பசையுள்ள, வெளிறிய, தொடுவதற்கு குளிர்ந்த தோல்": 2, "തടിച്ചത്, മിനുസമാർന്നത്, എണ്ണമയമുള്ളത്, വെളുത്തത്, തൊടുമ്പോൾ തണുപ്പുള്ളത്": 2,
    "Thin, dry, rough, may have cracks, tends to tremble": 0, "மெல்லிய, வறண்ட, சொரசொரப்பான நாக்கு, வெடிப்புகள் இருக்கலாம், லேசாக நடுங்கும்": 0, "മെലിഞ്ഞത്, വരണ്ടത്, പരുപരുത്തത്, വിള്ളലുകൾ ഉണ്ടായേക്കാം, വിറയ്ക്കാൻ സാധ്യതയുള്ളത്": 0,
    "Medium-sized, pinkish-red, may have ulcers or inflammation": 1, "நடுத்தர அளவு, இளஞ்சிவப்பு-சிவப்பு நாக்கு, புண்கள் அல்லது அழற்சி வரலாம்": 1, "മിത വലുപ്പം, ചുവപ്പ് കലർന്ന പിങ്ക് നിറം, വ്രണങ്ങൾ അല്ലെങ്കിൽ വീക്കം ഉണ്ടാകാം": 1,
    "Thick, smooth, pale, and moist": 2, "தடிமனான, மென்மையான, வெளிறிய மற்றும் ஈரப்பதமான நாக்கு": 2, "തടിച്ചത്, മിനുസമാർന്നത്, വെളുത്ത നിറം, ഈർപ്പമുള്ളത്": 2,
    "Sunken or bony, may appear hollow": 0, "ஒட்டிய அல்லது எலும்புகள் தெரியும் கன்னங்கள், குழி விழுந்து காணப்படலாம்": 0, "ഒട്ടിയത് അല്ലെങ്കിൽ അസ്ഥികൾ ദൃശ്യമാകുന്നത്, കുഴിഞ്ഞതായി തോന്നാം": 0,
    "Medium, sometimes reddish": 1, "நடுத்தர அளவு, சில நேரங்களில் சிவந்து காணப்படும்": 1, "മിത വലുപ്പം, ചിലപ്പോൾ ചുവപ്പ് കലർന്ന നിറം": 1,
    "Full, plump, smooth": 2, "முழுமையான, உப்பலான, மென்மையான கன்னங்கள்": 2, "നിറഞ്ഞത്, ഉരുണ്ടത്, മിനുസമാർന്ന കവിളുകൾ": 2,
    "Thin, long, often dry": 0, "மெலிந்த, நீளமான, பெரும்பாலும் வறண்ட கழுத்து": 0, "മെലിഞ്ഞത്, நீളമുള്ളത്, മിക്കപ്പോഴും വരണ്ടത്": 0,
    "Medium-sized, muscular, warm": 1, "நடுத்தர அளவு, தசைப்பிடிப்புள்ள, வெப்பமான கழுத்து": 1, "മിത വലുപ്പം, പേശീബലമുള്ളത്, ചൂടുള്ളത്": 1,
    "Thick, strong, short, smooth": 2, "தடிமனான, வலுவான, குட்டையான, மிருதுவான கழுத்து": 2, "തടിച്ചത്, ശക്തമായത്, ചെറുത്, മിനുസമാർന്നത്": 2,
    "Restless, energetic, anxious, talkative, active but tires quickly": 0, "அமைதியற்ற, சுறுசுறுப்பான, கவலைப்படும், அதிகம் பேசும் குணம், எளிதில் சோர்வடைதல்": 0, "அസ്വസ്ഥതയുള്ള, ഊർജ്ജസ്വലനായ, ഉത്കണ്ഠയുള്ള, സംസാരിക്കാൻ ഇഷ്ടപ്പെടുന്ന, വേഗം തളരുന്ന സ്വഭാവം": 0,
    "Ambitious, determined, goal-oriented, sharp-witted, sometimes aggressive": 1, "பேராசை அல்லது லட்சியம் கொண்ட, உறுதியான, கூர்மையான புத்தி, சில நேரங்களில் ஆக்ரோஷம்": 1, "ഉയർന്ന ലക്ഷ്യങ്ങളുള്ള, ദൃഢനിശ്ചയമുള്ള, മൂർച്ചയുള്ള ബുദ്ധിയുള്ള, ചിലപ്പോൾ आक्रमणോത്സുക സ്വഭാവം": 1,
    "Calm, steady, slow-moving, content, resistant to change, relaxed": 2, "அமைதியான, நிலையான, மெதுவான, திருப்தியான, மாற்றங்களை எதிர்க்கும் குணம், நிதானம்": 2, "ശാന്തമായ, സ്ഥിരതയുള്ള, പതുക്കെയുള്ള, സംതൃപ്തിയുള്ള, മാറ്റങ്ങളെ എതിർക്കുന്ന സ്വഭാവം": 2,
    "Thin, long, light bones, may lack muscle": 0, "மெலிந்த, நீளமான, இலகுவான எலும்புகள், தசை குறைவாக இருக்கலாம்": 0, "മെലിഞ്ഞത്, നീളമുള്ളത്, ഭാരം കുറഞ്ഞ അസ്ഥികൾ, പേശികൾ കുറവായിരിക്കാം": 0,
    "Medium, defined, warm": 1, "மிதமான தசை, தெளிவான வடிவமைப்பு, வெப்பமான கைகள்": 1, "മിത വലുപ്പം, വ്യക്തമായ ആകൃതി, ചൂടുള്ള കൈകൾ": 1,
    "Strong, thick, heavy": 2, "வலுவான, தடிமனான, கனமான கைகள்": 2, "ശക്തമായത്, തടിച്ചത്, ഭാരമുള്ള കൈകൾ": 2,
    "Dry, thin, long fingers, cold to touch": 0, "வறண்ட, மெலிந்த, நீளமான விரல்கள், தொடுவதற்கு குளிர்ச்சியாக இருக்கும்": 0, "വരണ്ടത്, മെലിഞ്ഞത്, നീളമുള്ള വിരലുകൾ, തൊടുമ്പോൾ തണുപ്പുള്ളത്": 0,
    "Medium-sized, warm, pinkish": 1, "நடுத்தர அளவு, வெப்பமான, இளஞ்சிவப்பு நிற உள்ளங்கைகள்": 1, "മിത വലുപ്പം, ചൂടുള്ളത്, പിങ്ക് നിറമുള്ള ഉള്ളംകൈകൾ": 1,
    "Thick, large, soft, moist palms": 2, "தடிமனான, பெரிய, மென்மையான, ஈரப்பதமான உள்ளங்கைகள்": 2, "തടിച്ചത്, വലുത്, മൃദുവായത്, ഈർപ്പമുള്ള ഉള്ളംകൈകൾ": 2
}

# ============================================================
# 📊 AUTOMATED GOOGLE SHEETS PIPELINE LOG ENGINE
# ============================================================
def log_patient_to_google_sheet(patient_info, mapped_answers, prediction_label, probabilities, lime_data, clinical_narrative):
    SPREADSHEET_ID = "1rjBd4Ij7DJNhjgLZrk-O6ir9Wjj8MG7Txt7UmZZWW8k"
    RANGE_NAME = "Admin_18_Feature_Audit!A:ZZ"

    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]

        if os.path.exists("credentials.json"):
            creds = Credentials.from_service_account_file("credentials.json", scopes=scopes)
        elif os.environ.get("GOOGLE_CREDENTIALS_JSON"):
            service_account_info = json.loads(os.environ.get("GOOGLE_CREDENTIALS_JSON"))
            creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
        else:
            raise FileNotFoundError("Google Cloud service account credentials not found.")

        service = build("sheets", "v4", credentials=creds)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row_payload = [
            timestamp,
            patient_info.get("name", "Anonymous"),
            patient_info.get("age", "N/A"),
            patient_info.get("gender", "N/A"),
            patient_info.get("location", "N/A"),
            patient_info.get("language", "en-US"),
            prediction_label,
            f"{probabilities.get('Vata', 0):.2f}%",
            f"{probabilities.get('Pitta', 0):.2f}%",
            f"{probabilities.get('Kapha', 0):.2f}%"
        ]

        answer_map = {item["feature"]: item for item in mapped_answers}

        for feature in MODEL_FEATURES:
            item = answer_map.get(feature, {})

            row_payload.extend([
                feature,
                item.get("patient_response", "N/A"),
                item.get("selected_option", "N/A"),
                item.get("value", "N/A"),
                item.get("dosha", "N/A"),
                ", ".join(item.get("keywords", [])) if item.get("keywords") else "N/A",
                item.get("reason", "N/A"),
                item.get("input_mode", "N/A")
            ])

        top_lime = lime_data[:5] if lime_data else []

        for i in range(5):
            if i < len(top_lime):
                row_payload.extend([
                    top_lime[i].get("feature", "N/A"),
                    top_lime[i].get("weight", "N/A"),
                    top_lime[i].get("impact", "N/A")
                ])
            else:
                row_payload.extend(["N/A", "N/A", "N/A"])

        row_payload.append(clinical_narrative)

        body = {"values": [row_payload]}

        service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=RANGE_NAME,
            valueInputOption="USER_ENTERED",
            body=body
        ).execute()

        print("[CLOUD PIPELINE SUCCESS] 18-feature audit log synced to Google Sheets.")

    except Exception as e:
        print(f"[GOOGLE API EXCEPTION WARNING] Cloud sync failed: {e}")

def generate_explainable_ai_rationale(prediction, probabilities, patient_vector, language_code):
    try:
        genai.configure(api_key=GEMINI_API_KEY)

        gemini_model = genai.GenerativeModel("gemini-2.5-flash")

        global_rf_importance = model.feature_importances_

        reasoning_payload = []
        for i in range(min(len(patient_vector), len(SELECTED_FEATURES))):
            reasoning_payload.append({
                "feature": SELECTED_FEATURES[i],
                "user_token_value": int(patient_vector[i]),
                "mathematical_model_weight": float(global_rf_importance[i])
            })

        prompt = f"""
        You are an expert Explainable AI (XAI) clinical interpreter for a Siddha Yakkai Ilakkanam classification software pipeline.
        The machine learning system (Random Forest) evaluated a patient and predicted their dominant constitution as: {prediction}.

        Model Probabilities Breakdown:
        - Vata: {probabilities[0]*100:.2f}%
        - Pitta: {probabilities[1]*100:.2f}%
        - Kapha: {probabilities[2]*100:.2f}%

        Granular Feature Route Data Matrix:
        {json.dumps(reasoning_payload)}

        CRITICAL TASK:
        Generate a cohesive, professional Explainable AI diagnosis report written entirely in the requested regional language context.
        Identify the top 2 features that had the highest mathematical model weight and explain how the patient's choices contributed to the final prediction.

        LANGUAGE CONSTRAINT:
        - If language_code is 'ta-IN', write the entire output statement in highly natural, readable TAMIL text script.
        - If language_code is 'ml-IN', write the entire output statement in highly natural, readable MALAYALAM text script.
        - Otherwise, write the output statement in standard clinical ENGLISH text.

        Do not output any raw code blocks, bullet keys, markdown markers, or JSON tags. Output ONLY plain conversational text paragraphs.
        """

        response = gemini_model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print(f"[XAI LOGGING EXCEPTION]: {e}")
        return f"Classification alignment verified. Predominant attributes trace back directly to classic {prediction} vectors."
@app.route('/')
def home():
    return render_template('index5.html')

@app.route('/speak_prompt', methods=['POST'])
def speak_prompt():
    data = request.json
    text_to_speak = data.get('text')
    lang_code = data.get('language').split('-')[0]
    try:
        tts = gTTS(text=text_to_speak, lang=lang_code, slow=False)
        if not os.path.exists('static'):
            os.makedirs('static')
        filename = "static/speech_prompt.mp3"
        tts.save(filename)
        return jsonify({"success": True, "audio_url": f"/{filename}?v={os.urandom(4).hex()}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/parse_response', methods=['POST'])
def parse_response():
    data = request.json or {}

    question_number = int(data.get("question_number"))
    input_mode = data.get("input_mode", "speech_or_text")
    speech_text = data.get("speech_text", "").strip()
    selected_option = data.get("selected_option", "")

    try:
        if input_mode == "option_click":
            mapped = map_clicked_option(question_number, selected_option)
        else:
            mapped = map_answer_with_gemini(question_number, speech_text)

        return jsonify({
            "success": True,
            "mapped_answer": mapped,
            "token": mapped["value"]
        })

    except Exception as e:
        print("[PARSE RESPONSE ERROR]:", e)
        return jsonify({"success": False, "error": str(e)})

@app.route('/predict_prakriti', methods=['POST'])
def predict_prakriti():
    global last_diagnostic_report

    data = request.json or {}

    mapped_answers = data.get("mapped_answers", [])
    demographics = data.get("demographics", {})
    lang_context = data.get("language", "en-US")

    try:
        report = run_consultation(mapped_answers)

        report["demographics"] = demographics
        report["language"] = lang_context

        last_diagnostic_report = report

        aligned_vector, _ = build_feature_vector(mapped_answers)
        raw_responses = [item.get("patient_response", "") for item in mapped_answers]

        new_record = {
            "Patient_Name": demographics.get("name", "Unknown"),
            "Patient_Age": demographics.get("age", "N/A"),
            "Patient_Gender": demographics.get("gender", "N/A"),
            "Patient_Location": demographics.get("location", "N/A"),
            "Final_Prediction": report["prediction"],
            "Vata_Probability": f"{report['v_prob']:.2f}%",
            "Pitta_Probability": f"{report['p_prob']:.2f}%",
            "Kapha_Probability": f"{report['k_prob']:.2f}%",
            "XAI_Report": report["explanation"]
        }

        if os.path.exists(EXCEL_FILE_PATH):
            try:
                df = pd.read_excel(EXCEL_FILE_PATH)
                df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            except Exception:
                df = pd.DataFrame([new_record])
        else:
            df = pd.DataFrame([new_record])

        df.to_excel(EXCEL_FILE_PATH, index=False)

        patient_info_packet = {
            "name": demographics.get("name", "Anonymous"),
            "age": demographics.get("age", "N/A"),
            "gender": demographics.get("gender", "N/A"),
            "location": demographics.get("location", "N/A")
        }

        patient_info_packet["language"] = lang_context

        log_patient_to_google_sheet(
            patient_info_packet,
            mapped_answers,
            report["prediction"],
            {
                "Vata": report["v_prob"],
                "Pitta": report["p_prob"],
                "Kapha": report["k_prob"]
            },
            report.get("lime_data", []),
            report.get("explanation", "")
        )

        return jsonify({"success": True, "redirect": "/result_dashboard"})

    except Exception as e:
        print("[CRITICAL APPLICATION EXCEPTION]:", e)
        return jsonify({"success": False, "error": str(e)})
@app.route('/result_dashboard')
def result_dashboard():
    global last_diagnostic_report
    if not last_diagnostic_report:
        return "No active records located. Please execute a diagnostic session first."
    return render_template('result5.html', report=last_diagnostic_report, timestamp=int(time.time()))

@app.route('/download_excel', methods=['GET'])
def download_excel():
    if os.path.exists(EXCEL_FILE_PATH):
        return send_file(EXCEL_FILE_PATH, as_attachment=True, download_name="Siddha_Prakriti_Database_Optimized.xlsx")
    return jsonify({"success": False, "error": "No records found in database."})

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)