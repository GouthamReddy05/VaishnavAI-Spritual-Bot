from flask import Flask, render_template, request, jsonify
from ask_llm import search_faiss, ask_llm, build_context
from deep_translator import GoogleTranslator
from flask_cors import CORS
from langdetect import detect, DetectorFactory
import os
DetectorFactory.seed = 0 
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get('message')
    language = data.get('language')
    scripture = data.get('scripture') 

    language_codes = {
    'english': 'en',
    'telugu': 'te',
    'hindi': 'hi'
}

    if not user_query or user_query.strip() == "":
        return jsonify({'response': '⚠️ Please enter something to ask.'})

    try:
        detected_lang = detect(user_query)
    except:
        detected_lang = 'en'  
    
    if detected_lang != 'en':
        user_query = GoogleTranslator(source='auto', target='en').translate(user_query)

    if user_query:
        retrieved_verses = search_faiss(user_query, tok_k=20, scripture=scripture)

        if retrieved_verses:
            context = build_context(retrieved_verses, scripture)
        response = ask_llm(user_query, scripture, context) 
        

    tar_lang = language_codes.get(language.lower())

    if(tar_lang != 'en'):
        translator = GoogleTranslator(source='auto', target=tar_lang)
        response = translator.translate(response)

    return jsonify({'response': response})

if __name__ == '__main__':
    # port = int(os.environ.get("PORT", 7860))
    app.run(debug=True)
