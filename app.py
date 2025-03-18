# YouTubeTranscriptAPI Imports
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, VideoUnavailable, TooManyRequests, TranscriptsDisabled, NoTranscriptAvailable
from youtube_transcript_api.formatters import TextFormatter

# Flask Imports
from flask import Flask, jsonify, request, send_from_directory, render_template, redirect

# NLTK Imports
import nltk

# Other Imports
import os
import sys

# Summarizer Import (Our Another File: summarizer.py)
from summarizer import gensim_summarize, spacy_summarize, nltk_summarize, sumy_lsa_summarize, sumy_luhn_summarize, sumy_text_rank_summarize

# Waitress Import for Serving at Heroku
from waitress import serve

import google.generativeai as genai

# Configure Google Generative AI with your API key
genai.configure(api_key="AIzaSyAW6gQppI6lizfFad9Tlia_3mFIPbeDDlU")


def generate_gemini_summary(transcript_text, summary_percentage):
        base_prompt = """Welcome, Video Summarizer! Your task is to distill the essence of a given YouTube video transcript into a concise summary. Your summary should capture the key points and essential information. Let's dive into the provided transcript and extract the vital details for our audience."""
        
        # Adjust the prompt based on the summary percentage
        prompt_map = {
            10: "Summarize the video to 10% of its original content, focusing only on the most essential points.",
            20: "Summarize the video to 20% of its original content, highlighting the key points.",
            30: "Summarize the video to 30% of its original content, covering the main takeaways.",
            40: "Summarize the video to 40% of its original content, providing the key points with some additional details.",
            50: "Summarize the video to 50% of its original content, with moderate detail and key points.",
            60: "Summarize the video to 60% of its original content, providing more detailed information.",
            70: "Summarize the video to 70% of its original content, including most of the content in detail.",
            80: "Summarize the video to 80% of its original content, covering most points with detailed information.",
            90: "Summarize the video to 90% of its original content, providing a thorough summary with nearly all points covered.",
            100: "Summarize the video to 100% of its original content, providing a full and detailed summary with as much information as possible."
        }
        
        selected_prompt = base_prompt + " " + prompt_map.get(int(summary_percentage), "")
        
        # Call Gemini Pro API
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(selected_prompt + transcript_text)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Error generating summary with Gemini Pro: {e}")



def create_app():
    app = Flask(__name__)

    # "Punkt" download before nltk tokenization
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print('Downloading punkt')
        nltk.download('punkt', quiet=True)

    # "Wordnet" download before nltk tokenization
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print('Downloading wordnet')
        nltk.download('wordnet')

    # "Stopwords" download before nltk tokenization
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print('Downloading Stopwords')
        nltk.download("stopwords", quiet=True)

    @app.route('/summarize/', methods=['GET'])
    def transcript_fetched_query():
        video_id = request.args.get('id')
        percent = request.args.get('percent')
        choice = request.args.get('choice')

        if video_id and percent and choice:
            choice_list = ["gensim-sum", "spacy-sum", "nltk-sum", "sumy-lsa-sum", "sumy-luhn-sum", "sumy-text-rank-sum","gemini-pro"]
            if choice in choice_list:
                try:
                    formatter = TextFormatter()
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    formatted_text = formatter.format_transcript(transcript).replace("\n", " ")

                    num_sent_text = len(nltk.sent_tokenize(formatted_text))
                    select_length = int(num_sent_text * (int(percent) / 100))

                    if select_length > 0:
                        if num_sent_text > 1:
                            summary = None
                            if choice == "gensim-sum":
                                summary = gensim_summarize(formatted_text, percent)
                            elif choice == "spacy-sum":
                                summary = spacy_summarize(formatted_text, percent)
                            elif choice == "nltk-sum":
                                summary = nltk_summarize(formatted_text, percent)
                            elif choice == "sumy-lsa-sum":
                                summary = sumy_lsa_summarize(formatted_text, percent)
                            elif choice == "sumy-luhn-sum":
                                summary = sumy_luhn_summarize(formatted_text, percent)
                            elif choice == "sumy-text-rank-sum":
                                summary = sumy_text_rank_summarize(formatted_text, percent)
                            elif choice == "gemini-pro":
                                summary = generate_gemini_summary(formatted_text, percent)

                            num_sent_summary = len(nltk.sent_tokenize(summary))

                            response_list = {
                                'processed_summary': summary,
                                'length_original': len(formatted_text),
                                'length_summary': len(summary),
                                'sentence_original': num_sent_text,
                                'sentence_summary': num_sent_summary
                            }

                            return jsonify(success=True, message="Subtitles for this video were fetched and summarized successfully.", response=response_list), 200

                        else:
                            return jsonify(success=False, message="Subtitles are not formatted properly for this video. Unable to summarize.", response=None), 400

                    else:
                        return jsonify(success=False, message="Number of lines in the subtitles of your video is not enough to generate a summary.", response=None), 400

                except VideoUnavailable:
                    return jsonify(success=False, message="VideoUnavailable: The video is no longer available.", response=None), 400
                except TooManyRequests:
                    return jsonify(success=False, message="TooManyRequests: YouTube is receiving too many requests from this IP. Wait until the ban on server has been lifted.", response=None), 500
                except TranscriptsDisabled:
                    return jsonify(success=False, message="TranscriptsDisabled: Subtitles are disabled for this video.", response=None), 400
                except NoTranscriptAvailable:
                    return jsonify(success=False, message="NoTranscriptAvailable: No transcripts are available for this video.", response=None), 400
                except NoTranscriptFound:
                    return jsonify(success=False, message="NoTranscriptFound: No transcripts were found.", response=None), 400
                except Exception as e:
                    print(e)
                    sys.stdout.flush()
                    return jsonify(success=False, message="Some error occurred. Contact the administrator if it is happening too frequently.", response=None), 500
            else:
                return jsonify(success=False, message="Invalid Choice: Please create your request with the correct choice.", response=None), 400
        else:
            return jsonify(success=False, message="Please request the server with your arguments correctly.", response=None), 400

    @app.route('/favicon.ico')
    def favicon():
        return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.png', mimetype='image/vnd.microsoft.icon')

    @app.route('/')
    def root_function():
        return render_template('root.html')

    @app.route('/web/')
    def summarizer_web():
        return render_template('web.html')
    
    @app.route('/gemini/')
    def summarizer_gemini():
        return render_template('gemini.html')

    @app.route('/api/')
    def summarizer_api_info_route():
        return render_template('api.html')

    @app.before_request
    def enforce_https_in_heroku():
        if 'DYNO' in os.environ:
            if request.headers.get('X-Forwarded-Proto') == 'http':
                url = request.url.replace('http://', 'https://', 1)
                code = 301
                return redirect(url, code=code)

    return app


if __name__ == '__main__':
    flask_app = create_app()
    flask_app.run(debug=True, port=5000)
    # Uncomment the following line if using Waitress for production
    # serve(flask_app, host='0.0.0.0', port=80)