import os
import json
import random
import sqlite3

from datetime import datetime
import torch
import transformers
from flask import Flask, request, render_template, jsonify
from flask_compress import Compress
from flask_caching import Cache
from threading import Thread


# Set up Flask app
app = Flask(__name__)
cache = Cache(app, config={"CACHE_TYPE": "simple"})

# Define constants
DLS_DIR = "models/dls"
BERT_DIR = "models/bert"


class ChatGPT:
    def generate_response(self, user_input):
        input_ids = self.tokenizer.encode(user_input, return_tensors="pt")
        input_ids = input_ids.to(self.device)

        # Generate bot response
        bot_response_ids = self.model.generate(
            input_ids,
            max_length=self.max_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.92,
            top_k=0,
        )

        bot_response = self.tokenizer.decode(bot_response_ids[0], skip_special_tokens=True)

        # Save last bot response
        self.last_bot_response = bot_response

        # Add bot response to conversation history
        self.add_to_conversation(user_input, bot_response)

        # Return bot response or empty string if bot response is None
        return bot_response or ''

    def __init__(self, config_path="config.json", model=None, tokenizer=None, db_path="newdb.db"):
        self.get_response_async = None
        self.last_bot_response = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = config_path
        self.db_path = db_path
        self.app = Flask(__name__)
        self.compress = Compress(self.app)
        app = Flask(__name__)
        Compress(app)
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.model_engine = self.config.get("model_engine", "gpt2-medium")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 1024)
        self.stop = self.config.get("stop", ["\n", "Human:"])
        self.max_new_tokens = self.config.get("max_new_tokens", 20)
        self.log_conversations = self.config.get("log_conversations", True)
        self.flows = self.config.get("conversational_flows", {})
        self.repetition_penalty = self.config.get("repetition_penalty", 1.0)

        # Load config file
        with open(config_path) as f:
            config = json.load(f)

        # Load API key
        self.api_key = config.get("api_key")

        # Set up GPT-2 model and tokenizer
        if model is not None and tokenizer is not None:
            self.tokenizer = tokenizer
            self.model = model
        else:
            self.model_engine = config.get("model_engine", "gpt2-medium")
            tokenizer_path = os.path.join(DLS_DIR, self.model_engine)
            model_path = os.path.join(DLS_DIR, self.model_engine)
            if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
                print("Downloading model...")
                tokenizer, model = transformers.GPT2Tokenizer.from_pretrained(self.model_engine), \
                                   transformers.GPT2LMHeadModel.from_pretrained(self.model_engine)
                tokenizer.save_pretrained(tokenizer_path)
                model.save_pretrained(model_path)
            else:
                print("Loading existing model...")
                tokenizer = transformers.GPT2Tokenizer.from_pretrained(tokenizer_path)
                model = transformers.GPT2LMHeadModel.from_pretrained(model_path)

            self.tokenizer = tokenizer
            self.model = model

        # Set seed for reproducibility
        self.seed = config.get("seed", 42)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set up database
        self.db_path = db_path
        self.create_tables()

        # Set up BERT for sentiment analysis
        self.bert_tokenizer, self.bert_model = self.download_bert()

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up conversational flows
        self.conversational_flows = config.get("conversational_flows", [])

        # Set up max_length for generation
        self.max_length = config.get("max_length", 50)

       # set up routes
        @self.app.route("/")
        def home():
            return render_template("index.html")

        def get_response_async(self, message):
            response = self.generate_response(message)
            self.responses[message] = response

        @self.app.route('/get_response/<message_id>')
        @self.compress.compressed()
        def get_response_by_id(message_id):
            response = self.responses.get(message_id)
            if response is not None:
                return jsonify({'response': response})
            else:
                return jsonify({'response': 'Response not found.'})

        def generate_response(self, user_input):
            input_ids = self.tokenizer.encode(user_input, return_tensors="pt")
            input_ids = input_ids.to(self.device)

            # Generate bot response
            bot_response_ids = self.model.generate(
                input_ids,
                max_length=self.max_tokens,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.92,
                top_k=0,
            )

            bot_response = self.tokenizer.decode(bot_response_ids[0], skip_special_tokens=True)

            # Save last bot response
            self.last_bot_response = bot_response

            # Add bot response to conversation history
            self.add_to_conversation(user_input, bot_response)

            # Return bot response or empty string if bot response is None
            return bot_response

        def run(self):
            print("Chatbot started, type 'exit' to exit.")
            while True:
                user_input = input("You: ")

                # Exit if user types "exit"
                if user_input.strip().lower() == "exit":
                    break

                # Generate response and save to database
                response = self.generate_response(user_input)
                last_user_message = self.get_last_user_message()
                last_user_sentiment = self.classify_sentiment(last_user_message) if last_user_message else ""
                sentiment = self.classify_sentiment(response)
                self.save_to_database(last_user_message, response, sentiment)

                # Print response
                print("Chatbot:", response)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # run Flask app
        self.app.run(threaded=True)

    def get_last_user_message(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT user_input FROM chat_history WHERE bot_response IS NOT NULL ORDER BY timestamp DESC LIMIT 1")
        last_user_message = c.fetchone()
        conn.close()
        return last_user_message[0] if last_user_message else None

    def download_models(self):
        # Set up GPT-2 model and tokenizer
        if self.model is not None and self.tokenizer is not None:
            self.tokenizer = self.tokenizer
            self.model = self.model
        else:
            self.model_engine = self.config.get("model_engine", "gpt2-medium")
            tokenizer_path = os.path.join(DLS_DIR, self.model_engine)
            model_path = os.path.join(DLS_DIR, self.model_engine)
            if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
                print("Downloading model...")
                tokenizer, model = transformers.GPT2Tokenizer.from_pretrained(self.model_engine), \
                                   transformers.GPT2LMHeadModel.from_pretrained(self.model_engine)
                tokenizer.save_pretrained(tokenizer_path)
                model.save_pretrained(model_path)
            else:
                print("Loading existing model...")
                tokenizer = transformers.GPT2Tokenizer.from_pretrained(tokenizer_path)
                model = transformers.GPT2LMHeadModel.from_pretrained(model_path)

            self.tokenizer = tokenizer
            self.model = model

        # Set up BERT for sentiment analysis
        self.bert_tokenizer, self.bert_model = self.download_bert()

    def download_bert(self):
        bert_engine = "bert-base-uncased"
        tokenizer_path = os.path.join(BERT_DIR, bert_engine)
        model_path = os.path.join(BERT_DIR, bert_engine)
        if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
            print("Downloading BERT...")
            tokenizer = transformers.AutoTokenizer.from_pretrained(bert_engine)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(bert_engine)
            tokenizer.save_pretrained(tokenizer_path)
            model.save_pretrained(model_path)
        else:
            print("Loading existing BERT...")
            tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)

        model.to(self.device)
        return tokenizer, model

    def create_tables(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                     (user_input TEXT, bot_response TEXT, sentiment TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        conn.commit()
        conn.close()

    def process_user_input(self, user_input):
        # Clean user input
        user_input = user_input.strip().lower()
        # Remove punctuation and special characters
        user_input = "".join(c for c in user_input if c.isalnum() or c.isspace())

        return user_input

    def classify_sentiment(self, text):
        inputs = self.bert_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        sentiment = "positive" if probs[0][1] > 0.5 else "negative"
        return sentiment

    def save_to_database(self, user_message, bot_response, sentiment):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = str(int(datetime.timestamp(datetime.now())))
        c.execute("INSERT INTO chat_history VALUES (?, ?, ?, ?)",
                  (user_message, bot_response, sentiment, timestamp))
        conn.commit()
        conn.close()

    def get_conversation_flow_response(self, user_input, last_user_message, last_user_sentiment):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        query = "SELECT * FROM conversation_flows WHERE trigger_phrase=?"
        c.execute(query, (last_user_message,))
        result = c.fetchone()
        if result is None:
            query = "SELECT * FROM conversation_flows WHERE trigger_phrase IS NULL ORDER BY RANDOM() LIMIT 1"
            c.execute(query)
            result = c.fetchone()
        response = result[1]
        if response == "input":
            response = self.generate_response(user_input)
        elif response == "repeat":
            response = self.generate_response(last_user_message)
        elif response == "same_topic":
            response = self.generate_response(last_user_message + " " + user_input)
        elif response == "switch_topic":
            response = self.generate_response(user_input)
        conn.close()

        return response

    def run(self):
        print("Chatbot started, type 'exit' to exit.")
        while True:
            # Get user input from web interface
            user_input = request.args.get("msg")

            # Exit if user types "exit"
            if user_input.strip().lower() == "exit":
                break

            # Generate response and save to database
            response = self.generate_response(user_input)
            last_user_message = self.get_last_user_message()
            last_user_sentiment = self.classify_sentiment(last_user_message) if last_user_message else ""
            sentiment = self.classify_sentiment(response)
            self.save_to_database(last_user_message, response, sentiment)

            # Return response as JSON object
            response_dict = {"response": response}
            return response_dict

    def add_to_conversation(self, user_input, bot_response):
        pass


if __name__ == "__main__":
    chatbot = ChatGPT(config_path="config.json")
    if chatbot.device.type == "cuda":
        torch.cuda.empty_cache()
    chatbot.run()
