import os
import json
import random
import sqlite3
from datetime import datetime
import torch
import transformers


# Define constants
DLS_DIR = "models/dls"
BERT_DIR = "models/bert"


class ChatGPT:
    def __init__(self, config_path="config.json", model=None, tokenizer=None, db_path="chathis.db"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = config_path
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

    def get_last_user_message(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT user_input FROM chat_history WHERE bot_response IS NOT NULL ORDER BY timestamp DESC LIMIT 1")
        last_user_message = c.fetchone()
        conn.close()
        return last_user_message[0] if last_user_message else None

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

    def generate_response(self, user_input):
        # Clean user input
        user_input = self.process_user_input(user_input)

        # Classify sentiment of user message
        last_user_message = self.get_last_user_message()
        if last_user_message is None:
            last_user_sentiment = ""
        else:
            last_user_sentiment = self.classify_sentiment(last_user_message)

        # Save user message to database
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO chat_history (user_input, sentiment, timestamp) VALUES (?, ?, ?)",
                  (user_input, last_user_sentiment, timestamp))
        conn.commit()
        conn.close()

        # Generate bot response
        input_ids = self.tokenizer.encode(user_input, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        max_length = random.randint(50, 100)
        sample_outputs = self.model.generate(
            input_ids,
            do_sample=True,
            max_length=max_length,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )

        bot_response = self.tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

        # Save bot response to database
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO chat_history (bot_response, sentiment, timestamp) VALUES (?, ?, ?)",
                  (bot_response, last_user_sentiment, timestamp))
        conn.commit()
        conn.close()

        return bot_response

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
            user_input = input("You: ")
            if user_input.strip().lower() == "exit":
                break
            response = self.generate_response(user_input)
            print("Bot:", response)

if __name__ == "__main__":
    chatbot = ChatGPT(config_path="config.json")
    if chatbot.device.type == "cuda":
        torch.cuda.empty_cache()
    chatbot.run()
