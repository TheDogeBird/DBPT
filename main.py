import os
import random
import torch
import transformers
import json
import sqlite3
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

MODELS_DIR = "models"
DLS_DIR = os.path.join(MODELS_DIR, "dls")


class ChatGPT:
    def __init__(self, config_path="config.json", model=None, tokenizer=None):
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
            self.model_engine = config.get("model_engine", "text-davinci-002")
            tokenizer_path = os.path.join(DLS_DIR, self.model_engine)
            model_path = os.path.join(DLS_DIR, self.model_engine)
            if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
                print("Downloading model...")
                tokenizer, model = self.download_model(self.model_engine)
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

        # Set up database connection
        self.db_path = config.get("db_path", "chat_history.db")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create table for chat history if it doesn't exist
        self.cursor.execute("CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                            "user_message TEXT, bot_message TEXT, timestamp TEXT)")

        # Create thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=2)

    def generate_response(self, user_message):
        input_ids = self.tokenizer.encode(user_message, return_tensors="pt")
        response = self.model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2,
                                       early_stopping=True)
        bot_message = self.tokenizer.decode(response[0], skip_special_tokens=True)

        # Check if there is a previous conversation in the database
        previous_conversation = self.cursor.execute("SELECT * FROM chat_history WHERE user_message = ? "
                                                    "ORDER BY timestamp DESC LIMIT 1",
                                                    (user_message,)).fetchone()
        if previous_conversation is not None:
            # If there is a previous conversation, use the context to generate the response
            context = previous_conversation[2]  # Column index for bot message
            input_ids = self.tokenizer.encode(context + user_message, return_tensors="pt")
            response = self.model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2,
                                           early_stopping=True)
            bot_message = self.tokenizer.decode(response[0], skip_special_tokens=True)

        # Store the current conversation in the database
        timestamp = str(datetime.datetime.now())
        self.cursor.execute("INSERT INTO chat_history (user_message, bot_message, timestamp) VALUES (?, ?, ?)",
                            (user_message, bot_message, timestamp))
        self.conn.commit()

        return bot_message.strip()

    def download_model(self, engine):
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(engine)
        model = transformers.GPT2LMHeadModel.from_pretrained(engine)
        return tokenizer, model

    def chat(self):
        print("Type 'exit' to end the conversation.")
        username = input("Enter your name: ")
        print(f"Hello {username}! Let's start the conversation.")

        # Download and load model
        model_engine = "text-davinci-002"
        tokenizer_path = os.path.join(DLS_DIR, model_engine)
        model_path = os.path.join(DLS_DIR, model_engine)
        if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
            print("Downloading model...")
            tokenizer, model = self.download_model(model_engine)
            tokenizer.save_pretrained(tokenizer_path)
            model.save_pretrained(model_path)
        else:
            print("Loading existing model...")
            tokenizer = transformers.GPT2Tokenizer.from_pretrained(tokenizer_path)
            model = transformers.GPT2LMHeadModel.from_pretrained(model_path)

        # Initialize chatbot
        chatbot = ChatGPT(model=model, tokenizer=tokenizer)

        # Define function for chatbot response generation
        def get_response():
            user_message = input(f"{username}: ")
            bot_message = chatbot.generate_response(user_message)
            print(f"Chatbot: {bot_message}\n")

        # Start chatting
        with ThreadPoolExecutor() as executor:
            while True:
                user_input = input("Press enter to continue or type 'exit' to end the conversation: ")
                if user_input.lower() == "exit":
                    print("Chat ended.")
                    break
                executor.submit(get_response)

        # Close database connection
        self.conn.close()

if __name__ == "__main__":
    chatbot = ChatGPT()
    chatbot.chat()
