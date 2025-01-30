# RAG Chatbot (Retrieval-Augmented Generation) with Flask

## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot using a fine-tuned Qwen model. The chatbot retrieves relevant context from indexed web pages and generates responses using a pre-trained language model. The Flask-based API serves chatbot responses, and users can interact with it via HTTP endpoints.

## Features
- Uses `Qwen2.5-3B-Instruct` fine-tuned with `BitsAndBytesConfig` for efficient quantization.
- Retrieves contextual information from web pages using a vector store index.
- Deploys a Flask API for chatbot interaction.
- Supports history tracking for chat conversations.

---
## Project Setup
### 1. Install Dependencies
Run the following command to install all required libraries:
```sh
!pip install transformers
!pip install pyngrok
!pip install flask
!pip install bs4
!pip install llama-index
!pip install bitsandbytes
!pip install sentencepiece
!pip install accelerate
!pip install llama_index.embeddings.huggingface
!pip install llama_index.llms.huggingface
```

### 2. Download and Quantize Model
The model and tokenizer are initialized with `AutoTokenizer` and `AutoModelForCausalLM`.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", quantization_config=quantization_config, torch_dtype=torch.float16)
```
Save the model locally for reuse:
```python
tokenizer.save_pretrained("./local_tokenizer")
model.save_pretrained("./local_model")
```

### 3. Run Flask Server
```sh
app = Flask(__name__)

# In-memory chat history
chat_history = []
message_id = 0  # For auto-incrementing IDs

class RAGChatbot:
    def __init__(self):

        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained("/content/local_tokenizer")
        self.model = AutoModelForCausalLM.from_pretrained("/content/local_model")
      

        self.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        self.llm = HuggingFaceLLM(
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=128,
        )

        # Settings configuration
        self.settings = Settings
        self.settings.llm = self.llm
        self.settings.embed_model = self.embed_model
        self.settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        self.settings.num_output = 50
        self.settings.context_window = 2048
        self.settings.generate_kwargs = {
            "do_sample": False,
            "temperature": 0.1,
            "max_new_tokens": 128
        }

    def scrape_website(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            print(f"Error scraping website: {e}")
            return None

    def create_index_from_websites(self, urls):
        texts = []
        for url in urls:
            text = self.scrape_website(url)
            if text:
                texts.append(text)

        # Create documents from scraped text
        documents = [Document(text = text) for text in texts]
        
        # Create index
        self.index = VectorStoreIndex.from_documents(
            documents,
            embed_model=self.settings.embed_model,
            node_parser=self.settings.node_parser,
            show_progress=True
        )

        # Create query engine
        self.query_engine = self.index.as_query_engine(
            llm=self.settings.llm,
            similarity_top_k=5,
            response_mode="compact",
            verbose=True,
            generate_kwargs=self.settings.generate_kwargs,
            context_window=self.settings.context_window,
            num_output=self.settings.num_output,
            show_progress=True
        )

    def get_response(self, query):
        response = self.query_engine.query(query)
        return str(response)

# Initialize chatbot
chatbot = RAGChatbot()

# Example websites - replace with your target URLs
websites = [
    "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
]
chatbot.create_index_from_websites(websites)

def add_to_history(role,query, response):
    global message_id
    message_id += 1
    message = {
        "id": message_id,
        "timestamp": datetime.utcnow().isoformat(),
        "role": role,
        "User_query": query,
        "generated_response": response
    }
    chat_history.append(message)
    return message

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Get response from chatbot
        response = chatbot.get_response(query)

        # Store system response
        system_message = add_to_history('system', query, response)

        return jsonify({
            "response": response,
            "message_id": system_message["id"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({"history": chat_history})

if __name__ == '__main__':
    
    ngrok.connect(5000)

    # Get the public URL
    tunnels = ngrok.get_tunnels()
    ngrok_url = tunnels[0].public_url
    print(f" * Public URL: {ngrok_url}")

    # Run Flask app
    app.run(port=5000)
```
The server will be accessible at `http://127.0.0.1:5000/`.

---
## Web Scraping and Indexing
The chatbot retrieves data from web pages and stores it in a vector index for efficient retrieval.
```python
chatbot.create_index_from_websites(["https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)"])
```

---
## API Usage
### 1. Chat with the Chatbot
Send a POST request to the `/chat` endpoint:
```sh
curl -X POST "http://127.0.0.1:5000/chat" -H "Content-Type: application/json" -d '{"query": "What is a transformer model?"}'
```
Expected response:
```json
{
  "response": "A transformer model is a deep learning model based on self-attention mechanisms...",
  "message_id": 1
}
```

### 2. Retrieve Chat History
Get chat history via the `/history` endpoint:
```sh
curl -X GET "http://127.0.0.1:5000/history"
```

---
## Testing Flask Endpoints
To validate the API endpoints run Test_Script_for_RAG_Chatbot.ipynb:
```python
import requests
import json

def chat_with_bot(query, url="https://c6f3-34-125-53-18.ngrok-free.app/"):
    """
    Send a query to the chatbot and get response
    """
    try:
        response = requests.post(
            f"{url}/chat",
            json={"query": query}
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}

def get_chat_history(url="https://c6f3-34-125-53-18.ngrok-free.app/"):
    """
    Retrieve chat history
    """
    try:
        response = requests.get(f"{url}/history")
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}

# Example usage
if __name__ == "__main__":
    # Example chat
    query = "what is attention machnism in transformers"
    print("\nSending query to chatbot...")
    response = chat_with_bot(query)
    print("User query", query)
    print("Response:", json.dumps(response, indent=2))

    # Example get history
    print("\nGetting chat history...")
    history = get_chat_history()
    print("History:", json.dumps(history, indent=2))
```






