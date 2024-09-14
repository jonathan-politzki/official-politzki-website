from flask import Flask, request, jsonify
from flask_cors import CORS
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}})  # Be cautious with '*' in production

# Define the model IDs
primary_model_id = "gpt2-medium"
fallback_model_id = "gpt2"

try:
    print(f"Attempting to load {primary_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(primary_model_id)
    model = AutoModelForCausalLM.from_pretrained(primary_model_id)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    
    print(f"Successfully loaded {primary_model_id}")

except Exception as e:
    print(f"Error loading {primary_model_id}: {e}")
    print(f"Falling back to {fallback_model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(fallback_model_id)
    model = AutoModelForCausalLM.from_pretrained(fallback_model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

def generate_response(prompt, max_length=300):
    full_prompt = f"""
As an AI assistant, provide a detailed and informative answer to the following question:

Question: {prompt}

Detailed Answer:"""

    response = pipe(full_prompt, max_new_tokens=max_length, num_return_sequences=1, do_sample=True, temperature=0.5)[0]['generated_text']
    
    # Extract the generated answer
    answer = response.split("Detailed Answer:")[-1].strip()
    return answer

#test 
@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Server is running!'})

#not test
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data['question']
    answer = generate_response(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, port=5000)