import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os

def load_model(run_dir: str):
    """Load the base model and apply the LoRA adapter."""
    # Full path to the run directory in the mounted volume
    full_run_dir = os.path.join("/runs", run_dir)
    adapter_path = os.path.join(full_run_dir, "lora-out")
    
    # Debug information
    st.info(f"""
    Checking paths:
    - Run directory: {full_run_dir}
    - Adapter path: {adapter_path}
    """)
    
    # Verify adapter exists
    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        st.error(f"adapter_config.json not found in {adapter_path}")
        st.error(f"Contents of {full_run_dir}:")
        st.error(os.listdir(full_run_dir))
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

    # Load the base model
    base_model_name = "mistralai/Mistral-7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="cpu",  # Use CPU instead of CUDA
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # Load the tokenizer from the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

    
    config = PeftConfig.from_pretrained(adapter_path)

    # Resize the token embeddings to match the LoRA adapter's vocabulary size
    print(f"Resizing token embeddings to match the LoRA adapter (vocab_size=32002)...")
    model.resize_token_embeddings(32002,mean_resizing=False)  # Resize to match the LoRA adapter

    # Load the LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path,config=config)    
    

    return model, tokenizer

def generate_answer(query: str, run_dir: str):
    """Generate answer for a given question and query."""
    # Format the input properly
    fullinput = """ [INST] Analyze the sentiment of the following text: 
                    {instruction} [/INST]
                """.format(instruction=query)
    
    try:
        # Load the model and tokenizer
        model, tokenizer = load_model(run_dir)
        
        # Tokenize the input
        inputs = tokenizer.encode(fullinput, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cpu")
        attention_mask = inputs.ne(tokenizer.pad_token_id).int()  # Create attention mask
        
        # Generate output using the model
        outputs = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=128, use_cache=True)
        
        # Decode the generated output
        raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the actual result (e.g., "1") from the output
        # Assume the result is always after the `[INST]` formatting
        processed_answer = raw_answer.split("[/INST]")[-1].strip()  # Remove input and keep the response
        
        return processed_answer
    except Exception as e:
        st.error(f"Error during generation: {str(e)}")
        return f"Error: {str(e)}"


def appmain():
    st.set_page_config(
        page_title="Sentiment Analysis Interface",
        page_icon="🎭",
        layout="centered"
    )

    st.title("💭 How does it sound in Tunisian?")
    st.write("Enter text to analyze its sentiment using our fine-tuned Mistral model.")

    # Input section
    with st.form("sentiment_form"):
        text_input = st.text_area(
            "Enter text to analyze",
            height=150,
            placeholder="Type or paste your text here..."
        )
        
        run_name = st.text_input(
            "Model Run Name",
            placeholder="Enter the training run name (e.g., axo-2024-01-16-12-34-56-ab)",
            help="This is the identifier of your trained model run"
        )

        submit_button = st.form_submit_button("Analyze Sentiment")

    if submit_button and text_input and run_name:
        with st.spinner("Analyzing sentiment... Note: this may take up to 15 minutes on the first prompt loading the model."):
            try:
                result = generate_answer(text_input, run_name)

                
                
                # Format result
                if result == "1":
                    result = "Positive"
                elif result == "-1":
                    result = "Negative"
                elif result == "0":
                    result = "Neutral"
                else:
                    result = result
                
                st.success("Analysis Complete!")
                st.subheader("Results")
                st.write(result)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please check your run name and try again.")
    
    # Add some usage instructions
    with st.expander("ℹ️ Usage Instructions"):
        st.markdown("""
        1. Enter the text you want to analyze in the text area
        2. Provide the run name of your trained model (from `.last_run_name` file)
        3. Click 'Analyze Sentiment' to get the results
        
        The run name can be found in the `.last_run_name` file after training,
        or by running `modal volume ls training-runs-vol`
        """)

if __name__ == "__main__":
    appmain()