import torch
import gradio as gr
from transformers import AutoTokenizer
from model import TransformerSeq2Seq  # Import your Transformer model class

# Load Tokenizers
pseudocode_tokenizer = AutoTokenizer.from_pretrained("pseudocode_tokenizer.model")
code_tokenizer = AutoTokenizer.from_pretrained("code_tokenizer.model")

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerSeq2Seq()
model.load_state_dict(torch.load("p2c.pth", map_location=device))
model.to(device)
model.eval()

def predict(pseudocode):
    """Generate C++ code from given pseudocode."""
    inputs = pseudocode_tokenizer(pseudocode, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=512)
    generated_code = code_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_code

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, placeholder="Enter pseudocode here..."),
    outputs=gr.Code(language="cpp"),
    title="Pseudocode to C++ Converter",
    description="Convert pseudocode into C++ code using a Transformer-based Seq2Seq model.",
)

demo.launch()
