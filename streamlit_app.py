"""
Streamlit UI for Code Plagiarism Detection
Interactive web app to test the fine-tuned CodeBERT model
"""

import streamlit as st
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from pathlib import Path
import numpy as np


# Page configuration
st.set_page_config(
    page_title="Code Plagiarism Detector",
    page_icon="🔍",
    layout="wide"
)

# Constants
HF_MODEL_ID = "OmarHassan44/plagiarism-detector"
MAX_LENGTH = 255

@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer (cached)"""
    
    model = RobertaForSequenceClassification.from_pretrained(HF_MODEL_ID)
    tokenizer = RobertaTokenizerFast.from_pretrained(HF_MODEL_ID)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer, device


def tokenize_code_pair(code1: str, code2: str, tokenizer):
    """
    Tokenize two code snippets using the same method as training.
    
    During training, both code snippets are tokenized together as a list
    and then flattened to create a single concatenated sequence.
    """
    tokenized_inputs = tokenizer(
        [code1, code2],  # Pass as list, not separate args
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=MAX_LENGTH,
    )
    # Flatten to concatenate both sequences (same as training)
    tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"].flatten().unsqueeze(0)
    tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"].flatten().unsqueeze(0)
    
    return tokenized_inputs


def predict_plagiarism(code1: str, code2: str, model, tokenizer, device):
    """
    Predict plagiarism similarity between two code snippets
    
    Returns:
        - prediction: 0 (not plagiarism) or 1 (plagiarism)
        - probability: plagiarism probability [0-1]
        - confidence: confidence score
    """
    
    # Tokenize
    inputs = tokenize_code_pair(code1, code2, tokenizer)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1).item()
        plagiarism_prob = probs[0][1].item()
        confidence = torch.max(probs).item()
    
    return prediction, plagiarism_prob, confidence


# Example code pairs
EXAMPLES = {
    "Identical Code": {
        "code1": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr""",
        "code2": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr"""
    },
    "Similar Logic, Different Style": {
        "code1": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr""",
        "code2": """def sort_array(numbers):
    length = len(numbers)
    for idx in range(length):
        for jdx in range(0, length-idx-1):
            if numbers[jdx] > numbers[jdx+1]:
                temp = numbers[jdx]
                numbers[jdx] = numbers[jdx+1]
                numbers[jdx+1] = temp
    return numbers"""
    },
    "Completely Different": {
        "code1": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr""",
        "code2": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1"""
    }
}


# Main UI
def main():
    st.title("🔍 Code Plagiarism Detector")
    st.markdown("### Powered by Fine-Tuned CodeBERT")
    
    st.markdown("""
    This tool uses a fine-tuned CodeBERT model to detect code plagiarism by analyzing 
    the semantic similarity between two code snippets.
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer, device = load_model_and_tokenizer()
    
    st.success(f"✓ Model loaded successfully! Using device: {device}")
    
    # Sidebar
    st.sidebar.header("📋 Examples")
    selected_example = st.sidebar.selectbox(
        "Load an example:",
        ["Custom Input"] + list(EXAMPLES.keys())
    )
    
    # Initialize session state for tracking last selected example
    if "last_example" not in st.session_state:
        st.session_state.last_example = None
    
    # Initialize or update code snippets when example changes
    if "code1_input" not in st.session_state:
        st.session_state.code1_input = ""
    if "code2_input" not in st.session_state:
        st.session_state.code2_input = ""
    
    # Update code snippets when a new example is selected
    if selected_example != "Custom Input":
        if st.session_state.last_example != selected_example:
            st.session_state.code1_input = EXAMPLES[selected_example]["code1"]
            st.session_state.code2_input = EXAMPLES[selected_example]["code2"]
            st.session_state.last_example = selected_example
    else:
        if st.session_state.last_example != "Custom Input":
            st.session_state.code1_input = ""
            st.session_state.code2_input = ""
            st.session_state.last_example = "Custom Input"
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Interpretation Guide")
    st.sidebar.markdown("""
    - **Similarity > 85%**: Likely plagiarism
    - **Similarity < 85**: Likely not plagiarism
    
    The model considers semantic meaning,
    not just syntax.
    """)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Code Snippet 1")
        code1 = st.text_area(
            "Enter first code snippet:",
            value=st.session_state.code1_input,
            height=300,
            key="code1_input"
        )
    
    with col2:
        st.subheader("Code Snippet 2")
        code2 = st.text_area(
            "Enter second code snippet:",
            value=st.session_state.code2_input,
            height=300,
            key="code2_input"
        )
    
    # Analyze button
    if st.button("🔬 Analyze for Plagiarism", type="primary", use_container_width=True):
        if not code1.strip() or not code2.strip():
            st.error("Please enter both code snippets!")
        else:
            with st.spinner("Analyzing..."):
                prediction, plagiarism_prob, confidence = predict_plagiarism(
                    code1, code2, model, tokenizer, device
                )
            
            # Results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Metrics in columns
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                result_text = "PLAGIARISM DETECTED" if prediction == 1 else "NOT PLAGIARISM"
                result_color = "🔴" if prediction == 1 else "🟢"
                st.metric("Verdict", f"{result_color} {result_text}")
            
            with metric_col2:
                st.metric("Similarity Score", f"{plagiarism_prob*100:.1f}%")
            
            with metric_col3:
                st.metric("Confidence", f"{confidence*100:.1f}%")
            
            # Progress bar for similarity
            st.markdown("### Similarity Visualization")
            st.progress(plagiarism_prob)
            
            # Color-coded interpretation
            if plagiarism_prob > 0.85:
                st.error("⚠️ **High Similarity**: These code snippets are very likely plagiarized.")
            elif 0.5 <plagiarism_prob < 0.85:
                st.warning("⚠️ **Moderate Similarity**: These code snippets show some similarities.")
            else:
                st.success("✓ **Low Similarity**: These code snippets appear to be different.")
            
            # Detailed breakdown
            with st.expander("🔍 Detailed Analysis"):
                st.markdown(f"""
                **Probability Distribution:**
                - Not Plagiarism: {(1-plagiarism_prob)*100:.2f}%
                - Plagiarism: {plagiarism_prob*100:.2f}%
                
                **Model Confidence:** {confidence*100:.2f}%
                
                **Note:** This model analyzes semantic similarity, meaning it can detect
                plagiarism even when variable names or code structure have been changed.
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    Built with CodeBERT | Fine-tuned on SemanticCloneBench & GPTCloneBench
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
