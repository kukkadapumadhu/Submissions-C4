import gradio as gr
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Load model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    "suriya7/bart-finetuned-text-summarization"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "suriya7/bart-finetuned-text-summarization"
)

# -----------------------------
# Summarization function
# -----------------------------
def generate_summary(text):
    inputs = tokenizer(
        [text],
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=100,
        do_sample=False
    )

    return tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

# -----------------------------
# Export function
# -----------------------------
def export_summary(summary):
    file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".txt",
        mode="w",
        encoding="utf-8"
    )
    file.write(summary)
    file.close()
    return file.name

# -----------------------------
# Simple Blocks UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Text Summarizer")

    input_text = gr.Textbox(
        lines=8,
        label="Input Text",
        placeholder="Paste text to summarize..."
    )

    summarize_btn = gr.Button("Summarize")

    summary_output = gr.Textbox(
        lines=5,
        label="Summary"
    )

    export_btn = gr.Button("Export Summary")
    file_output = gr.File(label="Download Summary (.txt)")

    summarize_btn.click(
        fn=generate_summary,
        inputs=input_text,
        outputs=summary_output
    )

    export_btn.click(
        fn=export_summary,
        inputs=summary_output,
        outputs=file_output
    )

demo.launch()
