import os
import gradio as gr
import textwrap
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import openai
import random
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "cyberagent/open-calm-1b"

openai.api_key = "my api key"
system_message = {"role": "system", "content": "You are a helpful assistant."}

theme = gr.themes.Monochrome(
    primary_hue="green",
    secondary_hue="green",
    neutral_hue="slate"
)

def generate(text, temperature=0.7, top_p=1.0, repetition_penalty=1.0, max_new_tokens=256):
  if text == '':
    return '入力できていません'

  repetition_penalty = float(repetition_penalty) # must be

  inputs = tokenizer(text, return_tensors="pt")
  inputs.to(device)
  with torch.no_grad():
    tokens = model.generate(
        **inputs, 
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
  tokens.detach().cpu()
  return tokenizer.decode(tokens[0], skip_special_tokens=True)

with gr.Blocks(theme=theme, analytics_enabled=False) as demo:
  with gr.Tabs():
    with gr.TabItem("cyberagent-open-calm"):
      with gr.Row():
        with gr.Column(scale=2):
          with gr.Column():
            text = gr.Textbox(
                placeholder="テキストを入力してください。",
                label="Text",
                elem_id="q-input",
            )
            output = gr.Textbox(
                placeholder="",
                lines=5,
                label="Infer",
                elem_id="q-output",
            )
        with gr.Column(scale=1):
          with gr.Accordion("Parameters", open=True):
            temperature = gr.Slider(
                label="Temperature",
                value=0.7,
                minimum=0.01,
                maximum=1.00,
                step=0.01,
                interactive=True,
            )
            top_p = gr.Slider(
                label="Top P",
                value=1.0,
                minimum=0.01,
                maximum=1.00,
                step=0.01,
                interactive=True,
            )
            repetition_penalty = gr.Slider(
                label="Repetition penalty",
                value=1.00,
                minimum=1.00,
                maximum=2.00,
                step=0.01,
                interactive=True,
            )
            max_new_tokens = gr.Slider(
                label="Maximum length",
                value=64,
                minimum=1,
                maximum=256,
                step=1,
                interactive=True,
            )
      with gr.Row():
        submit = gr.Button("Generate", variant="primary")

      submit.click(
        generate,
        inputs=[text, temperature, top_p, repetition_penalty, max_new_tokens],
        outputs=[output],
      )
    
    with gr.TabItem("open-api-key"):
      chatbot = gr.Chatbot()
      msg = gr.Textbox()
      clear = gr.Button("Clear")

      state = gr.State([])

      def user(user_message, history):
          return "", history + [[user_message, None]]

      def bot(history, messages_history):
          user_message = history[-1][0]
          bot_message, messages_history = ask_gpt(user_message, messages_history)
          messages_history += [{"role": "assistant", "content": bot_message}]
          history[-1][1] = bot_message
          time.sleep(1)
          return history, messages_history

      def ask_gpt(message, messages_history):
          messages_history += [{"role": "user", "content": message}]
          response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=messages_history
          )
          return response['choices'][0]['message']['content'], messages_history

      def init_history(messages_history):
          messages_history = []
          messages_history += [system_message]
          return messages_history

      msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
          bot, [chatbot, state], [chatbot, state]
      )

      clear.click(lambda: None, None, chatbot, queue=False).success(init_history, [state], [state])



model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto', low_cpu_mem_usage=True, torch_dtype=torch.float16, cache_dir="./")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="./")
demo.queue(concurrency_count=1).launch(debug=True, show_error=True)