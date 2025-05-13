import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
import re


from ticket_db import TicketDB
from prompt_builder import PromptBuilder

load_dotenv()
api_key = os.getenv("API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
db = TicketDB()

second_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(second_model_id)
model = AutoModelForCausalLM.from_pretrained(
    second_model_id,
    device_map="auto",
    torch_dtype="auto"
)


last_title = ""
last_description = ""
last_estimation = 0
last_similar_tickets = []


def check_toxicity(text):
    global tokenizer, model
    prompt = f"""Check if the following message contains toxic, offensive, or inappropriate content.
    Reply only with "TOXIC" or "SAFE".

    Message: {text}
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip().upper()
        return result == "TOXIC"
    except Exception as e:
        messagebox.showerror("Toxicity Check Failed", str(e))
        return True


def get_estimation():
    global last_title, last_description, last_similar_tickets, last_estimation

    title = title_entry.get()
    description = description_text.get("1.0", tk.END).strip()

    if not title:
        messagebox.showwarning("Missing Title", "Please provide at least a title.")
        return
    
    if not description:
        description = ""

    similar_tickets = db.get_similar_tickets(title, description)
    prompt = PromptBuilder.construct_storypoint_prompt(similar_tickets, title, description)

    last_title = title
    last_description = description
    last_similar_tickets = similar_tickets

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert agile project manager."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        content = response.choices[0].message.content.strip()
        last_line = content.split("\n")[-1].strip()

        try:
            last_estimation = int(''.join(filter(str.isdigit, content.split()[-1])))
            if last_line.isnumeric():
                last_estimation = int(last_line)
        except:
            last_estimation = 0


        if check_toxicity(content) or check_toxicity(title) or check_toxicity(description):
            chat_window.insert(tk.END, "\nDeepSeek:\nSorry, couldn't help with that.\n0\n", "assistant")
        else:
            chat_window.insert(tk.END, f"\nYou:\nTitle: {title}\nDescription: {description}\n", "user")
            chat_window.insert(tk.END, f"\nDeepSeek:\n{content}\n", "assistant")
        chat_window.see(tk.END)

    except Exception as e:
        messagebox.showerror("Error", str(e))


def give_feedback(direction):
    global last_estimation, last_title, last_description, last_similar_tickets, model, tokenizer
    if not last_title or not last_description or not last_similar_tickets:
        messagebox.showwarning("No Estimate Yet", "Submit a ticket first.")
        return
    
    if last_estimation == 0:
        messagebox.showwarning("Invalid previous input", "Submit a valid ticket first.")
        return

    try:
        feedback_prompt = f"""
You're helping tune a prompt for estimating Jira story points.

The original instruction was to estimate a story point based on past tickets.

Please write a new instruction to ask for a slightly {"higher" if direction == "higher" else "lower"} estimate, if justified.
The original story point estimate was {last_estimation}.
Keep it brief and professional.
"""

        feedback_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": feedback_prompt}]
        )

        new_instruction = feedback_response.choices[0].message.content.strip()

        # messages = [
        #     {"role": "system", "content": "You are an expert llm prompt engineer."},
        #     {"role": "user", "content": feedback_prompt},
        # ]

        # formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
        # outputs = model.generate(
        #     **inputs,
        #     max_new_tokens=200,
        #     temperature=0.0,
        #     repetition_penalty=1.2
        # )

        # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # response = generated_text.split("<|assistant|>")[1]
        # if " instruction:" in response:
        #     response = response.split("Modified instruction:")[-1].strip()
        # breakpoint()
        # last_estimation = re.findall(r'\d+', response)[-1]


        base_prompt = new_instruction + "\n\nSimilar tickets:\n"
        for idx, ticket in enumerate(last_similar_tickets, 1):
            base_prompt += f"{idx}. [Title]: \"{ticket['title']}\" → Story Points: {ticket['storypoint']}\n"

        base_prompt += f"\nNew ticket:\nTitle: \"{last_title}\"\nDescription: \"{last_description}\"\n"
        base_prompt += "\nWhat is the most likely story point estimate?"
        base_prompt += "\nProvide a brief and professional response, including the estimated story point value and a short justification for the estimate.\n"
        base_prompt += "\nOn the last line of your message, only return the number of story points.\n"


        chat_window.insert(tk.END, f"\nYou (Feedback: {direction}): Adjust the estimate {direction}.\n", "user")


        adjusted_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert agile project manager."},
                {"role": "user", "content": base_prompt},
            ]
        )

        new_estimate = adjusted_response.choices[0].message.content.strip()
        last_line = new_estimate.split("\n")[-1].strip()
        try:
            last_estimation = int(''.join(filter(str.isdigit, new_estimate.split()[-1])))
            if last_line.isnumeric():
                last_estimation = int(last_line)
        except:
            last_estimation = 0

        if check_toxicity(new_estimate):
            chat_window.insert(tk.END, "\nDeepSeek (Adjusted):\nSorry, couldn't help with that.\n", "assistant")
        else:
            chat_window.insert(tk.END, f"\nDeepSeek (Adjusted):\n{new_estimate}\n", "assistant")
        chat_window.see(tk.END)

    except Exception as e:
        messagebox.showerror("Error", str(e))



root = tk.Tk()
root.title("Jira Ticket Estimator (Chat + Feedback)")

tk.Label(root, text="Ticket Title:").pack(anchor="w")
title_entry = tk.Entry(root, width=60)
title_entry.pack(pady=5)

tk.Label(root, text="Ticket Description:").pack(anchor="w")
description_text = tk.Text(root, height=5, width=60)
description_text.pack(pady=5)

tk.Button(root, text="Estimate Story Points", command=get_estimation).pack(pady=10)

chat_window = scrolledtext.ScrolledText(root, width=70, height=20, wrap=tk.WORD)
chat_window.pack(padx=10, pady=10)
chat_window.tag_configure("user", foreground="blue", font=("Arial", 10, "bold"))
chat_window.tag_configure("assistant", foreground="darkgreen", font=("Arial", 10))

feedback_frame = tk.Frame(root)
feedback_frame.pack(pady=5)

tk.Label(feedback_frame, text="Was the estimate okay?").pack(side=tk.LEFT)
tk.Button(feedback_frame, text="Higher", command=lambda: give_feedback("higher")).pack(side=tk.LEFT, padx=10)
tk.Button(feedback_frame, text="Lower", command=lambda: give_feedback("lower")).pack(side=tk.LEFT, padx=10)

root.mainloop()