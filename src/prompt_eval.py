from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
# from deepeval.models import AzureOpenAIModel
from dotenv import load_dotenv
import os
from openai import OpenAI

from ticket_db import TicketDB


title = "Generic webwork aliases may clash with other plugins"
description = "Some web work actions have commands that have very generic aliases, such as ""Update"", ""Move"" and ""Synch"" etc. These aliases have a good chance to clash with the action name or aliases used by other plugins.  These aliases should be given a name specific to the plugin, such as ""UpdateGreenHopperDropBoard""... Alternatively, the bang syntax can be used in place of command aliases, such as ""DropBoardAction!update"". In this case, there is no need to define aliases for commands. (However, still need to define alias as an empty String; otherwise, a NPE will be thrown when parsing the xml.)"
expected = 5

db = TicketDB()
similar_tickets = db.get_similar_tickets(title, description, 6)
similar_tickets = [ticket for ticket in similar_tickets if title not in ticket['title']]

# Construct the first prompt
prompt1 ="Given past tickets and their story points, estimate the story point for a new ticket.\n\n"
prompt1+= "Similar past tickets:\n"

for idx, ticket in enumerate(similar_tickets, 1):
    short_text = ticket['title'].split(".")[0]
    prompt1 += f"{idx}. [Title]: \"{short_text}\" → Story Points: {ticket['storypoint']}\n"

prompt1 += "\nNew ticket:\n"
prompt1 += f"Title: \"{title}\"\n"
prompt1 += f"Description: \"{description}\"\n"

prompt1 += "\nQuestion: What is the most likely story point estimate?"
prompt1 += "\nProvide a brief and professional response, including the estimated story point value and a short justification for the estimate.\n"
prompt1 += "\nOn the last line of your message, only return the number of story points.\n"


# Construct the second prompt
prompt2 = "Analyze the complexity of the following new task compared to previous tickets. Use the examples to guide your reasoning and determine a suitable story point using the Fibonacci sequence: 1, 2, 3, 5, 8, 13.\n\n"
prompt2 += "Previous ticket examples:\n"

for idx, ticket in enumerate(similar_tickets, 1):
    short_text = ticket['title'].split(".")[0]
    prompt2 += f"{idx}. Title: \"{short_text}\" — Story Points: {ticket['storypoint']}\n"

prompt2 += "\nEvaluate this new ticket:\n"
prompt2 += f"Title: \"{title}\"\n"
prompt2 += f"Description: \"{description}\"\n"

prompt2 += "\nProvide a brief and professional response, including the estimated story point value and a short justification for the estimate.\n"
prompt2 += "\nOn the last line of your message, only return the number of story points.\n"


# Construct the third prompt
prompt3 = "Use the table below of past Jira tickets to assess the story point for a new task. Base your estimate on similarity in effort and scope.\n\n"
prompt3 += "Past Tickets:\n"
prompt3 += "-------------------------------\n"
prompt3 += "| Title                      | SP |\n"
prompt3 += "-------------------------------\n"

for ticket in similar_tickets:
    short_title = ticket['title'].split(".")[0][:25].ljust(25)
    sp = str(ticket['storypoint']).rjust(2)
    prompt3 += f"| {short_title} | {sp} |\n"

prompt3 += "-------------------------------\n\n"
prompt3 += f"New Ticket:\nTitle: \"{title}\"\nDescription: \"{description}\"\n\n"
prompt3 += "What is the most appropriate story point? Respond professionally with reasoning, and return the final number on the last line.\n"


load_dotenv()
api_key = os.getenv("API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# openai_model = AzureOpenAIModel(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))


respose1 = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are an expert agile project manager."},
        {"role": "user", "content": prompt1},
    ]
)
response1 = respose1.choices[0].message.content.strip()
last_line1 = response1.split("\n")[-1].strip()
last_estimation1 = int(''.join(filter(str.isdigit, response1.split()[-1])))

response2 = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are an expert agile project manager."},
        {"role": "user", "content": prompt2},
    ]
)
response2 = response2.choices[0].message.content.strip()
last_line2 = response2.split("\n")[-1].strip()
last_estimation2 = int(''.join(filter(str.isdigit, response2.split()[-1])))

response3 = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are an expert agile project manager."},
        {"role": "user", "content": prompt3},
    ]
)
response3 = response3.choices[0].message.content.strip()
last_line3 = response3.split("\n")[-1].strip()
last_estimation3 = int(''.join(filter(str.isdigit, response3.split()[-1])))

context = []
for ticket in similar_tickets:
    context.append(f"{ticket['title']} → {ticket['storypoint']}")

breakpoint()


test_cases = [
    LLMTestCase(
        input=prompt1,
        actual_output=last_estimation1,
        context=context,
        expected_output=str(expected)
    ),
    LLMTestCase(
        input=prompt2,
        actual_output=last_estimation2,
        context=context,
        expected_output=str(expected)
    ),
    LLMTestCase(
        input=prompt3,
        actual_output=last_estimation3,
        context=context,
        expected_output=str(expected)
    )
]

relevance_metric = AnswerRelevancyMetric(model=openai_model)
faithfulness_metric = FaithfulnessMetric(model=openai_model)

relevance_metric.measure(test_cases)
faithfulness_metric.measure(test_cases)


for i, test_case in enumerate(test_cases, 1):
    print(f"\n🧪 Prompt Variant {i}")
    print(f"Estimate: {test_case.actual_output.strip().splitlines()[-1]}")
    print("Answer Relevance:", test_case.scores[relevance_metric.name])
    print("Faithfulness:", test_case.scores[faithfulness_metric.name])
    print("-----")
