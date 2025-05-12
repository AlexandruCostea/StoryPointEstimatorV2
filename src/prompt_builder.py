class PromptBuilder:
    def construct_storypoint_prompt(similar_tickets, new_title, new_description):
        body = "Analyze the complexity of the following new task compared to previous tickets. Use the examples to guide your reasoning and determine a suitable story point using the Fibonacci sequence: 1, 2, 3, 5, 8, 13.\n\n"
        body += "Previous ticket examples:\n"

        for idx, ticket in enumerate(similar_tickets, 1):
            short_text = ticket['title'].split(".")[0]
            body += f"{idx}. Title: \"{short_text}\" — Story Points: {ticket['storypoint']}\n"

        body += "\nEvaluate this new ticket:\n"
        body += f"Title: \"{new_title}\"\n"
        body += f"Description: \"{new_description}\"\n"

        body += "\nProvide a brief and professional response, including the estimated story point value and a short justification for the estimate.\n"
        body += "\nOn the last line of your message, only return the number of story points.\n"

        return body