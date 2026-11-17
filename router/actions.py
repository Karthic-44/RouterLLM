from router.clients import ask
from router.retrieval import passage_block

NOTE = "Use these passages if they help. If they do not contain the answer, use your own knowledge."

TASKS = {
    "math": 'Solve this math problem. Think step by step, then end with a line "Answer: <number>".',
    "mc": 'Answer this multiple choice question. Think briefly, then end with a line "Answer: <letter>".',
    "fact": 'Answer this question in a few words. End with a line "Answer: <answer>".',
    "open": 'Answer this question. If it needs working out, show short steps. End with a line "Answer: <final answer>".',
}


def split_action(action):
    return action.split("_")[0], action.endswith("_rag")


def build_prompt(kind, question, passages=None):
    parts = []
    if passages:
        parts.append(NOTE + "\n" + passage_block(passages))
    parts.append(TASKS[kind])
    parts.append("Question: " + question)
    return "\n\n".join(parts)


def run(action, kind, question, passages, clients, models, max_tokens):
    key, rag = split_action(action)
    prompt = build_prompt(kind, question, passages if rag else None)
    return ask(clients[key], models[key], prompt, max_tokens)
