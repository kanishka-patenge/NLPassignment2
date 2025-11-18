import os
import json
import time
import re
from typing import List, Dict

from dotenv import load_dotenv
from app.rag_grog import retrieve_scene_context
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------
# 1. Setup LLM (Groq) + Prompt
# ---------------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set in environment. Please set it before running evaluation.py")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
)

prompt = ChatPromptTemplate.from_template("""
You are a strict tutor for Shakespeare’s *Julius Caesar*.

You must answer ONLY using the provided context.
If the answer is not clearly in the context, say exactly:
"Context insufficient to answer this question."

<context>
{context}
</context>

Question: {question}

Answer:
""")


# ---------------------------------------------------------
# 2. Official 25 baseline questions (from assignment)
#    + 10 additional analytical/thematic questions
# ---------------------------------------------------------
EVAL_QUESTIONS: List[Dict] = [
    # --- Baseline 25 factual questions ---
    {
        "id": 1,
        "question": "How does Caesar first enter the play?",
        "ideal_answer": "In a triumphal procession; he has defeated the sons of his deceased rival, Pompey."
    },
    {
        "id": 2,
        "question": "What does the Soothsayer say to Caesar?",
        "ideal_answer": "Beware the Ides of March."
    },
    {
        "id": 3,
        "question": "What does Cassius first ask Brutus?",
        "ideal_answer": "He asks why Brutus has been so distant and contemplative lately."
    },
    {
        "id": 4,
        "question": "What does Brutus admit to Cassius?",
        "ideal_answer": "That he fears the people want Caesar to be king."
    },
    {
        "id": 5,
        "question": "What does Antony offer Caesar in the marketplace?",
        "ideal_answer": "He offers Caesar the crown."
    },
    {
        "id": 6,
        "question": "That night, which of the following omens are seen?",
        "ideal_answer": "All of the above: dead men walking, lions strolling in the marketplace, lightning."
    },
    {
        "id": 7,
        "question": "What finally convinces Brutus to join the conspirators?",
        "ideal_answer": "Forged letters planted by Cassius that appear to be from Roman citizens."
    },
    {
        "id": 8,
        "question": "Why does Calpurnia urge Caesar to stay home rather than appear at the Senate?",
        "ideal_answer": "She has had nightmares about his death and fears the omens."
    },
    {
        "id": 9,
        "question": "Why does Caesar ignore Calpurnia's warnings?",
        "ideal_answer": "Decius convinces him that Calpurnia has misinterpreted the dream and omens."
    },
    {
        "id": 10,
        "question": "What does Artemidorus offer Caesar in the street?",
        "ideal_answer": "A letter warning him about the conspiracy."
    },
    {
        "id": 11,
        "question": "What do the conspirators do at the Senate?",
        "ideal_answer": "They kneel around Caesar, stab him to death, and proclaim 'Tyranny is dead!'."
    },
    {
        "id": 12,
        "question": "What does Antony do when he arrives at Caesar's body?",
        "ideal_answer": "He weeps over Caesar’s body, shakes hands with the conspirators, and swears allegiance to Brutus for the moment."
    },
    {
        "id": 13,
        "question": "After the assassination of Caesar, which of the conspirators addresses the plebeians first?",
        "ideal_answer": "Brutus."
    },
    {
        "id": 14,
        "question": "What is Brutus's explanation for killing Caesar?",
        "ideal_answer": "He claims that Caesar was ambitious."
    },
    {
        "id": 15,
        "question": "What does Antony tell the crowd?",
        "ideal_answer": "That Brutus is an honorable man, that Caesar brought riches to Rome and turned down the crown, and that Caesar bequeathed the citizens money."
    },
    {
        "id": 16,
        "question": "What is the crowd's response to Antony's speech?",
        "ideal_answer": "Rage; they are moved to riot and chase the conspirators from the city."
    },
    {
        "id": 17,
        "question": "Who is Octavius?",
        "ideal_answer": "Caesar's adopted son and appointed heir."
    },
    {
        "id": 18,
        "question": "Octavius and Antony join together with whom?",
        "ideal_answer": "They join with Lepidus."
    },
    {
        "id": 19,
        "question": "Why do Brutus and Cassius argue?",
        "ideal_answer": "Because Brutus asked Cassius for money and Cassius withheld it."
    },
    {
        "id": 20,
        "question": "What news do Brutus and Cassius receive from Rome?",
        "ideal_answer": "That Portia is dead, many senators are dead, and the armies of Antony and Octavius are marching toward Philippi."
    },
    {
        "id": 21,
        "question": "What appears at Brutus's bedside in camp?",
        "ideal_answer": "Caesar's ghost."
    },
    {
        "id": 22,
        "question": "What does Cassius think has happened to his and Brutus's armies?",
        "ideal_answer": "He believes that they have been defeated by Antony and Octavius."
    },
    {
        "id": 23,
        "question": "What is Cassius's response to this situation?",
        "ideal_answer": "He has his servant stab him."
    },
    {
        "id": 24,
        "question": "What does Brutus do when he sees the battle is lost?",
        "ideal_answer": "He kills himself."
    },
    {
        "id": 25,
        "question": "What does Antony call Brutus at the end?",
        "ideal_answer": "He calls Brutus 'the noblest Roman of them all'."
    },

    # --- Additional 10 analytical / thematic questions ---
    {
        "id": 26,
        "question": "What are Brutus's internal conflicts as shown in his soliloquy in Act 2, Scene 1?",
        "ideal_answer": "He struggles between his love and friendship for Caesar and his fear that Caesar’s power will harm Rome; he decides to kill Caesar for the sake of the republic, not out of personal hatred."
    },
    {
        "id": 27,
        "question": "How does Cassius manipulate Brutus into joining the conspiracy?",
        "ideal_answer": "Cassius flatters Brutus, questions why Caesar should be more important than Brutus, and uses forged letters to make Brutus believe that the Roman people support action against Caesar."
    },
    {
        "id": 28,
        "question": "How does Antony use rhetoric and irony in his funeral speech to turn the crowd against the conspirators?",
        "ideal_answer": "He repeatedly calls Brutus 'an honorable man' while presenting evidence that contradicts Brutus's claim that Caesar was ambitious, such as Caesar refusing the crown and caring for the poor, thereby turning the crowd emotionally."
    },
    {
        "id": 29,
        "question": "Compare the leadership styles of Brutus and Cassius as shown in their argument in Act 4, Scene 3.",
        "ideal_answer": "Cassius is more emotional and concerned with personal loyalty and money, while Brutus insists on moral principles and public honor; Brutus appears more idealistic whereas Cassius is more pragmatic and resentful."
    },
    {
        "id": 30,
        "question": "What does Portia’s behavior reveal about her character and her relationship with Brutus?",
        "ideal_answer": "Portia is strong, intelligent, and determined; she wounds herself to prove her constancy and demands that Brutus share his secrets, revealing a relationship based on love but strained by Brutus’s secrecy."
    },
    {
        "id": 31,
        "question": "How does the play explore the theme of fate versus free will through Caesar and Cassius?",
        "ideal_answer": "Caesar tends to dismiss omens and warnings, suggesting confidence in fate or his own invincibility, while Cassius insists that men are masters of their own fate; yet both are destroyed, suggesting a complex interplay between destiny and human choice."
    },
    {
        "id": 32,
        "question": "In what ways do the plebeians in Act 3, Scene 2 illustrate the theme of mob mentality?",
        "ideal_answer": "They quickly shift their loyalty from Brutus to Antony, are easily swayed by emotional appeals, and become violent and irrational, showing how the crowd can be manipulated and become dangerous."
    },
    {
        "id": 33,
        "question": "Why can Brutus be seen as both a patriot and a tragic hero?",
        "ideal_answer": "He kills Caesar believing he is saving Rome from tyranny, showing patriotism, but his idealism, poor judgment, and refusal to listen to advice lead to his downfall, fitting the pattern of a tragic hero."
    },
    {
        "id": 34,
        "question": "How does Shakespeare use omens and supernatural events to build tension in the play?",
        "ideal_answer": "Storms, strange sights in the streets, Calpurnia’s dream, and Caesar’s ghost all foreshadow tragedy and create a sense that the natural order has been disturbed by political actions."
    },
    {
        "id": 35,
        "question": "What does the resolution of the play suggest about the cost of political violence?",
        "ideal_answer": "The deaths of Caesar, Brutus, Cassius, and others, along with civil war, show that political violence leads to chaos and personal tragedy rather than the idealized freedom the conspirators hoped for."
    }
]


# ---------------------------------------------------------
# 3. Utility: write evaluation.json
# ---------------------------------------------------------
def write_evaluation_json(path: str = "evaluation.json"):
    data = [{"question": q["question"], "ideal_answer": q["ideal_answer"]} for q in EVAL_QUESTIONS]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[info] Wrote {len(data)} questions to {path}")


# ---------------------------------------------------------
# 4. Context limiting (same idea as API)
# ---------------------------------------------------------
def build_limited_context(docs, max_tokens: int) -> str:
    parts = []
    token_count = 0

    for d in docs:
        text = d["chunk"].replace("\n", " ")
        tokens = text.split()
        n = len(tokens)

        if token_count + n > max_tokens:
            remaining = max_tokens - token_count
            if remaining > 0:
                parts.append(" ".join(tokens[:remaining]))
            break

        parts.append(" ".join(tokens))
        token_count += n

        if token_count >= max_tokens:
            break

    return "\n\n".join(parts)


# ---------------------------------------------------------
# 5. Simple RAGAs-like metrics: relevancy + faithfulness
# ---------------------------------------------------------
def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def answer_relevancy(system_answer: str, ideal_answer: str) -> float:
    gt_tokens = tokenize(ideal_answer)
    ans_tokens = tokenize(system_answer)
    if not gt_tokens or not ans_tokens:
        return 0.0
    gt_set = set(gt_tokens)
    overlap = sum(1 for t in ans_tokens if t in gt_set)
    return overlap / len(gt_tokens)


def faithfulness(system_answer: str, context_text: str) -> float:
    ans_tokens = tokenize(system_answer)
    ctx_tokens = set(tokenize(context_text))
    if not ans_tokens:
        return 0.0
    grounded = sum(1 for t in ans_tokens if t in ctx_tokens)
    return grounded / len(ans_tokens)


# ---------------------------------------------------------
# 6. Main evaluation loop
# ---------------------------------------------------------
def run_evaluation():
    results = []
    total = len(EVAL_QUESTIONS)
    rel_sum = 0.0
    faith_sum = 0.0

    for item in EVAL_QUESTIONS:
        qid = item["id"]
        question = item["question"]
        ideal = item["ideal_answer"]

        print(f"\n=== [{qid}/{total}] Q: {question} ===")

        # 1) Retrieve context
        docs = retrieve_scene_context(question, top_k=10)

        # 2) Build limited context for LLM
        MAX_CONTEXT_TOKENS = 2500
        context_text = build_limited_context(docs, MAX_CONTEXT_TOKENS)

        # 3) Invoke LLM
        messages = prompt.format_messages(
            context=context_text,
            question=question,
        )
        t0 = time.time()
        response = llm.invoke(messages)
        dt = time.time() - t0

        answer = response.content.strip()
        print(f"[answer] {answer}")
        print(f"[time] {dt:.2f}s")

        # 4) Compute metrics
        rel = answer_relevancy(answer, ideal)
        faith = faithfulness(answer, context_text)
        rel_sum += rel
        faith_sum += faith

        print(f"[metrics] answer_relevancy={rel:.3f} | faithfulness={faith:.3f}")

        # 5) Save detailed record
        results.append({
            "id": qid,
            "question": question,
            "ideal_answer": ideal,
            "system_answer": answer,
            "answer_relevancy": rel,
            "faithfulness": faith,
        })

    avg_rel = rel_sum / total if total else 0.0
    avg_faith = faith_sum / total if total else 0.0

    print("\n=== Overall Metrics ===")
    print(f"Average Answer Relevancy: {avg_rel:.3f}")
    print(f"Average Faithfulness:    {avg_faith:.3f}")

    # 6) Save results
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "overall": {
                "avg_answer_relevancy": avg_rel,
                "avg_faithfulness": avg_faith,
                "num_questions": total,
            },
            "details": results,
        }, f, indent=2, ensure_ascii=False)

    print("[info] Saved detailed results to evaluation_results.json")


if __name__ == "__main__":
    # 1) Write evaluation.json (for assignment requirement)
    write_evaluation_json("evaluation.json")

    # 2) Run evaluation loop
    run_evaluation()
