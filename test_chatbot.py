from pipeline.retrieval import ask_all_models

question = "Summarize case CASE0001"

answers, docs = ask_all_models(question)

print("\n==============================")
print("LLM Outputs")
print("==============================\n")

for model, answer in answers.items():

    print("=" * 80)
    print(model)
    print("=" * 80)
    print(answer)
    print()

print("\n==============================")
print("Retrieved Documents")
print("==============================\n")

for doc in docs:

    print(doc.metadata)
    print(doc.page_content[:500])
    print("-" * 80)