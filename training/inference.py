from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

def evaluate(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for i, (gen, ref) in enumerate(zip(generated, reference)):
        bleu = sentence_bleu([ref.split()], gen.split())
        rouge_scores = scorer.score(ref, gen)  # reference first, then generated text

        print(f"Sample {i+1}:")
        print(f"Generated: {gen}")
        print(f"Reference: {ref}")
        print(f"BLEU: {bleu:.4f}")
        print(f"ROUGE-L F1: {rouge_scores['rougeL'].fmeasure:.4f}\n")

if __name__ == "__main__":
    generated = [
        "you need a scholarship to study in Turkiye",
        "scholarships are available for international students in Turkiye",
        "You need a GPA above 3.5 to qualify.",
        "Application close on August 31st",
        "You can apply online through the official website"
    ]

    reference = [
        "Students must have a GPA over 3.5 to qualify for scholarships in Turkiye",
        "Deadline for scholarship applications is August 31st",
        "You can apply online through the official website",
        "Scholarships are available for international students in Turkiye",
        "You need a scholarship to study in Turkiye"
    ]

    evaluate(generated, reference)
