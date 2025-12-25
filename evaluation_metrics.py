# ================================================================
# evaluation_metrics.py
# ================================================================
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric

def build_metrics(custom_llm, threshold, task_type="common"):
    """
    Build evaluation metrics based on task type.
    task_type: "common", "translation", or "summarization"
    """
    common_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]

    # Standard rubric scale
    common_rubric = [
        Rubric(score_range=(0, 4), expected_outcome="Subpar"),
        Rubric(score_range=(5, 6), expected_outcome="Marginal"),
        Rubric(score_range=(7, 8), expected_outcome="Good"),
        Rubric(score_range=(9, 10), expected_outcome="Excellent"),
    ]

    metrics = {}

    # ================================================================
    # 🧩 1️⃣ COMMON METRICS
    # ================================================================
    metrics_common = {
        "Factual Fidelity" : FaithfulnessMetric(
            threshold=threshold,
            model=custom_llm,
            include_reason=True
        ),

        "Content Importance & Relevance" : ContextualRelevancyMetric(
            threshold=threshold,
            model=custom_llm,
            include_reason=True
        ),

        "Professional Tone & Grammar" : GEval(
            name="Professional Tone & Grammar",
            model=custom_llm,
            evaluation_params=common_params,
            criteria="""Evaluate whether the actual_output maintains a professional analyst tone appropriate for the Malaysian equity research market.
            Rules:
            - Language should reflect professional standards: formal, respectful, and objective.
            - Avoid overly casual expressions, slang, contractions, emojis, speculative tone, or emotionally charged language.
            - The tone should be neutral and insight-driven rather than promotional.

            Reject or downgrade if:
            - The output includes informal, vague, or speculative phrases (e.g., “maybe,” “I think,” “kind of”).
            - The tone is too conversational or emotionally biased.
            - Formatting or grammar undermines the output’s professional quality.""",

            evaluation_steps=[
                "Scan for casual or unprofessional expressions such as slang, emojis, unnecessary contractions (e.g., 'gonna', 'wanna'), or filler phrases (e.g., 'you know', 'kind of').",
                "Ensure the tone is objective and neutral — avoid speculative or emotionally charged language (e.g., 'we're very excited', 'this is amazing').",
                "Check grammar, sentence structure, and formatting consistency to ensure the text looks polished and professional.",
                "Check for the correct application of localized financial formatting, specifically the use of 'RM' and standard unit abbreviations like 'bn' and 'm'.",
                "Do not penalize informality in strategic quotes regarding corporate actions, but any emotional or promotional quotes (e.g., 'we are delighted', 'proud to announce') must result in a significant score reduction.",
                "Assign a score between 0.0 and 1.0 based on the following scale: Score ≥ 0.9: Output demonstrates an impeccable professional tone, with formal, precise, and neutral language fully aligned to a financial news domain.; Score ≥ 0.7: Output maintains a generally professional tone with only minor lapses; Score ≥ 0.5: Output shows noticeable issues with tone, such as informal expressions, vague or speculative language, or inconsistent terminology, but remains somewhat serviceable in a professional context.; Score < 0.5: Output is unprofessional in tone, with casual language, misuse of terminology, emotional bias, or poor grammar that undermines credibility."
            ],
            threshold=threshold,
            rubric=common_rubric,
        ),

        "Executive Writing Quality" : GEval(
            name="Executive Writing Quality",
            model=custom_llm,
            evaluation_params=common_params,
            criteria="""Evaluate whether the 'actual_output' is optimized for a senior investment professional. 
    
            1. Structural Integrity: The summary must be exactly 3 to 6 sentences. 
            2. The News Lead: The opening sentence must contain the most material corporate development (Bottom Line Up Front).
            3. Information Density: Prioritize precise figures (RM, %, bn, m) over descriptive adjectives. 
            5. No Fluff: Penalize any promotional language or emotional quotes (e.g., 'we are delighted', 'exceptional results').""",

            evaluation_steps=[
                "Check if the opening sentence delivers the 'key news' immediately without background filler.",
                "Check if the summary is using impactful sentences ensures the message is easily and quickly understood without ambiguity.",
                "Penalize outputs that are too brief or overly verbose (below 3 or above 6 sentences).", 
                "Penalize any promotional language or emotional quotes (e.g., 'we are delighted', 'exceptional results').",
                "Assign a score: - Score ≥ 0.8: Highly polished, clear, concise, professional with top-notch executive quality - Score ≥ 0.5: Clear and professional, but may bury the lead or lack sufficient numeric density.- Score < 0.5: Wordy, promotional, or fails the 3-5 sentence structural constraint.",
            ],
            threshold=threshold,
            rubric = common_rubric,
        ),
    }

    # ================================================================
    # 🌐 2️⃣ TRANSLATION METRICS
    # ================================================================
    metrics_translation = {
        "Translation Tone Adherence": GEval(
            name = "Translation Accuracy & Completeness",
            criteria = """
            Evaluate whether the 'actual_output' is a **highly accurate and complete translation** of the 'input'.
            Rules:
            - The output must **preserve the full semantic meaning** of the original input.
            - The output must **not omit any meaningful content**, including facts, figures, qualifiers, or nuance.
            - The output must **not introduce** any new content, elaborations, interpretations, or assumptions not present in the input.
            - Phrasing can differ, but **meaning, intent, and information coverage** must remain intact.
            - Cultural adaptation is acceptable **only if** it does not distort the factual or thematic meaning.
            Reject or downgrade if:
            - The translation **omits key sentences, figures, or arguments** from the input.
            - The output includes **fabricated or inferred content** not explicitly supported by the original.
            - Any part of the meaning is **distorted or misrepresented** due to mistranslation or oversimplification.""",
            evaluation_steps=[
            "Compare the 'actual_output' to the 'input' on a point-by-point or sentence-by-sentence basis.",
            "Verify that every piece of information (including data, claims, and subtleties) from the 'input' is present in the 'actual_output'.",
            "Identify any information in the 'actual_output' that was not originally in the 'input'.",
            "Check for any shifts in meaning, misinterpretations, or mistranslations of general phrases.",
            "A perfect score means the translation is a 1:1 semantic and informational match to the original.",
            "Assign a score between 0.0 and 1.0 based on the following scale: Score ≥ 0.9: Translation is highly accurate and complete; fully preserves meaning, intent, and nuance, with no omissions, additions, or distortions.; Score ≥ 0.7: Translation is generally accurate and complete; minor omissions or slight rephrasings that do not materially affect the overall meaning.; Score ≥ 0.5: Translation has noticeable shortcomings; omissions, inaccuracies, or minor additions create partial loss of fidelity or completeness.; Score < 0.5: Translation is poor; major omissions, distortions, or added content make it unreliable and unfaithful to the original."],
            model=custom_llm,
            threshold=threshold,
            evaluation_params=common_params,
            rubric=common_rubric,
        ),

        "Translation Financial Entity Localization": GEval(
            name = "Translation Financial Entity Localization",
            criteria="""Evaluate if key financial entities in the 'actual_output' (translation) are correctly localized for a professional audience in the target language.
            - **Stock Tickers & Identifiers (e.g., AAPL, ISINs):** These must be preserved exactly as they are. They must NOT be translated.
            - **Acronyms & Institutions (e.g., Fed, ECB, PBoC):** These must be translated to their commonly accepted name or acronym in the target language (e.g., 'the Fed' might become 'la Fed' in French), not a clumsy literal translation.
            - **Standard Financial Jargon (e.g., 'headwinds', 'yield curve'):** These must be translated into the standard, professional equivalent term used in the target market.
            - **Currency Symbols & Formatting:** Currency symbols ($, €, ¥) and values must be accurate and follow local conventions.
 
            In addition, if a predefined localization map is provided {context}, verify that branded or key terms (e.g., report titles, strategy names) in the 'input' are localized exactly as expected in the 'actual_output'.
            - Apply strict matching to mapped terms from source language to other target as provided.""",
            evaluation_steps=[
            "Scan the 'input' to identify all specific financial entities: stock tickers, currency symbols, institutional acronyms (Fed, ECB, etc.), and key financial jargon.",
            "For each entity identified, locate its counterpart in the 'actual_output' and apply the following checks:",
            "   a. **Tickers/Identifiers:** Verify they are identical to the source. Penalize any modification.",
            "   b. **Acronyms/Institutions:** Verify they match the common, standard name used by finance professionals in the target locale. Penalize literal or awkward translations.",
            "   c. **Jargon:** Verify the term used is the correct professional equivalent in the target language.",
            "   d. **Currencies:** Confirm the symbol and number are correct and the format is natural for the locale.",
            "If a localization map is provided in the context:",
            "   - Check that all key terms in the map are accurately translated from source to target as specified.",
            "   - Penalize incorrect or inconsistent mappings, especially for product or brand-sensitive terms.",
            "   - Reward exact matches to mapped expectations.",
            "Assign a score between 0.0 and 1.0 based on the following scale: Score ≥ 0.9: All financial entities (tickers, acronyms, jargon, currencies, mapped terms) are correctly localized or preserved as per conventions and mappings, with no errors.; Score ≥ 0.7: Localization is generally correct; minor lapses in one or two entities, but nothing that significantly harms professional readability or correctness.; Score ≥ 0.5: Localization has noticeable issues; several inconsistencies, literal translations, or formatting errors make the text partially unsuitable for professionals.; Score < 0.5: Localization is poor; major errors in preserving tickers, acronyms, jargon, or mappings make the translation unprofessional and unreliable."],
            model=custom_llm,
            threshold=threshold,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
            rubric=common_rubric,
        ),
    }


    # ================================================================
    # 🧠 3️⃣ SUMMARIZATION METRICS
    # ================================================================
    metrics_summarization = {
        "Summary Coherence & Flow": GEval(
            name="Summary Coherence & Flow",
            model=custom_llm,
            evaluation_params=common_params,
            criteria="""Evaluate whether the output follows the established hierarchy of a professional daily news watch summary. The information must begin with a definitive lead sentence that captures the primary corporate development or financial event. Subsequent sentences must provide supporting data and strategic context. Strategic Quote Integration: In accordance with the defined Quote Policy, the summary should ideally include a quote that conveys material strategic facts or future outlook to provide authoritative weight to the news. The structure should avoid chronological storytelling and instead focus on a 'Key Point + Substantiation' model.
            Reject or downgrade if:
            - Ideas are randomly ordered or grouped in a confusing manner.
            - Transitions between ideas are missing or unclear.
            - The output contradicts itself or disrupts logical flow.
            - Structuring is present but feels unnatural or forced.""",
            evaluation_steps=[
            "Verify if the opening sentence contains the most material financial news or corporate action, such as a merger, earnings result, or contract win.",
            "Assess if the supporting sentences follow a logical descent from high-impact news to operational details.",
            "Penalize jumbled, incoherent, contradictory, or disjointed outputs, regardless of format.",
            "Do not penalize for tone, factual precision, or strategic selection — focus only on idea structure and clarity of flow.",
            "Assign a score based on the scale: Score ≥ 0.9: Exceptional logical structure that prioritizes strategic impact. Score < 0.5: Jumbled presentation of facts without a clear cause and effect relationship."
            ],
            threshold=threshold,
            rubric=common_rubric,
        ),
    }

    # ================================================================
    # TASK ROUTING
    # ================================================================
    if task_type == "common":
        return metrics_common

    elif task_type == "translation":
        return {**metrics_common, **metrics_translation}

    elif task_type == "summarization":
        return {**metrics_common, **metrics_summarization}
    elif task_type == "recovery_summarization":
        return {
            "Executive Writing Quality": metrics_common["Executive Writing Quality"],
            "Summary Coherence & Flow": metrics_summarization["Summary Coherence & Flow"]
        }
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
