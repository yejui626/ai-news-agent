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
        # GEval(
        #     name="Factual Fidelity",
        #     model=custom_llm,
        #     evaluation_params=common_params,
        #     criteria="""Evaluate if the key points presented in the 'actual_output' are factually and semantically faithful to the 'input'.
        #     The output must not introduce any information, claims, or figures that are not supported by the source text.
        #     While the output is not expected to be exhaustive, every piece of information it *does* contain must be accurate and correctly interpreted.
        #     The primary goal is to assess the absence of hallucinations and misinterpretations, not the completeness of the summary.""",
        #     evaluation_steps=[
        #     "For every claim, data point, or figure present in 'actual_output', verify it matches the 'input'.",
        #     "Identify any fabricated information or 'hallucinations' in 'actual_output' that are not present in the 'input'.",
        #     "Check for misinterpretations or distortions of the original meaning in the points that were included.",
        #     "Assess if the necessary context for understanding the points *included* in 'actual_output' is preserved, preventing them from being misleading.",
        #     "Acknowledge that 'actual_output' is a selective summary and should NOT be penalized for omitting information from the 'input', as long as the omissions do not distort the meaning of the included points.",
        #     "Assign a score between 0.0 and 1.0 based on the following scale: Score ≥ 0.9: Output is factually and semantically very faithful with no hallucinations or misinterpretations.; Score ≥ 0.7: Output is mostly faithful with only minor or borderline issues that don’t significantly affect understanding.; Score ≥ 0.5: Output contains some factual or semantic issues, but they are not severe; caution is advised.; Score < 0.5: Output contains serious hallucinations, misinterpretations, or fabricated facts that distort meaning."],
        #     threshold=threshold,
        #     rubric=common_rubric,
        # ),

        "Content Importance & Relevance" : ContextualRelevancyMetric(
            threshold=threshold,
            model=custom_llm,
            include_reason=True
        ),
        # GEval(
        #     name="Content Importance & Relevance",
        #     model=custom_llm,
        #     evaluation_params=common_params,
        #     criteria="""Evaluate whether the actual_output prioritizes and highlights the most strategically important information for the intended audience (e.g., investors, executives, analysts).
        #     Rules:
        #     - The output should emphasize actionable insights, forward-looking statements, key financial drivers, significant risks, and major opportunities.
        #     - Less critical background details, historical descriptions, or purely descriptive information should be filtered out or minimized.
        #     - Prioritization should reflect investor relevance and decision-making value.
        #     - The output must not be cluttered with low-value or tangential content that detracts from strategic clarity.
        #     Reject or downgrade if:
        #     - Key strategic elements (e.g., major risks or opportunities) mentioned in the input are omitted or ignored.
        #     - The output includes excessive non-strategic or background information that reduces focus on critical points.
        #     - Actionable or forward-looking insights are missing or insufficiently emphasized.
        #     - The summary appears unfocused or overwhelmed by irrelevant details.""",

        #     evaluation_steps=[
        #         "Analyze the input to identify key strategic components: actionable insights, forward-looking statements, key drivers, significant risks, and major opportunities.",
        #         "Review the actual_output to assess coverage and emphasis on these strategic components.",
        #         "Check if non-strategic or purely descriptive background details are minimized or excluded.",
        #         "Determine if the actual_output provides a clear prioritization of information that aligns with investor decision needs.",
        #         "Penalize if critical strategic points are missing or if irrelevant details dominate the output.",
        #         "Score higher when the output delivers a concise, focused summary highlighting the most valuable strategic content.",
        #         "Assign a score between 0.0 and 1.0 based on the following scale: Score ≥ 0.9: Output strongly prioritizes strategic, actionable, investor-relevant content with exceptional clarity.; Score ≥ 0.7: Output covers most strategic points and minimizes irrelevant content, though a few areas could be improved.; Score ≥ 0.5: Output shows some attempt at prioritizing strategic content but contains notable omissions or distractions.; Score < 0.5: Output fails to focus on strategic content and is dominated by irrelevant or low-value information"
        #     ],
        #     threshold=threshold,
        #     rubric=common_rubric,
        # ),

        "Professional Tone & Grammar" : GEval(
            name="Professional Tone & Grammar",
            model=custom_llm,
            evaluation_params=common_params,
            criteria="""Evaluate whether the actual_output maintains a professional tone and writing style appropriate for the intended audience and context.
            Rules:
            - Language should reflect professional standards: formal, respectful, and objective.
            - Avoid overly casual expressions, slang, contractions, emojis, speculative tone, or emotionally charged language.
            - Terminology should align with domain-specific norms (e.g., financial, medical, technical) when applicable.
            - For financial or executive communication (e.g., CIO reports), tone should be neutral, precise, and insight-driven.

            Reject or downgrade if:
            - The output includes informal, vague, or speculative phrases (e.g., “maybe,” “I think,” “kind of”).
            - The tone is too conversational or emotionally biased.
            - Terminology is misused or lacks the precision expected in a professional setting.
            - Formatting or grammar undermines the output’s professional quality.""",

            evaluation_steps=[
                "Assess whether the 'actual_output' maintains a formal and professional tone appropriate for the audience (e.g., investors, executives, customers).",
                "Scan for casual or unprofessional expressions such as slang, emojis, unnecessary contractions (e.g., 'gonna', 'wanna'), or filler phrases (e.g., 'you know', 'kind of').",
                "Evaluate whether the terminology used is appropriate and accurate for the domain (e.g., financial terms like 'basis points as bps', '10 year as 10Y').",
                "Ensure the tone is objective and neutral — avoid speculative or emotionally charged language (e.g., 'we're very excited', 'this is amazing').",
                "Check grammar, sentence structure, and formatting consistency to ensure the text looks polished and professional.",
                "Do not penalize for paraphrased phrasing as long as tone and domain alignment are preserved.",
                "Assign a score between 0.0 and 1.0 based on the following scale: Score ≥ 0.9: Output demonstrates an impeccable professional tone, with formal, precise, and neutral language fully aligned to the domain and audience expectations.; Score ≥ 0.7: Output maintains a generally professional tone with only minor lapses, such as slightly informal phrasing or minor stylistic inconsistencies that do not significantly detract from overall professionalism.; Score ≥ 0.5: Output shows noticeable issues with tone, such as informal expressions, vague or speculative language, or inconsistent terminology, but remains somewhat serviceable in a professional context.; Score < 0.5: Output is unprofessional in tone, with casual language, misuse of terminology, emotional bias, or poor grammar that undermines credibility."
            ],
            threshold=threshold,
            rubric=common_rubric,
        ),

        "Executive Writing Quality" : GEval(
            name="Executive Writing Quality",
            model=custom_llm,
            evaluation_params=common_params,
            criteria="""Evaluate whether the 'actual_output' is written with a high level of polish and clarity suitable for time-constrained, decision-making professionals (e.g., CIOs or investors).
            Rules:
            - The language must be clear, direct, grammatically correct, and free from spelling or punctuation errors.
            - Each point (or sentence) should be concise and free of unnecessary filler, vague modifiers, or marketing-style language.
            - Outputs should be framed for maximum reader impact — the 'so what?' or key takeaway should be obvious.
            - Language should avoid ambiguity, jargon, and speculative or fluffy phrases.
            - Effective phrasing includes action verbs, precise figures, and structured clarity (e.g., cause → effect).
            Reject or downgrade if:
            - Sentences are overly long, vague, or wordy.
            - The output contains spelling, grammar, or punctuation errors.
            - Sentences feel weak, unfocused, or padded with non-essential modifiers.
            - The key takeaway is buried, unclear, or absent.""",

            evaluation_steps=["Check if the language is grammatically correct, with no spelling or punctuation issues.",
            "Review each sentence for clarity: is the meaning immediately clear and unambiguous?",
            "Check if phrasing uses direct, impactful constructions (e.g., begins with a strong verb, includes clear outcomes or figures).",
            "Flag the use of filler, buzzwords, or ambiguous terms (e.g., 'potentially impactful', 'somewhat likely').",
            "Determine whether each point conveys a clear takeaway or adds value to the audience.",
            "Penalize outputs that contain fluff, buried conclusions, or complex wording that reduces clarity.",
            "Assign a score between 0.0 and 1.0 based on the following scale: Score ≥ 0.9: Output is highly polished, clear, and impactful — language is concise, direct, grammatically flawless, and each point conveys a strong takeaway suitable for executives.; Score ≥ 0.7: Output is generally clear and professional, with only minor issues such as slightly wordy phrasing or buried takeaways that do not significantly impede understanding.; Score ≥ 0.5: Output has noticeable weaknesses in clarity, conciseness, or impact — sentences may be vague, filler-heavy, or lack clear takeaways, though still serviceable.; Score < 0.5: Output is poorly written for executive consumption — wordy, confusing, unpolished, with grammatical errors or missing key points."
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
            criteria="""Evaluate whether the actual_output presents ideas in a logically ordered, coherent, and well-structured manner.
            Rules:
            - The sequence of information should follow a clear organizing principle such as thematic grouping, logical progression (e.g., Thesis → Evidence), or strategic framing (e.g., Risks → Implications → Outlook).
            - A high score should be given for outputs that thoughtfully reorganize input information (even if it means deviating from original order) to improve clarity, especially for investor understanding.
            - Outputs must avoid abrupt topic shifts or contradictions.
            Reject or downgrade if:
            - Ideas are randomly ordered or grouped in a confusing manner.
            - Transitions between ideas are missing or unclear.
            - The output contradicts itself or disrupts logical flow.
            - Structuring is present but feels unnatural or forced.""",
            evaluation_steps=[
            "Analyze the actual_output for whether it follows a coherent organizing principle: thematic grouping, chronological flow, strategic grouping (e.g., all risks together), or argument structure (e.g., thesis → evidence → implication).",
            "Check if the ideas are easy to follow, with smooth progression from one point to the next.",
            "Ensure that any reordering from the input enhances rather than confuses the output.",
            "Penalize jumbled, incoherent, contradictory, or disjointed outputs, regardless of format.",
            "Do not penalize for tone, factual precision, or strategic selection — focus only on idea structure and clarity of flow.",
            "Assign a score between 0.0 and 1.0 based on the following scale: Score ≥ 0.9: Output is exceptionally clear, logically ordered, and easy to follow; ideas are grouped and sequenced in a way that enhances understanding and readability.; Score ≥ 0.7: Output is generally coherent with only minor lapses in flow or slightly awkward transitions that don’t significantly confuse the reader.; Score ≥ 0.5: Output has noticeable flaws in logical flow, such as disjointed ordering, unclear groupings, or jarring transitions, but is still somewhat comprehensible.; Score < 0.5: Output lacks logical structure, is confusing or contradictory, with ideas presented in a random or incoherent way that hinders understanding."
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

    else:
        raise ValueError(f"Unknown task_type: {task_type}")
