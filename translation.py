from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from llm import generate_string, openai_llm
from costar_prompt import CostarPrompt
from typing import TypedDict, List, Optional
from langchain.tools import tool
import pandas as pd 


class Translator:
    def __init__(self, show_prompt=False):
        self.show_prompt = show_prompt
        self.last_prompts = {}

    @tool
    def terminology_lookup(term: str) -> Optional[str]:
        """
        Look up official localized term from terminology.csv.
        Handles acronyms, partial matches, and full matches.
        The CSV must have columns: 'en-MY' and 'zh-MY'.
        """
        try:
            df = pd.read_csv("terminology.csv")
            df["en-MY_lower"] = df["en-MY"].str.lower().str.strip()
            term_lower = term.lower().strip()

            # 1. Exact match
            exact = df[df["en-MY_lower"] == term_lower]
            if not exact.empty:
                return exact.iloc[0]["zh-MY"]

            # 2. Acronym match (e.g. "BNM" inside "Bank Negara Malaysia (BNM)")
            acronym_match = df[df["en-MY_lower"].str.contains(f"({term_lower})", regex=False)]
            if not acronym_match.empty:
                return acronym_match.iloc[0]["zh-MY"]

            # 3. Partial fuzzy match (e.g. "Khazanah" within "Khazanah Nasional Bhd")
            partial = df[df["en-MY_lower"].str.contains(term_lower, na=False)]
            if not partial.empty:
                return partial.iloc[0]["zh-MY"]

            return None

        except Exception as e:
            return f"[Lookup error: {str(e)}]"

    # -----------------------------------------------------------
    # STEP 1: Section Header Translation
    # -----------------------------------------------------------
    def translate_section_header(self, source_text: str, source_lang: str, target_lang: str):
        """Translate a short section header term."""
        costar_prompt = CostarPrompt(
            context=f"You are a Translator for a bank, translating financial terminology from {source_lang} to {target_lang}.",
            objective=f"Translate the given term '{source_text}' from {source_lang} to {target_lang}.",
            audience="Your audience is the bank's investment report senior editor.",
            response=f"Output just the translated term of '{source_text}' in {target_lang}. Do not include explanations or any additional text."
        )

        llm = openai_llm(temperature=0)
        result = generate_string(
            llm, str(costar_prompt), {}, show_prompt=self.show_prompt, system_prompt_only=True
        )
        return result.content

    # -----------------------------------------------------------
    # STEP 2: Full Translation
    # -----------------------------------------------------------
    def translate(self, source_text: str, source_lang: str, target_lang: str):
        """Perform initial translation in the tone of investment or corporate news."""
        costar_prompt = CostarPrompt(
            context=f"You are a Translator for a bank, translating financial text such as investment or corporate news from {source_lang} to {target_lang}.",
            objective=f"Translate the text '{source_text}' from {source_lang} to {target_lang}. Tone should match the style of a financial report.",
            audience="Your audience is the bank's investment report senior editor.",
            response=f"Output just the translation of '{source_text}' in {target_lang}, with no explanation or introduction."
        )

        llm = openai_llm(temperature=0)
        result = generate_string(
            llm, str(costar_prompt), {}, show_prompt=self.show_prompt, system_prompt_only=True
        )
        return result.content

    # -----------------------------------------------------------
    # STEP 4: Editor Comments Generation
    # -----------------------------------------------------------
    def editor_comments(self, source_text: str, translated_text: str, target_lang: str):
        """Generate feedback from senior editor on translation quality."""
        costar_prompt = CostarPrompt(
            context="You are a senior linguistic expert that specializes in financial text translation.",
            objective=f"""
            Based on the translated text below produced by a junior translator, provide constructive comments to improve the output.

            ### Source Text ###
            {source_text}

            ### Translated Text ###
            {translated_text}

            When writing your suggestions, pay attention to whether there are ways to improve the translation's:
            (i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
            (ii) fluency (by applying {target_lang} grammar, spelling, and punctuation rules, and ensuring there are no unnecessary repetitions unless in report headers),
            (iii) style (by ensuring the translations reflect the tone of a financial report),
            (iv) terminology (by ensuring terminology use is consistent with the financial domain in {target_lang}).
            """,
            audience="Your audience is the bank's translator who will revise the translation based on your comments.",
            response="Provide your improvement suggestions in concise bullet points."
        )

        llm = openai_llm(temperature=0)
        suggestion = generate_string(
            llm, str(costar_prompt), {}, show_prompt=self.show_prompt, system_prompt_only=True
        )
        return suggestion.content
    
    # -----------------------------------------------------------
    # STEP 4: Refinement (After Editor Suggestions)
    # -----------------------------------------------------------
    def refine_translation(self, source_text: str, initial_translated_text: str, improvements: str,
                           source_lang: str, target_lang: str):
        """Refine translation based on editor's improvement suggestions."""
        costar_prompt = CostarPrompt(
            context=f"You are a Translator for a bank, refining translations from {source_lang} to {target_lang} based on senior editor feedback.",
            objective=f"Given the initial translation:\n'{initial_translated_text}'\nand editor's comments:\n'{improvements}'\nrework the translation accordingly.",
            audience="Your audience is the bank's investment report senior editor.",
            response=f"Output the final refined translation in {target_lang} only — no explanations, just the text."
        )

        llm = openai_llm(temperature=0)
        result = generate_string(
            llm, str(costar_prompt), {}, show_prompt=self.show_prompt, system_prompt_only=True
        )
        return result.content
