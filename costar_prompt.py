class CostarPrompt:
    """Represents the CO-STAR framework prompt structure."""

    def __init__(self, context=None, objective=None, style=None, tone=None, audience=None, response=None):
        self.context = context
        self.objective = objective
        self.style = style
        self.tone = tone
        self.audience = audience
        self.response = response

    def __str__(self):
        costar_prompt = ""
        if self.context:
            costar_prompt += "# CONTEXT #\n" + self.context + "\n"
        if self.objective:
            costar_prompt += "# OBJECTIVE #\n" + self.objective + "\n"
        if self.style:
            costar_prompt += "# STYLE #\n" + self.style + "\n"
        if self.tone:
            costar_prompt += "# TONE #\n" + self.tone + "\n"
        if self.audience:
            costar_prompt += "# AUDIENCE #\n" + self.audience + "\n"
        if self.response:
            costar_prompt += "# RESPONSE #\n" + self.response + "\n"
        return costar_prompt

    def __repr__(self):
        return self.__str__()