from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.deckduckgo import DuckDuckGo

class WebAgent(Agent):
    def __init__(self):
        super().__init__(name="Web Agent")
        self.tools = [DuckDuckGo()]
        self.instructions = ["Always include sources in your response"]
        self.model = OpenAIChat(id="gpt-4o-mini")
        self.show_tool_calls = True
        self.markdown = True

    def generate(self, prompt: str) -> str:
        return self.print_response(prompt, stream=True)