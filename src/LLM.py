from langgraph.graph import START, StateGraph, END, MessagesState
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver


class State(MessagesState):
    question: str
    emotion: str
    answer: str
    next_node: str


class LLM:
    def __init__(self, api_key) -> None:
        self.api_key = api_key
        self.model = BaseChatOpenAI(
            api_key=api_key,
            max_tokens=1024,
            streaming=True
        )
        self.template = """
    Jesteś pomocnym chatbotem, który dostosowuje swoje odpowiedzi na podstawie emocji użytkownika.
    Emocja użytkownika: {emotion}
    Pytanie użytkownika: {question}

    Dostosuj swoją odpowiedź na podstawie emocji:
    - Jeśli emocja to "sad", postaraj się pocieszyć użytkownika.
    - Jeśli emocja to "disgust", bądź delikatny i unikaj drażliwych tematów.
    - Jeśli emocja to "angry", bądź spokojny i pomocny.
    - Jeśli emocja to "neutral", odpowiedź powinna być rzeczowa i konkretna.
    - Jeśli emocja to "fear", zapewnij wsparcie i podaj niezbędne informacje.
    - Jeśli emocja to "surprise", odpowiedź powinna być entuzjastyczna.
    - Jeśli emocja to "happy", bądź przyjazny i pozytywnie nastawiony.
    
    Twoje odpowiedzi nie powinny mieć więcej niż 3 zdania. Około 10 sekund mówionych. Nie kończ swojej wypowiedzi,
    insynuując kontynuację konwersacji. Odpowiadaj rzeczowo i na temat. 
    

    Odpowiedź:
    """
        self.prompt = PromptTemplate.from_template(self.template)
        self.memory = MemorySaver()
        self.graph = self.generate_graph()

    def generate(self, state: State):
        content = self.prompt.invoke(
                {"question": state["question"], "emotion": state["emotion"]}
        ).to_string()

        message_history = "".join(
            msg.content for msg in state.get("messages", []))

        response = self.model.invoke(content + message_history)
        

        return {
            "answer": response.content,
            "messages": state["messages"]
            + [state.get("question")]
            + [response]
        }

    def generate_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("generate", self.generate)
        graph_builder.add_edge(START, "generate")
        graph_builder.add_edge("generate", END)

        return graph_builder.compile(checkpointer=self.memory)

    def invoke(self, message: str, emotion: str):
        message = self.graph.invoke(
            {"question": message, "emotion" : emotion},
            config={"configurable": {"thread_id": 0}},
        )
        return message.get("answer")
