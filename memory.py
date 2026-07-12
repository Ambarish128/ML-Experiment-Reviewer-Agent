from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# LLM
llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0)

# Prompt - with History PlaceHolder

followup_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior ML engineer who has just completed a structured review 
of an ML experiment. You are now answering follow-up questions from the user.
Be concise and precise. Reference specific numbers from prior context when relevant.
Do not repeat the full review unless explicitly asked."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

#String Output parser - Getting simple string as response form LLM
parser= StrOutputParser()

#Chain
followup_chain= followup_prompt | llm | parser

# --------------------------------------------------
# Session class — scoped history, one per conversation
# --

class ReviewSession:
    """
    Encapsulates one review conversation.
    Each instance has its own isolated chat history.
    Instantiate one per user session — no shared global state.
    """

    def __init__(self,experiment_id:str):
        self.experiment_id=experiment_id
        self.chat_history=[]

    def seed(self,review_summary:str):
        """
        Seeds the session with the initial review result so the LLM
        has context before the first follow-up question.
        Call this immediately after running the analysis chains.
        """
        self.chat_history.append(
            HumanMessage(content=f"Please review experiment {self.experiment_id}.")
        )
        self.chat_history.append(
            AIMessage(content=review_summary)
        )

    def chat(self, user_input: str) -> str:
        """
        Send a follow-up message. Returns the assistant's response.
        History is updated automatically.
        """
        response = followup_chain.invoke({
            "chat_history": self.chat_history,
            "user_input": user_input
        })

        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response))

        return response
    
    def reset(self):
        """Clear history — start a fresh conversation for same experiment."""
        self.chat_history=[]

    def history_length(self):
        """Number of messages stored (each turn = 2 messages)."""
        return len(self.chat_history)
    
# --------------------------------------------------
# Sanity check
# --------------------------------------------------
if __name__ == "__main__":
    session = ReviewSession(experiment_id="EXP_001")

    # Seed with a summary of what the chains found
    session.seed(
        review_summary="""EXP_001 Review Complete:
- Model: ResNet50, 10 epochs, binary classification
- Overfitting detected at epoch 5
- Train loss: 0.142 (final), Val loss: 2.214 (final)
- Eval score peaked at 0.781 (epoch 4), degraded to 0.619 by epoch 10
- Flags: severe val loss divergence, no early stopping configured, all layers unfrozen on small dataset"""
    )

    print(f"History after seed: {session.history_length()} messages\n")

    turns = [
        "Why did the eval score keep dropping after epoch 4?",
        "Would freezing the early layers have helped?",
        "Give me one specific code change to fix the most critical issue."
    ]

    for i, question in enumerate(turns, 1):
        print(f"Turn {i}: {question}")
        response = session.chat(question)
        print(f"Response: {response}\n")
        print("=" * 60)
        

