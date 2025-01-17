import os
from typing import Dict, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain & 相关依赖
from langchain.schema import AIMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.agents import (
    initialize_agent,
    Tool
)
from langchain.agents.agent_types import AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory

# --------------- Step 1: Set up API keys ---------------
os.environ["OPENAI_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"

# --------------- Step 2: 定义 FastAPI ---------------
app = FastAPI(title="LangChain ChatBot")

# --------------- Step 3: Initialize the Tavily search tool ---------------
tool = TavilySearchResults(max_results=2)

# --------------- Step 4: Define tools ---------------
tools = [
    Tool(
        name="search",
        func=tool.run,
        description="Search the web and return top search results."
    ),
    Tool(
        name="calculator",
        func=lambda expr: f"The result of {expr} is {eval(expr)}",
        description="A calculator to evaluate mathematical expressions."
    )
]

# --------------- Step 5: Initialize the OpenAI language model ---------------
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

# --------------- Step 6: Initialize Memory & Agent (模板) ---------------
def create_new_agent():
    """
    创建一个新的 Agent 和 Memory，用于管理一个会话的上下文。
    每个 session_id 都拥有自己的 memory & agent。
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )
    return agent, memory


# --------------- 定义存储会话状态的数据结构 ---------------
class SessionState:
    """
    用于存储单个 session (会话) 的 agent, memory, messages 等。
    这样可以在每次调用时，基于 session_id 取回对话历史。
    """
    def __init__(self):
        self.agent, self.memory = create_new_agent()
        self.messages = []  # 存储 [HumanMessage, AIMessage, ...]

# --------------- 全局维护一个 session_states，用于管理多个 session ---------------
session_states: Dict[str, SessionState] = {}


# --------------- Step 7: 定义请求和响应模型 ---------------
class ChatRequest(BaseModel):
    """客户端请求数据"""
    session_id: str
    user_message: str

class ChatResponse(BaseModel):
    """服务端响应数据"""
    session_id: str
    chatbot_message: str


# --------------- Step 8: 定义聊天接口 ---------------
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    - 接收 session_id, user_message
    - 如果不存在该 session_id，则在服务器新建一个会话
    - 将 user_message 传给 agent.run()，产生回答
    - 返回回答，并存储在服务器端
    """
    session_id = request.session_id
    user_input = request.user_message.strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="User message cannot be empty.")

    # 如果 session_id 不存在，则新建会话
    if session_id not in session_states:
        session_states[session_id] = SessionState()

    # 取出本会话的 agent, memory, messages
    session_state = session_states[session_id]
    agent = session_state.agent
    messages = session_state.messages

    # 追加用户消息到本地存储
    user_msg = HumanMessage(content=user_input)
    messages.append(user_msg)

    # 由 Agent 生成回复
    response_text = agent.run(user_input)

    # 追加 AI 消息到本地存储
    ai_msg = AIMessage(content=response_text)
    messages.append(ai_msg)

    # 返回给客户端
    return ChatResponse(
        session_id=session_id,
        chatbot_message=response_text
    )


# --------------- Step 9: 启动应用 ---------------
# 在终端运行: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
