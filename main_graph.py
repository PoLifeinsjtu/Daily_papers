import os
import fitz  # PyMuPDF
import operator
from typing import TypedDict, List, Dict, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import paper_reading

# ================= 📂 核心修改：动态绝对路径锚定 =================
# 获取 main_graph.py 当前所在的绝对目录 (.../daily_papers/Daily_papers)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 强制将下载目录和 Inbox 目录绑定在 main_graph.py 同级
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "physics_knowledge_base")
INBOX_DIR = os.path.join(BASE_DIR, "Idea_Inbox")
# ================================================================

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

class PaperInsight(BaseModel):
    core_problem: str = Field(description="一句话总结这篇论文试图解决的核心物理或AI问题")
    methodology: str = Field(description="提炼论文使用了什么核心技术、算法或数学工具")
    cross_disciplinary_ideas: List[str] = Field(
        description="结合粒子物理和人工智能，提出 2 个具体的、有可行性的 follow-up 研究灵感"
    )

llm = ChatOpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url="https://api.siliconflow.cn/v1",
    model="deepseek-ai/DeepSeek-V3", 
    temperature=0.7
)
structured_llm = llm.with_structured_output(PaperInsight)

class AgentState(TypedDict):
    query: str
    downloaded_papers: Annotated[List[Dict], operator.add]
    analyzed_papers: Annotated[List[Dict], operator.add]

def scout_node(state: AgentState):
    print("\n🕵️ [Scout Node] 开始侦察和下载最新文献...")
    query = state["query"]
    # 👇 修改这里：传入绝对路径
    new_papers = paper_reading.fetch_papers(query=query, limit=3, save_dir=KNOWLEDGE_BASE_DIR)
    print(f"✅ [Scout Node] 侦察结束，成功获取 {len(new_papers)} 篇 PDF。")
    return {"downloaded_papers": new_papers}

# ... (analyst_node 代码完全保持不变) ...
def analyst_node(state: AgentState):
    print("\n🧠 [Analyst Node] 唤醒 DeepSeek-V3，开始提取跨学科 Idea...")
    papers = state.get("downloaded_papers", [])
    analyzed_results = []
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位精通【粒子物理学】与【人工智能】的顶尖研究员。
        你的任务是审阅以下论文片段，并为一名博士生提供极具洞察力的跨学科研究灵感。
        切忌说空话，必须结合物理概念和具体的 AI 模型架构（如 Transformer, GNN 等）。"""),
        ("human", "论文标题：{title}\n\n论文核心内容片段：\n{paper_text}\n\n请提取要点并生成具有启发性的交叉领域 Idea。")
    ])
    chain = prompt | structured_llm

    for paper in papers:
        pdf_path = paper.get("pdf_path")
        title = paper.get("title")
        print(f"📖 正在精读: {title[:50]}...")
        
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for i in range(min(2, len(doc))):
                text += doc[i].get_text("text") + "\n"
        except Exception as e:
            print(f"⚠️ 读取 {title} 失败: {e}")
            continue
            
        if len(text.strip()) < 100:
            print("⚠️ 提取文本过短，跳过。")
            continue
            
        try:
            insight: PaperInsight = chain.invoke({"title": title, "paper_text": text})
            analyzed_results.append({
                "title": title,
                "core_problem": insight.core_problem,
                "methodology": insight.methodology,
                "ideas": insight.cross_disciplinary_ideas
            })
            print("💡 Idea 提取成功！")
        except Exception as e:
            print(f"❌ DeepSeek 调用失败: {e}")
            
    return {"analyzed_papers": analyzed_results}

def publisher_node(state: AgentState):
    print("\n📝 [Publisher Node] 正在将生成的 Idea 写入个人知识库 Inbox...")
    analyzed = state.get("analyzed_papers", [])
    
    # 👇 修改这里：使用绝对路径
    if not os.path.exists(INBOX_DIR):
        os.makedirs(INBOX_DIR)
        
    for paper in analyzed:
        safe_title = paper_reading.sanitize_filename(paper['title'])
        file_path = os.path.join(INBOX_DIR, f"{safe_title}.md")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {paper['title']}\n\n")
            f.write(f"### 🎯 核心问题\n{paper['core_problem']}\n\n")
            f.write(f"### 🛠️ 方法与工具\n{paper['methodology']}\n\n")
            f.write("### 💡 跨学科启发 (DeepSeek V3)\n")
            for idx, idea in enumerate(paper['ideas']):
                f.write(f"{idx+1}. {idea}\n\n")
                
        print(f"📦 已归档灵感卡片: {file_path}")
    return {"analyzed_papers": []}

if __name__ == "__main__":
    workflow = StateGraph(AgentState)
    workflow.add_node("scout", scout_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("publisher", publisher_node)
    
    workflow.set_entry_point("scout")
    workflow.add_edge("scout", "analyst")
    workflow.add_edge("analyst", "publisher")
    workflow.add_edge("publisher", END)
    
    app = workflow.compile()
    
    initial_state = {
        "query": "quantum physics machine learning",
        "downloaded_papers": [],
        "analyzed_papers": []
    }
    
    print("=========================================")
    print("🚀 LangGraph 科研大模型流水线启动！")
    print("=========================================")
    
    for event in app.stream(initial_state):
        pass 
        
    print("\n🎉 全流程结束！请查看 Idea_Inbox 文件夹。")