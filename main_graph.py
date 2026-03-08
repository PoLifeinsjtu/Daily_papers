import os
import json
import fitz  # PyMuPDF
import operator
import datetime
from typing import TypedDict, List, Dict, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import paper_reading

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

# ================= 🌟 V3 专属：深度概念拆解模型 =================
class KeyConcept(BaseModel):
    term: str = Field(description="硬核基础概念名称（如 Langevin Dynamics, DPO, Transformer Attention等）")
    math_formulation: str = Field(description="该概念的核心数学公式，必须使用严谨的 LaTeX 语法，用 $$ 包裹")
    explanation: str = Field(description="用通俗易懂的语言解释这个公式的直觉意义，以及在本文中是如何被应用或更新的")

class PaperInsight(BaseModel):
    core_problem: str = Field(description="一句话总结这篇论文试图解决的核心问题")
    methodology: str = Field(description="提炼论文的核心算法或系统架构")
    # 强制大模型提取基础概念并解释公式
    fundamental_concepts: List[KeyConcept] = Field(description="提取文中 1-3 个最核心的基础算法、物理或数学概念，并提供公式和保姆级解释")
    takeaways: List[str] = Field(description="这篇文章对你的 Multi-Agent 或 LLM 学习有什么具体的启发？列出 2 点")

llm = ChatOpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url="https://api.siliconflow.cn/v1",
    model="deepseek-ai/DeepSeek-V3", 
    temperature=0.6 # 稍微降低温度，保证数学公式的严谨性
)
structured_llm = llm.with_structured_output(PaperInsight)

# ================= 📊 状态字典与历史记录工具 =================
HISTORY_FILE = "read_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history_list):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history_list, f)

class AgentState(TypedDict):
    queries: List[str]
    downloaded_papers: Annotated[List[Dict], operator.add]
    analyzed_papers: Annotated[List[Dict], operator.add]

# ================= 🤖 Agent 节点 =================

def scout_node(state: AgentState):
    print("\n🕵️ [Scout Node] 开始检索最新前沿文献...")
    queries = state["queries"]
    history = load_history()
    all_new_papers = []
    
    for q in queries:
        # 每次每个关键词只取最新的 2 篇，贵精不贵多
        papers = paper_reading.fetch_papers(query=q, limit=2, save_dir="ai_knowledge_base")
        
        for p in papers:
            paper_id = p.get("paperId")
            # 🌟 历史去重机制：看过的绝对不再看
            if paper_id not in history and paper_id != "unknown_id":
                all_new_papers.append(p)
                history.append(paper_id)
            else:
                print(f"⏩ 跳过已读或无效文献: {p['title'][:30]}...")
                
    save_history(history) # 更新本地记忆
    
    # 自身去重
    unique_papers = list({p["pdf_path"]: p for p in all_new_papers}.values())
    print(f"✅ [Scout Node] 侦察结束，获取到 {len(unique_papers)} 篇未读的新文献。")
    return {"downloaded_papers": unique_papers}

def analyst_node(state: AgentState):
    print("\n🧠 [Analyst Node] 唤醒 DeepSeek-V3，开始硬核公式拆解与深度学习...")
    papers = state.get("downloaded_papers", [])
    analyzed_results = []
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位顶尖的 AI 研究员与耐心严谨的导师。用户希望深入学习【大语言模型 (LLM)】与【智能体 (Multi-Agent)】的底层原理，偶尔涉及 AI for Science。
        用户的痛点是：对很多基础概念（如 EBM, Langevin Dynamics, PPO, 扩散模型等）只停留在表面，不懂数学推导。
        你的任务是：
        1. 总结论文核心内容。
        2. 像剥洋葱一样，把论文中提到的 1-3 个最核心的基础概念剥离出来。
        3. 写出该概念极其严谨的数学公式（使用 LaTeX $$...$$），并用大白话解释公式里每一个符号的直觉含义。"""),
        ("human", "论文标题：{title}\n\n论文前两页内容：\n{paper_text}\n\n请进行深度教学级别的提炼。")
    ])
    chain = prompt | structured_llm

    for paper in papers:
        pdf_path = paper.get("pdf_path")
        title = paper.get("title")
        print(f"📖 正在精读并拆解: {title[:50]}...")
        
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for i in range(min(2, len(doc))):
                text += doc[i].get_text("text") + "\n"
        except Exception:
            continue
            
        if len(text.strip()) < 100: continue
            
        try:
            insight: PaperInsight = chain.invoke({"title": title, "paper_text": text})
            analyzed_results.append({
                "title": title,
                "core_problem": insight.core_problem,
                "methodology": insight.methodology,
                "concepts": insight.fundamental_concepts, # 存入提取出的硬核概念
                "takeaways": insight.takeaways
            })
            print(f"💡 成功拆解 {len(insight.fundamental_concepts)} 个底层概念！")
        except Exception as e:
            print(f"❌ 分析失败: {e}")
            
    return {"analyzed_papers": analyzed_results}

def publisher_node(state: AgentState):
    print("\n📝 [Publisher Node] 正在生成《硬核学习笔记 (Deep Dive Study Notes)》...")
    analyzed = state.get("analyzed_papers", [])
    output_dir = "Study_Notes"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not analyzed:
        print("⚠️ 今日没有新的文献需要生成笔记。")
        return {"analyzed_papers": []}
        
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(output_dir, f"Deep_Dive_{today_str}.md")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"# 🧠 LLM & Agent 深度学习笔记 ({today_str})\n\n---\n\n")
        
        for idx, paper in enumerate(analyzed):
            f.write(f"## {idx+1}. {paper['title']}\n\n")
            f.write(f"**🎯 核心问题**：{paper['core_problem']}\n\n")
            f.write(f"**🛠️ 架构/方法**：{paper['methodology']}\n\n")
            
            f.write("### 🔬 底层概念拆解 (Hardcore Concepts)\n")
            for c in paper['concepts']:
                f.write(f"#### 🔹 {c.term}\n")
                f.write(f"**数学公式**：\n{c.math_formulation}\n\n")
                f.write(f"**保姆级解析**：\n{c.explanation}\n\n")
                
            f.write("### 💡 学习与启发 (Takeaways)\n")
            for t in paper['takeaways']:
                f.write(f"- {t}\n")
            f.write("\n---\n\n")
                
    print(f"📦 硬核笔记已生成！请使用支持 LaTeX 的 Markdown 渲染器打开: {file_path}")
    return {"analyzed_papers": []}

# ================= 🕸️ 启动引擎 =================
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
    
    # 🌟 V3 专属：完全转向纯 AI/Agent 前沿，并带有一点 Physics 交叉
    search_matrix = [
        "Large Language Model reasoning paths",
        "Multi-Agent reinforcement learning collaboration",
        "Energy-based models diffusion",
        "AI for Science Physics foundation models"
    ]
    
    initial_state = {
        "queries": search_matrix, 
        "downloaded_papers": [],
        "analyzed_papers": []
    }
    
    print("=========================================")
    print("🚀 LangGraph 深度学习框架 V3.0 启动！")
    print("=========================================")
    
    for event in app.stream(initial_state):
        pass