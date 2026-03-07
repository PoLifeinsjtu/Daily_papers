import operator
from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ================= 1. 定义全局状态 (State) =================
class AgentState(TypedDict):
    query: str
    # 使用 Annotated 和 operator.add 意味着不同节点返回的列表会被追加，而不是覆盖
    downloaded_papers: Annotated[List[Dict], operator.add] 
    analyzed_papers: Annotated[List[Dict], operator.add]
    approved_papers: Annotated[List[Dict], operator.add]

# ================= 2. 定义 Agent 节点 (Nodes) =================

def scout_node(state: AgentState):
    print("🕵️ [Scout Agent] 正在检索和下载论文...")
    query = state["query"]
    
    # 💡 在这里插入你刚才写的 Semantic Scholar 下载脚本的逻辑
    # 将原来的 main() 逻辑稍微改造，使其返回一个包含论文信息的列表
    # 比如：[{"title": "Paper 1", "pdf_path": "./physics_knowledge_base/Paper_1.pdf"}, ...]
    
    # 模拟下载结果
    mock_downloaded = [
        {"title": "Quantum Attention Networks", "pdf_path": "path/to/pdf1.pdf"},
        {"title": "GNN for Particle Tracking", "pdf_path": "path/to/pdf2.pdf"}
    ]
    print(f"✅ [Scout Agent] 成功下载 {len(mock_downloaded)} 篇文献。")
    return {"downloaded_papers": mock_downloaded}

def analyst_node(state: AgentState):
    print("🧠 [Analyst Agent] 正在阅读全文并提取 Idea...")
    papers = state.get("downloaded_papers", [])
    analyzed_results = []
    
    for paper in papers:
        # 💡 这里未来会接入 PDF 解析和 LLM 提取逻辑
        # text = parse_pdf(paper["pdf_path"])
        # ideas = llm.invoke(prompt + text)
        
        # 模拟 LLM 提取的 Idea
        analyzed_results.append({
            "title": paper["title"],
            "ideas": ["Idea 1: Replace traditional tracking with GNN...", "Idea 2: Use transformer for quantum state prep..."],
            "status": "pending_review" # 等待人工确认
        })
    print("✅ [Analyst Agent] 深度分析完成。")
    return {"analyzed_papers": analyzed_results}

def publisher_node(state: AgentState):
    print("🚀 [Publisher Agent] 正在将选中的 Idea 推送至 GitHub...")
    approved = state.get("approved_papers", [])
    
    if not approved:
        print("⚠️ 没有需要推送的内容。")
        return {"approved_papers": []}
        
    for paper in approved:
        # 💡 这里未来会接入 PyGithub 逻辑
        print(f"📦 已推送: {paper['title']} 的阅读笔记和灵感")
        
    return {"approved_papers": approved} # 状态保持不变或记录发布时间

# ================= 3. 构建并编译图 (Graph) =================

def build_graph():
    # 初始化状态图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("scout", scout_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("publisher", publisher_node)

    # 定义边（执行顺序）
    workflow.set_entry_point("scout")
    workflow.add_edge("scout", "analyst")
    workflow.add_edge("analyst", "publisher")
    workflow.add_edge("publisher", END)

    # 🛑 核心机制：设置记忆（Checkpointer）和中断点
    memory = MemorySaver()
    # 编译图：在 "publisher" 节点之前中断（挂起），等待人工审核
    app = workflow.compile(checkpointer=memory, interrupt_before=["publisher"])
    
    return app

# ================= 4. 运行与人工介入测试 =================
if __name__ == "__main__":
    app = build_graph()
    
    # 必须提供一个 thread_id，LangGraph 靠这个记住当前的运行状态
    config = {"configurable": {"thread_id": "phd_pipeline_001"}}
    initial_state = {
        "query": "quantum physics deep learning",
        "downloaded_papers": [],
        "analyzed_papers": [],
        "approved_papers": []
    }

    print("\n--- 🟢 第一阶段：自动抓取与分析开始 ---")
    # 运行图，它会在 publisher 之前停下
    for event in app.stream(initial_state, config):
        pass 
    
    print("\n--- ⏸️ 流程已挂起，等待人工 (The Curator) 介入 ---")
    
    # 获取当前图的状态
    current_state = app.get_state(config).values
    pending_papers = current_state.get("analyzed_papers", [])
    
    print(f"\n👇 请查看今日提取的 {len(pending_papers)} 篇论文 Idea 👇")
    approved_list = []
    
    # 模拟人工在命令行审核
    for idx, paper in enumerate(pending_papers):
        print(f"\n【文献 {idx+1}】: {paper['title']}")
        for idea in paper['ideas']:
            print(f"   💡 {idea}")
            
        # 人工决定是否 Approve
        choice = input("是否将此笔记归档并发布？(y/n): ")
        if choice.lower() == 'y':
            approved_list.append(paper)
            
    print("\n--- 🟢 第二阶段：更新状态并恢复执行 ---")
    # 将人工审核通过的论文更新到图的状态中
    app.update_state(config, {"approved_papers": approved_list})
    
    # 恢复图的运行（传入 None 代表继续执行下一个节点 publisher）
    for event in app.stream(None, config):
        pass
        
    print("\n🎉 全流程结束！")