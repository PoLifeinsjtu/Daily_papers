from typing import TypedDict, List, Dict, Annotated
import operator

# 1. 导入你刚才改造好的脚本
import paper_reading 

class AgentState(TypedDict):
    query: str
    downloaded_papers: Annotated[List[Dict], operator.add]
    # ... 其他状态 ...

def scout_node(state: AgentState):
    print("🕵️ [Scout Agent] 正在调用 Semantic Scholar API 抓取文献...")
    
    # 从状态机中获取你要搜索的关键词
    current_query = state.get("query", "quantum physics artificial intelligence")
    
    # 2. 直接调用你的下载函数！
    # 建议日常运行 limit 设为 3-5 篇即可，避免大模型 API 费用过高和人工审核疲劳
    new_papers = paper_reading.fetch_papers(query=current_query, limit=3)
    
    print(f"🎯 [Scout Agent] 任务完成，成功获取 {len(new_papers)} 篇论文移交分析节点。")
    
    # 3. 将结果打包进字典，LangGraph 会自动将其追加到全局状态的 downloaded_papers 列表中
    return {"downloaded_papers": new_papers}