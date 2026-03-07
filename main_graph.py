import os
from langchain_openai import ChatOpenAI 
from pydantic import BaseModel, Field

# 假设这是我们在前面定义好的结构化输出格式
class PaperInsight(BaseModel):
    core_problem: str = Field(description="核心问题")
    cross_disciplinary_ideas: list[str] = Field(description="跨学科 Idea")

# ================= 🌟 核心修改点 =================
# 虽然导入的是 ChatOpenAI，但我们把它“狸猫换太子”指向硅基流动
llm = ChatOpenAI(
    api_key=os.environ.get("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",  # 👈 必须加上硅基流动的 API 终点
    model="deepseek-ai/DeepSeek-V3",           # 👈 指定使用 DeepSeek-V3 模型
    temperature=0.7                            # 发散性适中
)

# 绑定结构化输出 (DeepSeek-V3 完美支持这个功能)
structured_llm = llm.with_structured_output(PaperInsight)
# ===============================================

# 测试调用一下看看（确保环境配通了）
if __name__ == "__main__":
    print("🤖 正在呼叫 DeepSeek 大脑...")
    result = structured_llm.invoke("论文摘要：本文提出了一种新的图神经网络来加速大型强子对撞机(LHC)中的粒子轨迹重建...")
    print(result)