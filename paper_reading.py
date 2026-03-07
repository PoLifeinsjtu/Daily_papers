import requests
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= ⚙️ 配置区域 =================
QUERY = "quantum physics"
API_KEY = "sk-GeZEc4gmYZ4jYee8E9Aa51Cd333247A2818c173d312c5a15"
DOWNLOAD_LIMIT = 10
SAVE_DIR = "physics_knowledge_base"
BATCH_SIZE = 100
MAX_WORKERS = 8   # 并发线程数（建议 6-10）
# ===============================================

PROXIES = {
    "http": "http://127.0.0.1:10809",
    "https": "http://127.0.0.1:10809",
}

def sanitize_filename(title):
    return re.sub(r'[\\/*?:"<>|]', "", title)[:150].strip()

def create_session():
    session = requests.Session()
    session.proxies.update(PROXIES)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    })
    return session

def download_pdf(session, url, filepath):
    try:
        with session.get(url, stream=True, timeout=60) as r:
            if r.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        f.write(chunk)
                return True
    except Exception:
        pass
    return False

def fetch_papers(query: str, limit: int = 5, save_dir: str = "physics_knowledge_base") -> list:
    """
    改造后的下载函数：接收参数，并返回成功下载的论文列表
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    api_headers = {'x-api-key': "sk-GeZEc4gmYZ4jYee8E9Aa51Cd333247A2818c173d312c5a15"} # 记得替换
    api_session = create_session()
    
    downloaded_papers = [] # 👈 用于存储成功下载的论文信息，供 LangGraph 使用
    downloaded_count = 0
    offset = 0

    print(f"🚀 [Scout] 开始 API 检索：{query}，目标 {limit} 篇")

    while downloaded_count < limit:
        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "offset": offset,
            "limit": 10, # 每次请求的批次可以小一点
            "fields": "title,isOpenAccess,openAccessPdf,citationCount,year",
            "sort": "citationCount:desc"
        }

        r = api_session.get(search_url, params=params, headers=api_headers)
        time.sleep(1.05)

        if r.status_code != 200:
            print("❌ API 错误:", r.status_code)
            break

        papers = r.json().get("data", [])
        if not papers:
            break

        tasks = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for paper in papers:
                if downloaded_count >= limit:
                    break

                pdf_info = paper.get("openAccessPdf")
                if not pdf_info or not pdf_info.get("url"):
                    continue

                title = paper.get("title", "Untitled")
                filename = sanitize_filename(title) + ".pdf"
                filepath = os.path.join(save_dir, filename)
                
                # 无论文件是刚下载的还是已经存在的，我们都记录下来
                paper_record = {"title": title, "pdf_path": filepath}

                if os.path.exists(filepath):
                    downloaded_papers.append(paper_record)
                    downloaded_count += 1
                    continue

                session = create_session()
                tasks.append(
                    (executor.submit(download_pdf, session, pdf_info["url"], filepath), paper_record)
                )

            # 收集多线程下载的结果
            for future, record in tasks:
                if future.result():
                    downloaded_papers.append(record)
                    downloaded_count += 1
                    print(f"✅ 下载完成: {record['title'][:30]}...")
                else:
                    print("❌ 下载失败")

        offset += 10

    return downloaded_papers # 👈 关键：返回这个列表给 LangGraph

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    api_headers = {'x-api-key': API_KEY}
    api_session = create_session()

    downloaded_count = 0
    offset = 0

    print(f"🚀 开始任务：{DOWNLOAD_LIMIT} 篇")

    while downloaded_count < DOWNLOAD_LIMIT:

        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"

        params = {
            "query": QUERY,
            "offset": offset,
            "limit": BATCH_SIZE,
            "fields": "title,isOpenAccess,openAccessPdf,citationCount,year",
            "sort": "citationCount:desc"
        }

        print(f"\n📡 API 请求 offset={offset}")
        r = api_session.get(search_url, params=params, headers=api_headers)
        time.sleep(1.05)  # 严格控制 API 速率

        if r.status_code != 200:
            print("❌ API 错误:", r.status_code)
            break

        data = r.json()
        papers = data.get("data", [])
        if not papers:
            break

        tasks = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for paper in papers:
                if downloaded_count >= DOWNLOAD_LIMIT:
                    break

                pdf_info = paper.get("openAccessPdf")
                if not pdf_info or not pdf_info.get("url"):
                    continue

                title = paper.get("title", "Untitled")
                filename = sanitize_filename(title) + ".pdf"
                filepath = os.path.join(SAVE_DIR, filename)

                if os.path.exists(filepath):
                    continue

                session = create_session()
                tasks.append(
                    executor.submit(download_pdf, session, pdf_info["url"], filepath)
                )

                downloaded_count += 1

            for future in as_completed(tasks):
                if future.result():
                    print("✅ 下载完成")
                else:
                    print("❌ 下载失败")

        offset += BATCH_SIZE

    print(f"\n🎉 完成，共下载 {downloaded_count} 篇")


if __name__ == "__main__":
    main()