import subprocess
import os

def extract_formulas_with_nougat(pdf_path: str, output_dir: str = "./parsed_papers", pages: str = "1,2,-1") -> str:
    """
    使用 Nougat OCR 提取物理论文的关键页面
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"🔬 [Nougat] 正在启动 OCR 引擎解析关键页面: {pdf_path}")
    
    # 构建命令行指令
    command = [
        "nougat", 
        pdf_path, 
        "--pages", pages, 
        "--out", output_dir
    ]
    
    try:
        # 运行子进程，捕获输出
        # 注意：这里可能会耗时几十秒到几分钟，取决于你的 GPU 算力
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Nougat 默认生成的输出文件名与 PDF 同名，但后缀是 .mmd
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}.mmd")
        
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                parsed_text = f.read()
            print("✅ [Nougat] 公式与文本提取成功！")
            return parsed_text
        else:
            print("⚠️ [Nougat] 解析完成，但未找到输出文件。")
            return ""
            
    except subprocess.CalledProcessError as e:
        print(f"❌ [Nougat] 执行失败: {e.stderr.decode('utf-8')}")
        return ""