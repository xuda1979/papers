# -*- coding: utf-8 -*-
"""
进一步清理和优化文本格式，使其更像传统书籍
"""

import re
import os

def clean_prose(content):
    """清理转换后的文本，使其更流畅"""
    
    # 1. 移除连续的空行（保留最多2个换行）
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # 2. 移除孤立的\textbf{}标题后面紧跟的空行+内容重复
    # 例如：\textbf{标题}\n\n\textbf{标题} -> \textbf{标题}
    content = re.sub(r'(\\textbf\{[^}]+\})\s*\n\n\1', r'\1', content)
    
    # 3. 将 \textbf{标题}\n\n 后紧跟的独立 \textbf{} 合并为段落开头
    # 这样看起来更自然
    
    # 4. 清理多余的 \vspace 命令
    content = re.sub(r'\\vspace\{[^}]*\}\s*\n*', '\n', content)
    
    # 5. 将孤立的粗体行转为段落开头
    # 例如：\n\n\textbf{某某概念}\n\n后面的内容 -> \n\n\textbf{某某概念}。后面的内容
    
    # 6. 移除空的章节导言（只有标题没有内容的情况）
    content = re.sub(r'\n\n\\textbf\{[^}]*\}\s*\n\n(?=\\section|\\subsection|\\chapter|\\textbf)', '\n\n', content)
    
    # 7. 清理引用块的格式
    content = re.sub(r'\\begin\{quote\}\s*\\textit\{', r'\\begin{quote}\n\\textit{', content)
    
    # 8. 合并连续的独立粗体段落
    def merge_bold_paragraphs(match):
        lines = match.group(0).strip().split('\n\n')
        if len(lines) <= 1:
            return match.group(0)
        # 检查是否都是短粗体行
        short_bolds = []
        for line in lines:
            line = line.strip()
            if re.match(r'^\\textbf\{[^}]{1,50}\}$', line):
                short_bolds.append(line)
            else:
                break
        if len(short_bolds) > 1:
            # 合并为一个段落
            return ' '.join(short_bolds) + '\n\n'
        return match.group(0)
    
    # 9. 标准化 itemize/enumerate 前后的空白
    content = re.sub(r'\n{2,}(\\begin\{itemize\})', r'\n\1', content)
    content = re.sub(r'(\\end\{itemize\})\n{2,}', r'\1\n\n', content)
    content = re.sub(r'\n{2,}(\\begin\{enumerate\})', r'\n\1', content)
    content = re.sub(r'(\\end\{enumerate\})\n{2,}', r'\1\n\n', content)
    
    # 10. 移除 \subsection* 前多余的空行
    content = re.sub(r'\n{3,}(\\subsection\*)', r'\n\n\1', content)
    
    # 11. 确保 section 前有适当空行
    content = re.sub(r'([^\n])\n(\\section)', r'\1\n\n\2', content)
    
    # 12. 确保 subsection 前有适当空行  
    content = re.sub(r'([^\n])\n(\\subsection)', r'\1\n\n\2', content)
    
    return content

def process_file(filepath):
    """处理单个文件"""
    print(f"清理文件: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = clean_prose(content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

def main():
    """主函数"""
    files = [
        'chapter1_infrastructure.tex',
        'chapter1.tex',
        'chapter2_efficiency.tex',
        'chapter2.tex',
        'chapter3_intelligence.tex',
        'chapter3.tex',
        'chapter4_risks.tex',
        'chapter4.tex',
        'chapter5_roadmap.tex',
        'chapter5.tex',
        'chapter6_governance.tex',
        'chapter6.tex',
        'chapter7_actions.tex',
        'chapter7.tex',
        'chapter8_science.tex',
        'chapter9_talent.tex',
        'chapter10_investment.tex',
        'chapter11_frontier.tex',
        'chapter12_newsecurity.tex',
        'foreword.tex',
        'preface.tex',
        'afterword.tex',
        'appendix.tex',
    ]
    
    for filename in files:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"文件不存在: {filename}")
    
    print("\n清理完成！")

if __name__ == '__main__':
    main()
