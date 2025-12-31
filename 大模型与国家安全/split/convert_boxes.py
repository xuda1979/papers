# -*- coding: utf-8 -*-
"""
将tcolorbox转换为传统书籍风格的流畅文字
全书box控制在10个以内
"""

import re
import os

def convert_box_to_prose(content):
    """将tcolorbox转换为流畅的散文风格"""
    
    # 匹配tcolorbox的正则
    box_pattern = r'\\begin\{tcolorbox\}\[([^\]]*)\](.*?)\\end\{tcolorbox\}'
    
    def replace_box(match):
        options = match.group(1)
        inner_content = match.group(2).strip()
        
        # 提取标题
        title_match = re.search(r'title=([^,\]]+)', options)
        if title_match:
            title = title_match.group(1).strip()
            # 清理标题中的LaTeX命令
            title = re.sub(r'\\textbf\{([^}]*)\}', r'\1', title)
            title = re.sub(r'\\large\\bfseries', '', title)
            title = re.sub(r'【[^】]*】', '', title)  # 移除【】标记
            title = title.strip()
        else:
            title = ""
        
        # 清理内容
        inner_content = inner_content.strip()
        
        # 根据标题类型决定转换方式
        if '本章核心' in title or '核心论断' in title or '核心警示' in title or '本章要点' in title or '本章定位' in title or '本章目标' in title:
            # 章节开头的导言 - 转为普通段落，用粗体开头
            return f"\n\n{inner_content}\n\n"
        
        elif '原创' in title or '本书原创' in title:
            # 原创理论 - 转为小节内容
            if title:
                clean_title = title.replace('本书原创：', '').replace('原创理论：', '').strip()
                return f"\n\n\\textbf{{{clean_title}}}\n\n{inner_content}\n\n"
            return f"\n\n{inner_content}\n\n"
        
        elif '深度' in title or '技术' in title or '细节' in title:
            # 技术细节 - 保持为普通段落
            if title:
                clean_title = re.sub(r'深度技术[：:]*', '', title)
                clean_title = re.sub(r'技术细节[：:]*', '', clean_title)
                clean_title = re.sub(r'深度分析[：:]*', '', clean_title)
                clean_title = clean_title.strip()
                if clean_title:
                    return f"\n\n\\textbf{{{clean_title}}}\n\n{inner_content}\n\n"
            return f"\n\n{inner_content}\n\n"
        
        elif '警示' in title or '警告' in title or '警醒' in title:
            # 警示内容 - 用粗体标注
            clean_title = title.replace('战略警示：', '').replace('警示：', '').replace('警告：', '').strip()
            if clean_title:
                return f"\n\n\\textbf{{{clean_title}}}。{inner_content}\n\n"
            return f"\n\n{inner_content}\n\n"
        
        elif '建议' in title or '行动' in title:
            # 建议内容
            clean_title = title.replace('建议：', '').replace('行动项', '').strip()
            # 移除数字编号
            clean_title = re.sub(r'^\d+[：:]\s*', '', clean_title)
            if clean_title:
                return f"\n\n\\textbf{{{clean_title}}}\n\n{inner_content}\n\n"
            return f"\n\n{inner_content}\n\n"
        
        elif '案例' in title or '里程碑' in title:
            # 案例分析
            clean_title = title.replace('案例分析：', '').replace('里程碑：', '').strip()
            if clean_title:
                return f"\n\n\\textbf{{{clean_title}}}\n\n{inner_content}\n\n"
            return f"\n\n{inner_content}\n\n"
        
        elif '结论' in title or '小结' in title:
            # 结论 - 转为本章小结
            return f"\n\n\\subsection*{{本章小结}}\n\n{inner_content}\n\n"
        
        else:
            # 其他情况 - 简单转为段落
            if title:
                clean_title = title.strip()
                return f"\n\n\\textbf{{{clean_title}}}\n\n{inner_content}\n\n"
            return f"\n\n{inner_content}\n\n"
    
    # 执行替换（使用DOTALL使.匹配换行）
    result = re.sub(box_pattern, replace_box, content, flags=re.DOTALL)
    
    # 清理多余空行
    result = re.sub(r'\n{4,}', '\n\n\n', result)
    
    return result

def process_file(filepath):
    """处理单个文件"""
    print(f"处理文件: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 统计原有box数量
    original_count = len(re.findall(r'\\begin\{tcolorbox\}', content))
    print(f"  原有box数量: {original_count}")
    
    # 转换
    new_content = convert_box_to_prose(content)
    
    # 统计转换后box数量
    new_count = len(re.findall(r'\\begin\{tcolorbox\}', new_content))
    print(f"  转换后box数量: {new_count}")
    
    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return original_count, new_count

def main():
    """主函数"""
    # 要处理的文件列表
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
    
    total_original = 0
    total_new = 0
    
    for filename in files:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            orig, new = process_file(filepath)
            total_original += orig
            total_new += new
        else:
            print(f"文件不存在: {filename}")
    
    print(f"\n总计: 原有{total_original}个box, 转换后{total_new}个box")

if __name__ == '__main__':
    main()
