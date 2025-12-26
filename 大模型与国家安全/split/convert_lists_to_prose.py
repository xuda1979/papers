# -*- coding: utf-8 -*-
"""
将过多的bullet points转换为流畅的段落叙述
使书籍看起来像传统的学术专著，而不是AI生成的报告
"""

import re
import os

def convert_itemize_to_prose(content):
    """将itemize列表转换为段落"""
    
    def process_itemize(match):
        full_match = match.group(0)
        items_text = match.group(1)
        
        # 提取所有item内容
        items = re.findall(r'\\item\s*(.*?)(?=\\item|$)', items_text, re.DOTALL)
        items = [item.strip() for item in items if item.strip()]
        
        if not items:
            return full_match
        
        # 如果只有1-2个item，转为简单段落
        if len(items) <= 2:
            result = []
            for item in items:
                # 清理item内容
                item = re.sub(r'\s+', ' ', item).strip()
                if item:
                    result.append(item)
            return ' '.join(result) + '\n\n'
        
        # 如果有3-10个item，转为连贯的段落
        if len(items) <= 10:
            result = []
            for i, item in enumerate(items):
                item = re.sub(r'\s+', ' ', item).strip()
                if not item:
                    continue
                # 检查是否以粗体标题开头
                bold_match = re.match(r'\\textbf\{([^}]+)\}[：:]*\s*(.*)', item)
                if bold_match:
                    title = bold_match.group(1)
                    rest = bold_match.group(2).strip()
                    if rest:
                        result.append(f"\\textbf{{{title}}}：{rest}")
                    else:
                        result.append(f"\\textbf{{{title}}}")
                else:
                    result.append(item)
            
            # 用句号或分号连接
            text = '；'.join(result)
            if not text.endswith('。') and not text.endswith('；'):
                text += '。'
            return text + '\n\n'
        
        # 如果超过10个item，保留列表格式但简化
        # 这种情况可能真的需要列表
        return full_match
    
    # 匹配itemize环境
    pattern = r'\\begin\{itemize\}(?:\[[^\]]*\])?\s*(.*?)\\end\{itemize\}'
    content = re.sub(pattern, process_itemize, content, flags=re.DOTALL)
    
    return content

def convert_enumerate_to_prose(content):
    """将enumerate列表转换为段落"""
    
    def process_enumerate(match):
        full_match = match.group(0)
        items_text = match.group(1)
        
        # 提取所有item内容
        items = re.findall(r'\\item\s*(.*?)(?=\\item|$)', items_text, re.DOTALL)
        items = [item.strip() for item in items if item.strip()]
        
        if not items:
            return full_match
        
        # 检查是否包含嵌套列表，如果有则保留
        if '\\begin{itemize}' in items_text or '\\begin{enumerate}' in items_text:
            return full_match
        
        # 如果只有1-3个item，转为连贯段落
        if len(items) <= 3:
            result = []
            ordinals = ['首先', '其次', '第三', '最后']
            for i, item in enumerate(items):
                item = re.sub(r'\s+', ' ', item).strip()
                if not item:
                    continue
                # 检查是否以粗体标题开头
                bold_match = re.match(r'\\textbf\{([^}]+)\}[：:]*\s*(.*)', item)
                if bold_match:
                    title = bold_match.group(1)
                    rest = bold_match.group(2).strip()
                    if i < len(ordinals):
                        if rest:
                            result.append(f"{ordinals[i]}，{title}——{rest}")
                        else:
                            result.append(f"{ordinals[i]}是{title}")
                    else:
                        if rest:
                            result.append(f"{title}——{rest}")
                        else:
                            result.append(title)
                else:
                    if i < len(ordinals) and i < len(items) - 1:
                        result.append(f"{ordinals[i]}，{item}")
                    elif i == len(items) - 1 and len(items) > 1:
                        result.append(f"最后，{item}")
                    else:
                        result.append(item)
            
            text = '。'.join(result)
            if not text.endswith('。'):
                text += '。'
            return text + '\n\n'
        
        # 如果有4-5个item，尝试转换为带序号的段落
        if len(items) <= 20: # 增加处理数量
            result = []
            is_long_items = any(len(item) > 50 for item in items)
            
            for i, item in enumerate(items):
                item = re.sub(r'\s+', ' ', item).strip()
                if not item:
                    continue
                # 用"第一、第二"等表达
                num_words = ['第一', '第二', '第三', '第四', '第五', '第六', '第七', '第八', '第九', '第十',
                             '第十一', '第十二', '第十三', '第十四', '第十五', '第十六', '第十七', '第十八', '第十九', '第二十']
                prefix = num_words[i] if i < len(num_words) else f'第{i+1}'
                
                bold_match = re.match(r'\\textbf\{([^}]+)\}[：:]*\s*(.*)', item)
                if bold_match:
                    title = bold_match.group(1)
                    rest = bold_match.group(2).strip()
                    if rest:
                        result.append(f"\\textbf{{{prefix}，{title}}}：{rest}")
                    else:
                        result.append(f"\\textbf{{{prefix}，{title}}}")
                else:
                    result.append(f"\\textbf{{{prefix}}}，{item}")
            
            if is_long_items:
                # 如果条目较长，使用分段
                return '\n\n'.join(result) + '\n\n'
            else:
                # 如果条目较短，合并为一段
                text = '。'.join(result)
                if not text.endswith('。'):
                    text += '。'
                return text + '\n\n'
        
        # 超过10个item，保留列表
        return full_match
    
    # 匹配enumerate环境（不包含嵌套）
    pattern = r'\\begin\{enumerate\}(?:\[[^\]]*\])?\s*(.*?)\\end\{enumerate\}'
    
    # 多次处理，因为可能有嵌套
    prev_content = None
    while prev_content != content:
        prev_content = content
        content = re.sub(pattern, process_enumerate, content, flags=re.DOTALL)
    
    return content

def convert_inline_lists_to_prose(content):
    """将行内列表（第一，... 第二，...）转换为段落"""
    
    # 处理 "。第二，" 这种模式，将其分段并加粗
    markers = ['第二', '第三', '第四', '第五', '第六', '第七', '第八', '第九', '第十']
    
    for m in markers:
        # 匹配 "。第X，" 或 "；第X，"
        # 替换为 "。\n\n\textbf{第X}，"
        # 注意：这里假设"第X，"后面紧跟的是内容
        pattern = r'([。；;])\s*(' + m + '，)'
        content = re.sub(pattern, r'\1\n\n\\textbf{' + m + '}，', content)
    
    # 处理段落开头的 "第一，"
    # 这需要更小心，只在后面有 "第二，" 的情况下才处理
    # 或者我们可以简单地将所有段落开头的 "第一，" 加粗（如果它还没加粗）
    
    # 查找 "\n\n第一，" 或文件开头的 "第一，"
    # pattern = r'(^|\n\n)\s*(第一，)'
    # content = re.sub(pattern, r'\1\\textbf{第一}，', content)
    
    # 更好的方法是：如果我们在上面做了分段，那么原来的段落现在可能包含 "第一，... \n\n \textbf{第二}，..."
    # 所以我们可以查找后面跟着 "\n\n\textbf{第二}，" 的 "第一，"
    
    if '\\textbf{第二}，' in content:
        # 尝试加粗前面的 "第一，"
        # 匹配： (段落开始)第一，... (直到遇到换行)
        content = re.sub(r'(^|\n\n)\s*第一，', r'\1\\textbf{第一}，', content)
        
        # 还有一种情况： "标题：第一，..."
        content = re.sub(r'([：:])\s*第一，', r'\1\n\n\\textbf{第一}，', content)

    return content

def clean_formatting(content):
    """清理格式"""
    # 移除过多空行
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # 修复句号重复
    content = re.sub(r'。。+', '。', content)
    
    # 修复分号后面紧跟句号
    content = re.sub(r'；。', '。', content)
    
    # 清理空的粗体
    content = re.sub(r'\\textbf\{\s*\}', '', content)
    
    return content

def process_file(filepath):
    """处理单个文件"""
    print(f"处理: {os.path.basename(filepath)}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 统计原有列表数量
    orig_itemize = len(re.findall(r'\\begin\{itemize\}', content))
    orig_enumerate = len(re.findall(r'\\begin\{enumerate\}', content))
    
    # 转换
    content = convert_itemize_to_prose(content)
    content = convert_enumerate_to_prose(content)
    content = convert_inline_lists_to_prose(content)
    content = clean_formatting(content)
    
    # 统计转换后
    new_itemize = len(re.findall(r'\\begin\{itemize\}', content))
    new_enumerate = len(re.findall(r'\\begin\{enumerate\}', content))
    
    print(f"  itemize: {orig_itemize} -> {new_itemize}")
    print(f"  enumerate: {orig_enumerate} -> {new_enumerate}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return (orig_itemize, orig_enumerate, new_itemize, new_enumerate)

def main():
    files = [
        'chapter1_new.tex',
        'chapter1.tex',
        'chapter2_new.tex', 
        'chapter2.tex',
        'chapter3_new.tex',
        'chapter3.tex',
        'chapter4_new.tex',
        'chapter4.tex',
        'chapter5_new.tex',
        'chapter5.tex',
        'chapter6_new.tex',
        'chapter6.tex',
        'chapter7_new.tex',
        'chapter7.tex',
        'chapter8_new.tex',
        'chapter9_talent.tex',
        'chapter10_investment.tex',
        'chapter11_frontier.tex',
        'chapter12_newsecurity.tex',
        'foreword.tex',
        'preface.tex',
        'afterword.tex',
        'appendix.tex',
    ]
    
    total_orig_i, total_orig_e = 0, 0
    total_new_i, total_new_e = 0, 0
    
    for filename in files:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            oi, oe, ni, ne = process_file(filepath)
            total_orig_i += oi
            total_orig_e += oe
            total_new_i += ni
            total_new_e += ne
    
    print(f"\n总计:")
    print(f"  itemize: {total_orig_i} -> {total_new_i}")
    print(f"  enumerate: {total_orig_e} -> {total_new_e}")
    print(f"  总列表减少: {total_orig_i + total_orig_e - total_new_i - total_new_e}")

if __name__ == '__main__':
    main()
