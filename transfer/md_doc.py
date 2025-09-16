import pypandoc
import os
pypandoc.download_pandoc()
# 输入输出路径
input_file = "NJ023.md"
output_file = "NJ023.docx"

# 调用 pandoc 转换
pypandoc.convert_file(
    input_file,
    'docx',
    outputfile=output_file,
    extra_args=[
        '--mathml'   # 保留 LaTeX 数学公式为 Word 原生公式
    ]
)

print(f"转换完成 ✅ Word 文件已生成: {output_file}")
