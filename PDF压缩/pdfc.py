#!/usr/bin/python3
# Author: jadger
# 29/04/2020

"""
Simple python wrapper script to use ghoscript function to compress PDF files.

Compression levels:
    0: default
    1: prepress
    2: printer
    3: ebook
    4: screen

Dependency: Ghostscript.
On MacOSX install via command line `brew install ghostscript`.
"""

import argparse
import subprocess
import os
import shutil
import sys

def get_ghostscript_path():
    gs_path = 'E://install//gs10.04.0//bin//gswin64c.exe'
    if os.path.exists(gs_path):
        return gs_path
    raise FileNotFoundError(f'No GhostScript executable was found at {gs_path}')

def compress(input_file_path, output_file_path=None, power=4, bulk=False):
    """Function to compress PDF via Ghostscript command line interface"""
    quality = {
        0: '/default',
        1: '/prepress',
        2: '/printer',
        3: '/ebook',
        4: '/screen'
    }

    if not bulk:
        parent_path = os.path.dirname(input_file_path)
        output_folder = os.path.join(parent_path if parent_path not in ['', '.'] else '.', 'compressed')

        count = 1
        while os.path.exists(output_folder):
            output_folder = f'compressed_{count}'
            count += 1
        os.mkdir(output_folder)

        output_file_path = os.path.join(output_folder, os.path.basename(input_file_path))

    # Basic controls
    if not os.path.isfile(input_file_path):
        print("错误: 请输入正确的pdf文件的路径")
        sys.exit(1)

    if input_file_path.split('.')[-1].lower() != 'pdf':
        print("错误: 你输入的文件可能不是PDF")
        sys.exit(1)

    gs = get_ghostscript_path()
    print("Compress PDF...", input_file_path)
    initial_size = os.path.getsize(input_file_path)
    try:
        subprocess.run([gs, '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
                        '-dPDFSETTINGS={}'.format(quality[power]),
                        '-dNOPAUSE', '-dQUIET', '-dBATCH',
                        '-dColorImageDownsampleType=/Bicubic',
                        '-dColorImageResolution=30',
                        '-dGrayImageDownsampleType=/Bicubic',
                        '-dGrayImageResolution=30',
                        '-dMonoImageDownsampleType=/Bicubic',
                        '-dMonoImageResolution=30',
                        '-dDownsampleColorImages=true',
                        '-dDownsampleGrayImages=true',
                        '-dDownsampleMonoImages=true',
                        '-dColorImageFilter=/DCTEncode',
                        '-dGrayImageFilter=/DCTEncode',
                        '-dCompressFonts=true',
                        '-dAutoFilterColorImages=false',
                        '-dAutoFilterGrayImages=false',
                        '-sOutputFile={}'.format(output_file_path),
                        input_file_path],
                       check=True)
    except subprocess.CalledProcessError as e:
        print(f"GhostScript 执行失败: {e}")
        sys.exit(1)

    final_size = os.path.getsize(output_file_path)
    ratio = 1 - (final_size / initial_size)
    print("Compression by {0:.0%}.".format(ratio))
    show_size = final_size / 1024
    if show_size < 1024:
        print("Final file size is {0:.1f}KB".format(show_size))
    else:
        show_size = show_size / 1024
        print("Final file size is {0:.1f}MB".format(show_size))
    print("----" * 5)
    print()

def bulk_compress(input_folder, power=4):
    print('<<<<<<<< bulk_compress run ... >>>>>>>>>>')
    if not os.path.isdir(input_folder):
        print("错误：你输入的不是路径")
        sys.exit(1)

    output_folder = 'compressed'
    count = 1
    while os.path.exists(output_folder):
        output_folder = f'compressed_{count}'
        count += 1
    os.mkdir(output_folder)

    items = [item for item in os.listdir(input_folder) if item.split('.')[-1].lower() == 'pdf']

    for item in items:
        output_file_path = os.path.join(output_folder, item)
        input_file_path = os.path.join(input_folder, item)
        compress(input_file_path, output_file_path, power, bulk=True)

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-f', '--file', help='Relative or absolute path of the input PDF file name')
    parser.add_argument('-r', '--route', help='Relative or absolute path PDF folder')
    parser.add_argument('-p', '--power', type=int, choices=range(0, 5), default=4, help='Compression power level (0-4)')
    args = parser.parse_args()

    if not args.file and not args.route:
        print('错误：请输入pdf的文件名或pdf所在文件夹')
        sys.exit(1)
    if args.file and args.route:
        print('错误：要么压缩单一文件，要么批量压缩')
        sys.exit(1)

    if args.file:
        compress(args.file, power=args.power)
    if args.route:
        bulk_compress(args.route, power=args.power)

if __name__ == '__main__':
    main()
