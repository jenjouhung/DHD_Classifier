import os
import os.path

TESTDATA_DIR = "../test_data/"
POSITIVE_DIR = "../test_data/positive/"
NEGATIVE_DIR = "../test_data/negative/"

OUT_POSITIVE_FILE = "../test_data/positive.txt"
OUT_NEGATIVE_FILE = "../test_data/negative.txt"

with open(OUT_POSITIVE_FILE, 'w', encoding='utf-8') as outf:
    for filename in os.listdir(POSITIVE_DIR):
        with open(os.path.join(POSITIVE_DIR, filename), 'r', encoding='utf-8') as f:
            content = ''
            for line in f:
                if line.strip() == '':
                    if len(content) > 0:
                        outf.write(content)
                        outf.write('\n')
                        content = ''
                else:
                    content += line.strip()
            if len(content) > 0:
                outf.write(content)
                outf.write('\n')
        print(filename)

with open(OUT_NEGATIVE_FILE, 'w', encoding='utf-8') as outf:
    for filename in os.listdir(NEGATIVE_DIR):
        with open(os.path.join(NEGATIVE_DIR, filename), 'r', encoding='utf-8') as f:
            content = ''
            for line in f:
                if line.strip() == '':
                    if len(content) > 0:
                        outf.write(content)
                        outf.write('\n')
                        content = ''
                else:
                    content += line.strip()
            if len(content) > 0:
                outf.write(content)
                outf.write('\n')
        print(filename)

