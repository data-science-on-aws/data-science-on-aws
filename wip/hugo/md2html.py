import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown"])

import markdown

with open('README.md', 'r') as f:
    text = f.read()
    html = markdown.markdown(text)

with open('index.html', 'w') as f:
    f.write(html)
    print('Wrote index.html')

