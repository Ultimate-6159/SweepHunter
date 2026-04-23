import pathlib
p = pathlib.Path('core/xauusd_hyper_core.py')
b = p.read_bytes()
# strip BOM if any
if b.startswith(b'\xef\xbb\xbf'):
    b = b[3:]
# detect: try utf-8 first, else cp1252
try:
    text = b.decode('utf-8')
    print('was utf-8')
except UnicodeDecodeError:
    text = b.decode('cp1252')
    print('was cp1252 -> converting to utf-8')
p.write_bytes(text.encode('utf-8'))
print('done')
import ast
ast.parse(text)
print('PARSE_OK')
