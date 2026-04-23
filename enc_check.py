import pathlib
for p in pathlib.Path('core').glob('*.py'):
    try:
        p.read_text(encoding='utf-8')
        print('OK', p)
    except Exception as e:
        print('FAIL', p, e)
