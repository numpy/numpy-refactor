import re
import subprocess


def remove_long_path():
    path = 'mtrand.c'
    pat = re.compile(r'"[^"]*mtrand\.pyx"')
    code = open(path).read()
    code = pat.sub(r'"mtrand.pyx"', code)
    open(path, 'w').write(code)


def main():
    assert subprocess.call(['cython', 'mtrand.pyx']) == 0
    remove_long_path()


if __name__ == '__main__':
    main()
