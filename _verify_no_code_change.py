"""
_verify_no_code_change.py -- alat bantu VERIFIKASI, bukan bagian model.

Tujuan: membuktikan bahwa penambahan komentar TIDAK mengubah logika code
sama sekali. Caranya membandingkan Abstract Syntax Tree (AST) file di
_backup_original/ (versi sebelum dikomentari) dengan file aktif sekarang.

AST = struktur program hasil parsing Python. Komentar ('#') dibuang total
oleh parser, jadi kalau AST kedua versi identik --> yang berubah HANYA
komentar, tidak ada satu pun statement/ekspresi yang tersentuh.

Jalankan:  python _verify_no_code_change.py
"""
import ast
import os
import sys
import types

dirPath = os.path.dirname(os.path.abspath(__file__))
BACKUP = os.path.join(dirPath, '_backup_original')

FILES = [
    'HelpDKT_Model.py',
    'RunModel.py',
    'inference.py',
    'GetData.py',
    'data.py',
    os.path.join('Encoder', 'LSTM_Encoder.py'),
    os.path.join('Encoder', 'Transformer_Encoder.py'),
    os.path.join('decoder', 'Decoder.py'),
]


def read(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def dump_ast(path):
    """Baca file -> string AST kanonik (komentar & spasi otomatis hilang)."""
    return ast.dump(ast.parse(read(path)), include_attributes=False)


def dump_bytecode(path):
    """Baca file -> tanda tangan BYTECODE (termasuk semua fungsi bersarang).

    Ini verifikasi terkuat: kalau bytecode identik, instruksi yang benar-benar
    dijalankan Python sama persis -- bukan sekadar struktur code-nya mirip.
    """
    def walk(code, out):
        out.append((code.co_name, code.co_argcount, code.co_kwonlyargcount,
                    code.co_nlocals, code.co_flags, code.co_code,
                    code.co_names, code.co_varnames,
                    code.co_freevars, code.co_cellvars))
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                walk(const, out)
            else:
                out.append(repr(const))
        return out

    return walk(compile(read(path), path, 'exec'), [])


def main():
    ok = True
    print('%-40s %-10s %s' % ('FILE', 'AST', 'BYTECODE'))
    for rel in FILES:
        orig = os.path.join(BACKUP, rel)
        curr = os.path.join(dirPath, rel)
        if not os.path.exists(orig):
            print('SKIP (tidak ada backup): %s' % rel)
            continue
        same_ast = dump_ast(orig) == dump_ast(curr)
        same_bc = dump_bytecode(orig) == dump_bytecode(curr)
        ok = ok and same_ast and same_bc
        print('%-40s %-10s %s' % (
            rel,
            'IDENTIK' if same_ast else 'BERBEDA',
            'IDENTIK' if same_bc else 'BERBEDA'))
    print('\nHASIL: %s' % ('semua code identik, hanya komentar yang bertambah'
                           if ok else 'ADA CODE YANG BERUBAH - periksa lagi!'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
