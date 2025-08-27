# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import os
import tiktoken

# Get tiktoken data files
tiktoken_data = []
try:
    import tiktoken_ext
    tiktoken_data.append((tiktoken_ext.__path__[0], 'tiktoken_ext'))
except ImportError:
    pass

a = Analysis([
    'server.py',
],
    pathex=['.'],
    binaries=[],
    datas=[
        ('core/*', 'core'),
        ('web_folder/*', 'web_folder'),
        ('rag_models/*', 'rag_models'),
        ('requirements.txt', '.'),
    ] + tiktoken_data,
    hiddenimports=[
        'tiktoken_ext.openai_public',
        'tiktoken_ext',
        'tiktoken',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='server')
