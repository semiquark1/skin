# misc utilities

# standard library
import sys
import base64
import binascii

# common numerical and scientific libraries
import pandas as pd


sensitive_colnames = ['name', 'path']

def myencode(s):
    s = str(s)
    try:
        s.encode('utf8')
        return s
    except UnicodeEncodeError:
        b = s.encode(sys.getfilesystemencoding(), 'surrogateescape')
        b_qp = binascii.b2a_qp(b)
        s_qp = b_qp.decode('ascii')
        return '&'+s_qp

def mydecode(s_qp):
    if s_qp.startswith('&'):
        b_qp = s_qp[1:].encode('ascii')
        b = binascii.a2b_qp(b_qp)
        s = b.decode(sys.getfilesystemencoding(), 'surrogateescape')
        return s
    else:
        return s_qp


def read_csv(path, **kwargs):
    df = pd.read_csv(path, **kwargs)
    for colname in sensitive_colnames:
        if colname in df.columns:
            df[colname] = df[colname].map(mydecode)
    return df

def write_csv(path, df, **kwargs):
    df = df.copy()
    for colname in sensitive_colnames:
        if colname in df.columns:
            df[colname] = df[colname].map(myencode)
    df.index = df.index.map(myencode)
    df.to_csv(path, **kwargs)

def write_pickle(path, df, **kwargs):
    df = df.copy()
    for colname in sensitive_colnames:
        if colname in df.columns:
            df[colname] = df[colname].map(myencode)
    df.index = df.index.map(myencode)
    df.to_pickle(path, **kwargs)
# vim: set sw=4 sts=4 expandtab :
