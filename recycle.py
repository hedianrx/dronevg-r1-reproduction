"""把文件/目录送入 Windows 回收站，不依赖 COM 与短路径 API。

用 SHFileOperationW（shell32），FOF_ALLOWUNDO 让操作可撤销。
用法: python recycle.py <path> [<path> ...]
"""
import ctypes
import ctypes.wintypes as wt
import os
import sys

FO_DELETE = 3
FOF_SILENT = 0x0004
FOF_NOCONFIRMATION = 0x0010
FOF_ALLOWUNDO = 0x0040
FOF_NOERRORUI = 0x0400


class SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [
        ("hwnd", wt.HWND),
        ("wFunc", wt.UINT),
        ("pFrom", wt.LPCWSTR),
        ("pTo", wt.LPCWSTR),
        ("fFlags", ctypes.c_uint16),
        ("fAnyOperationsAborted", wt.BOOL),
        ("hNameMappings", ctypes.c_void_p),
        ("lpszProgressTitle", wt.LPCWSTR),
    ]


def recycle(path):
    p = os.path.abspath(path)
    if not os.path.exists(p):
        return None, "路径不存在"
    # 双 NULL 结尾
    src = p + "\0\0"
    op = SHFILEOPSTRUCTW()
    op.hwnd = None
    op.wFunc = FO_DELETE
    op.pFrom = src
    op.pTo = None
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
    op.fAnyOperationsAborted = False
    op.hNameMappings = None
    op.lpszProgressTitle = None
    rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if rc != 0:
        return rc, "SHFileOperationW rc=%d" % rc
    if op.fAnyOperationsAborted:
        return rc, "用户中止"
    return 0, "ok"


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        rc, msg = recycle(arg)
        print("%-50s rc=%-3s %s  exists_after=%s" % (arg, rc, msg, os.path.exists(arg)))
