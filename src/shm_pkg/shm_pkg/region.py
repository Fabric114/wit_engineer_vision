"""/dev/shm 上一个内存映射文件的 RAII 封装 —— 纯逻辑, 零 ROS 依赖。

用 open/ftruncate/mmap 把一个文件映射进地址空间, 之后按字节偏移读写。这里
使用/dev/shm (Linux 上就是 tmpfs, 纯内存), 语义上更接近「共享内存」。

生命周期: 生产者 create() 拥有该区域 (owner=True), 退出时 close() 会 unlink
删除文件; 消费者 open() 只映射不拥有, close() 只解除映射不删文件。这跟
ShmRegion 的 owner_ 标志、只有 owner 删文件是同一套约定。
"""

from __future__ import annotations

import mmap
import os
from pathlib import Path

from . import layout


class ShmRegion:
    """一块映射进本进程地址空间的共享内存。用完务必 close()。"""

    def __init__(self, name: str, size: int, mm: mmap.mmap, fd: int, owner: bool):
        self.name = name
        self.size = size
        self.mm = mm
        self._fd = fd
        self.owner = owner
        self._closed = False

    # ---------- 工厂方法 ----------
    @classmethod
    def create(cls, name: str, size: int) -> "ShmRegion":
        """创建 (或覆盖) 并映射一块 size 字节的区域, 调用者成为 owner。"""
        path = cls.path_for(name)
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd, size)
            mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        except Exception:
            os.close(fd)
            path.unlink(missing_ok=True)
            raise
        return cls(name, size, mm, fd, owner=True)

    @classmethod
    def open(cls, name: str, size: int | None = None, writable: bool = False) -> "ShmRegion":
        """映射一块已存在的区域 (消费者)。size=None 表示用文件实际大小。

        writable=False 时以只读方式映射, 消费者无法误改共享内存。
        """
        path = cls.path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"共享内存区域不存在: {path}")
        flags = os.O_RDWR if writable else os.O_RDONLY
        fd = os.open(str(path), flags)
        try:
            actual = os.fstat(fd).st_size
            if size is None:
                size = actual
            elif actual < size:
                raise ValueError(f"区域过小: 需要 {size} 字节, 实际 {actual}")
            prot = mmap.PROT_READ | (mmap.PROT_WRITE if writable else 0)
            mm = mmap.mmap(fd, size, mmap.MAP_SHARED, prot)
        except Exception:
            os.close(fd)
            raise
        return cls(name, size, mm, fd, owner=False)

    # ---------- 工具 ----------
    @staticmethod
    def path_for(name: str) -> Path:
        """区域名 -> /dev/shm 下的文件路径, 容忍传进来带前导 '/'。"""
        return Path(layout.SHM_DIR) / name.lstrip("/")

    @staticmethod
    def exists(name: str) -> bool:
        return ShmRegion.path_for(name).exists()

    def close(self) -> None:
        """解除映射并关闭 fd; 若本进程是 owner, 顺带 unlink 删除文件。"""
        if self._closed:
            return
        self._closed = True
        try:
            self.mm.close()
        finally:
            os.close(self._fd)
            if self.owner:
                self.path_for(self.name).unlink(missing_ok=True)

    def __enter__(self) -> "ShmRegion":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        # 兜底, 但正常路径应显式 close() —— __del__ 时机不可控
        try:
            self.close()
        except Exception:
            pass
