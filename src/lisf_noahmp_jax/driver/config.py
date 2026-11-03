"""Minimal reader for LIS 'label: value ...' configuration files."""
import re


class Config:
    def __init__(self, path):
        self.d = {}
        for ln in open(path):
            ln = ln.split("#", 1)[0].rstrip()
            if ":" not in ln:
                continue
            k, v = ln.split(":", 1)
            self.d[k.strip()] = v.strip()

    def s(self, key):
        return self.d[key].strip().strip('"')

    def f(self, key):
        return float(self.d[key].split()[0])

    def i(self, key):
        return int(float(self.d[key].split()[0]))

    def fl(self, key):
        return [float(x) for x in self.d[key].split()]
