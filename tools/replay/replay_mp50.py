"""Replay of real LIS Noah-MP 5.0 driver calls (dump from lis_patches/make_noahmp50_hook.py) through the 4.0.1 JAX port via lis/dump50.py.
Usage: python lis/replay_mp50.py <dumpdir> [max_records] [start] [rand:N|sample:N]   (same arguments as replay_mp.py)"""
import sys
import dumpio_mp, dump50
dumpio_mp.load = dump50.load50
import replay_mp
replay_mp.main()
