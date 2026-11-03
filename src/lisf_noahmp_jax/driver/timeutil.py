"""Calendar helpers matching LIS_timeMgr conventions (Gregorian, no leap handling beyond Feb)."""
import datetime as dt


def add_minutes(t, m):
    return t + dt.timedelta(minutes=m)


def doy(t):
    return t.timetuple().tm_yday


def gmt_hours(t):
    return t.hour + t.minute / 60.0 + t.second / 3600.0


def nfeb(y):
    return 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28


def month_d(a12, y, m, d):
    """LIS/Noah month_d: 12 monthly values valid on the 15th, linear in day-of-month between them.
    a12 may be an array whose first axis has length 12."""
    nd = [31, nfeb(y), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if d == 15:
        return a12[m - 1]
    if d < 15:
        prev = m - 1 or 12
        post = m
        f = (nd[prev - 1] - 15 + d) / nd[prev - 1]
    else:
        prev = m
        post = m + 1 if m < 12 else 1
        f = (d - 15) / nd[prev - 1]
    return a12[prev - 1] * (1.0 - f) + a12[post - 1] * f
