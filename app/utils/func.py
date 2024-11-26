def try_or_default(fn, default):
    try:
        return fn()
    except:
        return default
