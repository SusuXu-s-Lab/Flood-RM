from __future__ import annotations

from contextlib import contextmanager


def iter_progress(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


class _NullProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, amount=1):
        return None

    def set_description_str(self, value, refresh=True):
        return None

    def set_postfix_str(self, value, refresh=True):
        return None


@contextmanager
def progress_bar(**kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield _NullProgress()
        return

    with tqdm(**kwargs) as bar:
        yield bar
