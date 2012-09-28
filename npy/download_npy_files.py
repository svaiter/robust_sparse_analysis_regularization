#!/usr/bin/env python

import urllib

urls = [
    ("https://dl.dropbox.com/u/19248649/fused_ratio_eta.npy",
        "fused_ratio_eta.npy"),
    ("https://dl.dropbox.com/u/19248649/fused_eps_eta.npy",
        "fused_eps_eta.npy")
]

for url,filename in urls:
    urllib.urlretrieve(url, filename)