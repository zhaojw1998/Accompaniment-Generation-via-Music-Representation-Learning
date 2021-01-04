import xmltodict
import dicttoxml
import xml.dom.minidom
import collections
import os
from tqdm import tqdm
import sys
import pretty_midi as pyd

mid_data = pyd.PrettyMIDI('D:/Computer Music Research/score scrape and analysis/scrape_musescore/musescore_midi/0028.mid')
print(len(mid_data.instruments))