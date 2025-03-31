# pip install soundata

import soundata

urbansound8k = soundata.initialize('urbansound8k')  # get the urbansound8k dataset
urbansound8k.download()  # download orchset
urbansound8k.validate()  # validate orchset
clip = urbansound8k.choice_clip()  # load a random clip
print(clip)  # see what data a clip contains
urbansound8k.clip_ids()  # load all clip ids