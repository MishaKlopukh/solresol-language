import enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.io import wavfile
import json

jupyter_nb_mode = False
try:
    assert jupyter_nb_mode
    from IPython.display import Audio
except:
    import sounddevice as sd
    def Audio(array,rate=44100):
        sd.play(array,rate)

class SolfegeSymbol(enum.Enum):
    DO,Do,do,D,d,p,o = 1,1,1,1,1,1,1
    RE,Re,re,R,r,k,e = 2,2,2,2,2,2,2
    MI,Mi,mi,M,m,i = 3,3,3,3,3,3
    FA,Fa,fa,F,f,a = 4,4,4,4,4,4
    SOL,Sol,sol,So,so,S,s,u = 5,5,5,5,5,5,5,5
    LA,La,la,L,l,au = 6,6,6,6,6,6
    SI,Si,si,TI,Ti,ti,T,t,ai = 7,7,7,7,7,7,7,7,7

    @property
    def freq(self,octave=4):
        notes = [261.63,293.66,329.63,349.23,392.00,440.00,493.88]
        return notes[self.value-1]*(2**(octave-4))
    
    @property
    def shortname(self):
        names = 'drmfslt'
        return names[self.value-1]
    
    @property
    def sescons(self):
        names = 'pkmfslt'
        return names[self.value-1]

    @property
    def sesvowel(self):
        names = list('oeiau')+['ai','au']
        return names[self.value-1]
    
    def makeglyph(self,xy,scale=1,color='black',weight=2,doubler=False):
        x,y=xy
        if doubler:
            shape = [
                patches.FancyArrowPatch((x-scale/2,y+2*scale/6),(x-scale/2,y+4*scale/6),arrowstyle='-',color=color,linewidth=weight),
                patches.FancyArrowPatch((x-scale/6,y+scale/2),(x+scale/6,y+scale/2),arrowstyle='-',color=color,linewidth=weight),
                patches.FancyArrowPatch((x-scale/2,y+2*scale/6),(x-scale/2,y+4*scale/6),arrowstyle='-',color=color,linewidth=weight),
                patches.FancyArrowPatch((x-4*scale/6,y+scale/2),(x-scale/3,y+scale/2),arrowstyle='-',color=color,linewidth=weight),
                patches.FancyArrowPatch((x-scale/2,y-scale/6),(x-scale/2,y+scale/6),arrowstyle='-',color=color,linewidth=weight),
                patches.FancyArrowPatch((x-2*scale/6,y+scale/2),(x-4*scale/6,y+scale/2),arrowstyle='-',color=color,linewidth=weight),
                patches.FancyArrowPatch((x-4*scale/6,y-scale/2),(x-scale/3,y-scale/2),arrowstyle='-',color=color,linewidth=weight),
            ][self.value-1]
            attachment = xy
        else:
            shape, attachment = [
                (patches.Circle((x+scale/2,y),scale/2,fill=False,color=color,linewidth=weight),(x+scale,y)),
                (patches.FancyArrowPatch((x,y),(x,y-scale),arrowstyle='-',color=color,linewidth=weight),(x,y-scale)),
                (patches.Arc((x+scale/2,y),scale,scale,theta1=0.0,theta2=180.0,color=color,linewidth=weight),(x+scale,y)),
                (patches.FancyArrowPatch((x,y),(x+scale,y-scale),arrowstyle='-',color=color,linewidth=weight),(x+scale,y-scale)),
                (patches.FancyArrowPatch((x,y),(x+scale,y),arrowstyle='-',color=color,linewidth=weight),(x+scale,y)),
                (patches.Arc((x,y-scale/2),scale,scale,theta1=90.0,theta2=-90.0,color=color,linewidth=weight),(x,y-scale)),
                (patches.FancyArrowPatch((x,y),(x+scale,y+scale),arrowstyle='-',color=color,linewidth=weight),(x+scale,y+scale))
            ][self.value-1]
        return shape, attachment

def generate_note(frequency, duration, sample_rate=44100, amplitude=1, envelope_ratio=1/3):
    fmul = 2*frequency*np.pi/sample_rate
    note = np.sin(fmul*np.arange(sample_rate*duration))
    env_time = int(envelope_ratio*sample_rate*duration)
    envelope = np.concatenate((np.linspace(0,amplitude,env_time),amplitude*np.ones(int(sample_rate*duration-2*env_time)),np.linspace(amplitude,0,env_time)))
    return note*envelope

class SolresolWord():
    def __init__(self, word, syntax='default'):
        if isinstance(word, list):
            if isinstance(word[0], SolfegeSymbol):
                self.word = word
            elif isinstance(word[0], str):
                self.word = [SolfegeSymbol[s] for s in word]
            elif isinstance(word[0], int):
                self.word = [SolfegeSymbol(i) for i in word]
        elif isinstance(word, str):
            if syntax in ['ses','s']:
                self.word = [SolfegeSymbol[s] for s in word.replace('ai','l').replace('au','t')]
            elif syntax in ['num','#',0]:
                self.word = [SolfegeSymbol(int(s)) for s in word.strip('0')]
            elif syntax in ['full','default']:
                self.word = []
                while len(word) > 0:
                    if word.lower().startswith('sol') and not word.lower().startswith('sola'):
                        self.word.append(SolfegeSymbol.SOL)
                        word = word[3:]
                    else:
                        self.word.append(SolfegeSymbol[word[:2]])
                        word = word[2:]
        elif isinstance(word, int):
            self.word = [SolfegeSymbol(int(s)) for s in oct(word)[2:].strip('0')]
            
    def __repr__(self):
        return f"{type(self).__name__}(['"+"','".join(smb.name for smb in self.word)+"'])"
    def __getitem__(self,ix):
        return self.word.__getitem__(ix)
    def __len__(self):
        return len(self.word)
    def __iter__(self):
        return iter(self.word)
    @property
    def ses(self):
        if len(self) == 1:
            return self.word[0].sesvowel
        else:
            return ''.join(ltr.sescons if ix%2==0 else ltr.sesvowel for ix,ltr in enumerate(self.word))
    @property
    def fulltext(self):
        return ''.join(smb.name for smb in self.word).lower()
    def __str__(self):
        return self.fulltext
    @property
    def value(self):
        return int(''.join(str(ltr.value) for ltr in self.word),8)
    @property
    def definition(self):
        return solresol_dict[self.fulltext]
    def __int__(self):
        return self.value
    def melody(self, note_len=0.2, amplitude=1, envelope_ratio=0.2, sample_rate=44100):
        return np.concatenate([generate_note(ltr.freq,note_len,sample_rate,amplitude,envelope_ratio) for ltr in self.word])
    def draw(self,ax,color='black',weight=2,startpos=(0,0)):
        pos=startpos
        for ix,ltr in enumerate(self.word):
            if ltr==SolfegeSymbol.LA and (self.word[ix-1]==SolfegeSymbol.SI or self.word[ix-1]==SolfegeSymbol.DO) and ix>0:
                pos = (pos[0]+0.5,pos[1]+0.5)
            g,pos = ltr.makeglyph(pos,color=color,weight=weight,doubler=(ltr==self.word[ix-1] and ix>0))
            ax.add_patch(g)
        ax.axis('scaled')
        ax.axis('off')
        return pos[0]+2,startpos[1]

class Solresol():
    def __init__(self, text, syntax='default'):
        if isinstance(text,str):
            text = text.translate(str.maketrans('','','!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
            self.words = [SolresolWord(word,syntax) for word in text.split()]
        elif isinstance(text,list):
            self.words = [SolresolWord(word,syntax) for word in text]
        elif isinstance(text,int):
            sw = oct(text)[2:]
            self.words = [SolresolWord(int(sw[i:i+5],8)) for i in range(0,len(sw),5)]
    @property
    def fulltext(self):
        return ' '.join(word.fulltext for word in self.words)
    @property
    def ses(self):
        return ' '.join(word.ses for word in self.words)
    @property
    def numlist(self):
        return [int(word) for word in self.words]
    @property
    def value(self):
        return int(''.join(oct(num)[2:].ljust(5,'0') for num in self.numlist),8)
    def __int__(self):
        return self.value
    def __str__(self):
        return self.fulltext
    def __getitem__(self,ix):
        return self.words.__getitem__(ix)
    def __len__(self):
        return len(self.words)
    def __iter__(self):
        return iter(self.words)
    def __repr__(self):
        return f"Solresol('{str(self)}')"
    def melody(self, note_len=0.2, amplitude=1, envelope_ratio=0.2, gap_ratio=1, sample_rate=44100):
        notes = []
        for word in self.words:
            notes.append(word.melody(note_len, amplitude, envelope_ratio, sample_rate))
            notes.append(np.zeros(int(note_len*sample_rate*gap_ratio)))
        return np.concatenate(notes)
    def play(self, note_len=0.2, amplitude=1, envelope_ratio=0.2, gap_ratio=1):
        return Audio(self.melody(note_len, amplitude, envelope_ratio, gap_ratio, 44100),rate=44100)
    def draw(self,color='black',weight=2,subplot_mode=False,rowmax=5):
        if len(self) > 1 and subplot_mode:
            fig,axs = plt.subplots(len(self)//rowmax+1,(len(self)-1)%rowmax+1)
            for word,ax in zip(self.words,axs):
                word.draw(ax,color=color,weight=weight)
        else:
            fig,ax = plt.subplots()
            pos = (0,0)
            for word in self.words:
                pos = word.draw(ax,color=color,weight=weight,startpos=pos)
        return fig
    def translate(self,alldefs=False,random=False,ix=0):
        translation = []
        for word in self.words:
            if alldefs:
                translation.append(f'{word.fulltext}: ({word.definition})')
            else:
                dfn = word.definition.split(',')
                ix = np.random.randint(len(dfn)) if random else ix
                translation.append(dfn[ix].strip())
        return ' '.join(translation)

with open('solresol_dict.json') as f:
    solresol_dict = json.load(f)

dictionary_url = "https://docs.google.com/spreadsheets/d/1-3lBxMURGN4AtGG846kuVGVNuEiHewCT88PiBahnODA/edit#gid=0"
