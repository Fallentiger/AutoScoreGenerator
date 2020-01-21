import librosa
# from array import array
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
# from matplotlib import use as mpluse
from scipy.signal import argrelextrema
# import sys
import os
from collections import Counter
import math




class ScoreGenerator:

    def __init__(self, audioPath, bpm, maxNodePerBeat=4, outDir=None):
        '''
        audioPath: 输入音频路径
        bpm: 实际速度
        maxNodePerBeat: 每拍音符数
        outDir: 输出目录
        '''
        self.audioPath = audioPath
        self.bpm = bpm
        self.maxNodePerBeat = maxNodePerBeat
        self.outDir = outDir
        self.binFacotr=1
        if self.outDir == None:
            self.prefix = os.path.splitext(self.audioPath)
        else:
            if not os.path.exists(self.outDir):
                os.makedirs(self.outDir)
            prefix=os.path.splitext(os.path.split(self.audioPath)[-1])[0]
            self.prefix = os.path.join(self.outDir, prefix)

        oo = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.oo7 = np.array([o+str(i) for i in range(1, 8) for o in oo]+[' '])

        self.loadMusic()
        self.getScore()
        self.plotScore()
        self.writeNodes()
        self.writeScore()

    def getCol(self, mtx):
        '''
        返回生成矩阵的每一列的生成器
        '''
        for i in range(mtx.shape[1]):
            yield mtx[:, i]

    def loadMusic(self, audioPath=None, trimEpt=True):
        '''
        读取音乐
        '''
        if audioPath == None:
            audioPath = self.audioPath

        y, sr = librosa.load(audioPath, sr=22050)
        print('file loaded sharp: ', y.shape, ',sampling rate:  %i hz' % sr)
        if trimEpt:
            y, offsetIndex = librosa.effects.trim(y)
            print('filtered sharp: ', y.shape, ',sampling rate:  %i hz' % sr)
        else:
            offsetIndex = (0, len(y))
        self.y = y
        self.sr = sr
        self.offsetIndex = offsetIndex

    def getScore(self):
        '''
        得到音符索引矩阵
        '''
        y = self.y
        sr = self.sr

        S = librosa.stft(y, n_fft=2048*self.binFacotr)
        S = np.abs(S)
        chroma = librosa.feature.chroma_stft(S=S**2, sr=sr)

        C = np.abs(librosa.cqt(y, sr=sr, hop_length=512*self.binFacotr))

        baseIndex = []
        chromaCols = self.getCol(chroma)
        for chromaCol in chromaCols:
            tempIndex = np.where(np.all(np.array([chromaCol > np.sum(
                chromaCol)*0.1, chromaCol > np.max(chromaCol)*0.5], dtype=bool), axis=0))[0]
            tempIndex = sorted(
                tempIndex, key=lambda x: chromaCol[x], reverse=True)
            baseIndex.append(tempIndex[:5])
        self.baseIndex = baseIndex

        Cdb = librosa.amplitude_to_db(C, ref=np.max)

        Cdb = Cdb-np.min(Cdb)
        realIndex = []
        for timeCounter in range(len(baseIndex)):
            tempIndex = baseIndex[timeCounter]
            ns = []
            for index in tempIndex:
                crtNodes = Cdb[index::12, timeCounter]
                if np.max(crtNodes) > 0:
                    divPct = np.abs(
                        (crtNodes-np.max(crtNodes))/np.max(crtNodes)) < 0.1
                    ns.append(divPct.argmax()*12+index)
            ns += [-1]*(5-len(ns))
            realIndex.append(ns)

        realIndex = np.array([np.array(row) for row in realIndex])
        self.realIndex = realIndex.T

    def plotScore(self):
        '''
        得出处理过频谱图像
        本质上是热图，每个点的值(颜色)不代表实际频谱值
        '''
        colorTable = list(range(80, -41, -25))
        newHeatmap = np.full((7*12, self.realIndex.shape[1]), -80)
        for i in range(self.realIndex.shape[1]):
            tempIndexes = self.realIndex[:,i]
            for j in range(len(tempIndexes)):
                if tempIndexes[j] > 0:
                    newHeatmap[tempIndexes[j], i] = colorTable[j]

        plt.figure(figsize=(50, 5), dpi=300)
        librosa.display.specshow(
            newHeatmap, sr=self.sr, x_axis='time', y_axis='cqt_note')
        plt.grid(b=0.1, which="both")
        plt.colorbar(format='%+2.0f dB')
        plt.title('Score')
        plt.tight_layout()
        plt.savefig(self.prefix+'_score.png')
        plt.close()

    def printMus(self, mtx, binIndex, timePerBin, timeOffset, destFile,inGroup=False):
        '''
        输出函数
        mtx: 二维numpy.array
        binIndex: 组序号
        timePerBin: 每组时间
        timeOffset: 起始位置时间偏移
        destFile: 输出文件IO类型
        inGroup: [bool|int]是否分组/分组大小
        '''
        fmtStr = '''{:>4}'''
        if inGroup:
            if isinstance(inGroup,bool):
                inGroup=4
            fmtStr = fmtStr*int(inGroup)+' |'
            fmtStr = fmtStr*math.ceil(mtx.shape[1]/inGroup)
        else:
            fmtStr=fmtStr*int(mtx.shape[1])
        for line in mtx:
            print(('''{:>8.3f} - {:>8.3f}'''+fmtStr).format(
                timeOffset+binIndex*timePerBin, timeOffset+(binIndex+mtx.shape[1])*timePerBin, *line), file=destFile)

    def writeNodes(self):
        '''
        按照音符矩阵输出
        按照每30个音符分为若干组
        每组最左端是改组的时间区间
        每列由上到下代表从可能性高到低排序的音符
        '''
        nodes=self.oo7[self.realIndex]
        # nodes = np.array([np.array([oo7[i] for i in row])
        #                   for row in self.realIndex])
        duration = librosa.get_duration(self.y, self.sr)
        timePerBin = duration/nodes.shape[1]
        timeOffset = self.offsetIndex[0]/self.sr
        # nodes = nodes.T
        nodesPerLine = 30
        destFile = open(self.prefix+'_nodes.txt', 'w', encoding='utf-8')
        for i in range(0, nodes.shape[1], nodesPerLine):
            self.printMus(nodes[:, i:i+nodesPerLine], i,
                          timePerBin, timeOffset, destFile)
            print('-'*(8*2+3+nodesPerLine*4), file=destFile)

        destFile.close()

    def writeScore(self):
        '''
        输出谱
        默认以一个小节为一组(4/4拍)，每列代表一个1/16时值，每4个(1拍)使用|划分
        其余格式同writeNodes
        '''
        binPerNode=self.sr*60/(self.bpm*self.maxNodePerBeat)/(512*self.binFacotr)
        expectedNodesNum = math.ceil(self.realIndex.shape[1]/binPerNode)
        score = np.full((5, expectedNodesNum),-1)
        for i in range(expectedNodesNum):
            tempNode = self.realIndex[:, math.floor(
                i*binPerNode):math.ceil((i+1)*binPerNode)]
            tempNode = tempNode[tempNode > 0]
            tempNode = Counter(tempNode)

            tempNode = np.array(sorted([kv for kv in tempNode.items() if kv[1] > 0.5*max(tempNode.values())],
                              key=lambda x: x[1], reverse=True))[:5,0]

            # tempNode += [-1]*(5-len(tempNode))
            # score[:,i] = np.array(tempNode)
            for j in range(len(tempNode)):
                score[j, i] = tempNode[j]
        score=self.oo7[score]
        self.score=score

        duration = librosa.get_duration(self.y, self.sr)
        timePerBin = duration/expectedNodesNum
        timeOffset = self.offsetIndex[0]/self.sr
        nodesPerLine = 16
        with open(self.prefix+'_score.txt', 'w', encoding='utf-8') as destFile:
            for i in range(0, score.shape[1], nodesPerLine):
                self.printMus(score[:, i:i+nodesPerLine], i,
                            timePerBin, timeOffset, destFile,4)
                print('-'*(8*2+3+nodesPerLine*4), file=destFile)


    def detectBPM(self,maxBpm=200):
        '''
        求取可能的BPM
        maxBpm: 最高可能的速度
        输出节拍的分布小提琴图，纵轴为bpm，横轴为分布密度
        最宽处即为可能bpm

        默认最短时值1/16音符(四分音符为一拍)
        如时值更短，可以将maxBpm乘以2**n

        性能极度不友好，可能需要几小时，视音频长度、采样率、最大bpm而定
        裁剪音频、降低采样率、增加maxBpm可减少时间
        '''
        yabs = np.abs(self.y)
        sr=self.sr
        beatIdxes = argrelextrema(
            yabs, np.greater, order=int(sr*60/maxBpm))[0]
        beatIdxes = beatIdxes[beatIdxes > sr*60/maxBpm]
        beatDuration = beatIdxes[1:]-beatIdxes[:-1]

        bpms = 60*sr/beatDuration
        for i in range(len(bpms)):
            if bpms[i] < 70:
                pass
            elif bpms[i] < 140:
                bpms[i] = bpms[i]/2
            elif bpms[i] < 280:
                bpms[i] = bpms[i]/4
            elif bpms[i] < 560:
                bpms[i] = bpms[i]/8
            else:
                bpms[i] = bpms[i]/16

        plt.figure(figsize=(8, 20))
        plt.violinplot(bpms)
        plt.yticks(np.arange(int(np.min(bpms)), np.max(bpms), 1))
        plt.grid(b=0.1, which='both')
        plt.savefig(self.prefix+'_bpm.png')
