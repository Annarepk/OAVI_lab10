from scipy import interpolate, signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


def integralImg(img: np.array) -> np.array:
    integImg = np.zeros(shape=img.shape)
    integImg[0, 0] = img[0, 0]

    for x in range(1, img.shape[0]):
        integImg[x, 0] = img[x, 0] + integImg[x - 1, 0]

    for y in range(1, img.shape[1]):
        integImg[0, y] = img[0, y] + integImg[0, y - 1]

    for x in range(1, img.shape[0]):
        for y in range(1, img.shape[1]):
            integImg[x, y] = img[x, y] \
                                 - integImg[x - 1, y - 1] \
                                 + integImg[x - 1, y] \
                                 + integImg[x, y - 1]

    return integImg


def frameSum(integImg: np.array, x: int, y: int, frameSize: int):
    len = integImg.shape[1] - 1
    hight = integImg.shape[0] - 1

    halfFrame = frameSize // 2
    above = y - halfFrame - 1
    low = y + halfFrame
    left = x - halfFrame - 1
    right = x + halfFrame

    A = integImg[max(above, 0), max(left, 0)]
    B = integImg[max(0, above), min(len, right)]
    C = integImg[min(hight, low), max(left, 0)]
    D = integImg[min(hight, low), min(right, len)]

    if max(left + 1, 0) == 0 and max(above + 1, 0) == 0:
        return D
    elif max(left + 1, 0) == 0:
        return D - B
    elif max(above + 1, 0) == 0:
        return D - C

    return D - C - B + A


def culcMean(integImg: np.array, x: int, y: int, frameSize):
    square = frameSize ** 2
    s = frameSum(integImg, x, y, frameSize)
    return s // square


def changeSampleRate(path, newSampleRate=22050):
    audioPath = "audio/" + path
    oldSampleRate, oldAudio = wavfile.read(audioPath)

    if oldSampleRate != newSampleRate:
        duration = oldAudio.shape[0] / oldSampleRate

        timeOld = np.linspace(0, duration, oldAudio.shape[0])
        timeNew = np.linspace(0, duration, int(oldAudio.shape[0] * newSampleRate / oldSampleRate))

        interpolator = interpolate.interp1d(timeOld, oldAudio.T)
        newAudio = interpolator(timeNew).T

        wavfile.write("audio/results/" + path, newSampleRate, np.round(newAudio).astype(oldAudio.dtype))


def formants(freqs, integralSpec, x, frameSize):
    res = [0] * integralSpec.shape[0]

    for i in range(1, integralSpec.shape[0], frameSize):
        res[i] = culcMean(integralSpec, x, i, frameSize)

    origin = res.copy()
    res.sort()

    res = res[-3:]

    return list(map(lambda power: (int(freqs[origin.index(power)]), int(power)), res))


def allFformants(freqs, integralSpec, frameSize):
    res = set()
    for i in range(integralSpec.shape[1]):
        formant = formants(freqs, integralSpec, i, frameSize)
        form = list(map(lambda bind: bind[0], formant))
        for j in range(3):
            res.add(form[j])

    res.discard(0)
    return res


def power(freqs, integralSpec, frameSize, formantS):
    power = dict()
    for i in formantS:
        power[i] = 0

    for i in range(integralSpec.shape[1]):
        for j in formants(freqs, integralSpec, i, frameSize):
            if (j[0] != 0):
                power[j[0]] += j[1]

    return power

def spectrogramPlot(samples, sampleRate, t=11000):
    if len(samples.shape) > 1:
        samples = np.mean(samples, axis=1)

    frequencies, times, mySpectrogram = signal.spectrogram(samples, sampleRate, scaling='spectrum', window=('hann'), nperseg=1024)
    spec = np.log10(mySpectrogram + 1e-12)
    plt.pcolormesh(times, frequencies, spec, shading='gouraud', vmin=spec.min(), vmax=spec.max())

    plt.ylim(top=t)
    plt.yticks(np.arange(min(frequencies), max(frequencies), 500))
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [с]')
    return mySpectrogram, frequencies