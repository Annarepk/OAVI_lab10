from voice import *

sounds = ['A', 'E', 'Meow']

for name in sounds:
    changeSampleRate(f"audio{name}.wav")
    sampleRate, samples = wavfile.read(f"audio/results/audio{name}.wav")
    spectogram, frequencies = spectrogramPlot(samples, sampleRate, 11000)
    spec = integralImg(spectogram)

    if name != 'Meow':
        formants = list(allFformants(frequencies, spec, 3))
        formants.sort()
        powr = power(frequencies, spec, 3, formants)

        with open(f"audio/results/result{name}.txt", 'w', encoding='utf-8') as file:
            print(f"Минимальная частота для звука {name}: {str(formants[0])}", file=file)
            print(f"Максимальная частота для звука {name}: {str(formants[-1])}", file=file)

            print(f"Тембрально окрашенный тон для звука {name}: {str(formants[0])}", file=file)

            # print(sorted(powr.items(), key=lambda item: item[1], reverse=True))
            print(f"Четыре самые сильные форманты: {str(sorted(powr, key=lambda i: powr[i])[-4:])}", file=file)

        plt.axhline(y=344, color='r', linestyle='-', lw=0.5, label="Форманты")
        plt.axhline(y=602, color='r', linestyle='-', lw=0.5)
        plt.axhline(y=861, color='r', linestyle='-', lw=0.5)
        plt.axhline(y=1119, color='r', linestyle='-', lw=0.5)
        plt.legend()

    else:
        formants = list(allFformants(frequencies, spec, 5))
        formants.sort()
        with open(f"audio/results/result{name}.txt", 'w', encoding='utf-8') as file:
            print(f"Минимальная частота для звука {name}: {str(formants[0])}", file=file)
            print(f"Максимальная частота для звука {name}: {str(formants[-2])}", file=file)

    dpi = 500
    plt.savefig(f"audio/results/spectrogram{name}.png", dpi=dpi)
    plt.clf()
