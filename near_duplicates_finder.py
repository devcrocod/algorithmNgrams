from simpleAPI2 import Text


class cl:
    def __init__(self, y, z):
        self.nGrams = y
        self.sents = z


def intersect(sent1, sent2):
    return [s for s in sent1 if s in sent2]


def compareSentWithClass(curClass, curSent):
    A = curSent.nGrams
    for sent in curClass.sents:
        A = intersect(A, sent.nGrams)
    avgOverlap = len(A) / len(curSent.nGrams)
    return avgOverlap


name = ""

text = Text("resources/" + name)
sents = text.sents
classes = []

for curSent in sents:
    if len(curSent.nGrams) == 0:
        continue
    bestOverlap = 0
    bestClass = 0
    for j in range(len(classes)):
        curClass = classes[j]
        curIntersect = intersect(curSent.nGrams, curClass.nGrams)
        curOverlap = compareSentWithClass(curClass, curSent)
        if curOverlap > bestOverlap:
            bestOverlap = curOverlap
            bestClass = j
    if bestOverlap < 0.5:
        classes.append(cl(curSent.nGrams, [curSent]))
    else:
        classes[bestClass].nGrams += curSent.nGrams
        classes[bestClass].sents.append(curSent)

cur = 0
with open(name + " result.txt", "w", encoding=text.encoding) as file:
    for curClass in classes:
        if len(curClass.sents) == 1:
            continue
        cur += 1
        file.write("========================= CLASS #%d =============================\n" % cur)
        file.write('\n'.join(
            ["(%d) {%d} [%d]: %s" % (sent.index, sent.start, sent.end, sent.sent) for sent in curClass.sents]))
        file.write("\n*****************************************************************\n")
