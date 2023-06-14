
class MyObject:
    tracks = []
    def __init__(self, id, xi, yi, max_age):
        self.id = id
        self.x = xi
        self.y = yi
        self.tracks = []
        self.done = False
        self.age = 0
        self.max_age = max_age
        self.saved = False
    def getTracks(self):
        return self.tracks
    def getId(self):
        return self.id
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x,self.y])
        self.x = xn
        self.y = yn
    def setDone(self):
        self.done = True
    def timedOut(self):
        return self.done
    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True
    def setSaved(self):
        self.saved = True