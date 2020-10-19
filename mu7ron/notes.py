class Note:
    div = ''
    def __init__(self, tick, res, tsig):
        self.tick = tick
        self.res  = res
        self.tsig = tsig
    def __repr__(self):
        return f'{self.__class__.__name__}(tick={self.tick}, res={self.res}, tsig={self.tsig})'


class WholeNote(Note):
    pass


class HalfNote(Note):
    pass


class QuarterNote(Note):
    pass


class EighthNote(Note):
    pass


class SixteenthNote(Note):
    pass


class ThirtySecondNote(Note):
    pass


class SixtyFourthSecondNote(Note):
    pass