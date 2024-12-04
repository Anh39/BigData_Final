from mrjob.job import MRJob

class MRWordFrequencyCount(MRJob):
    # DIRS = ["dependency"]

    def mapper(self, _, line):
        # from dependency import WOW, ndarray
        import numpy
        yield "chars", len(line)
        yield "words", len(line.split())
        yield "lines", 1

    def reducer(self, key, values):
        yield key, sum(values)


if __name__ == '__main__':
    MRWordFrequencyCount.run()