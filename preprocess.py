'''
this is a class containing various process filters to apply to numpy data
'''

class preProcess():
    @staticmethod
    def z_norm(s):
        '''
        takes in spectrogram data and returns z-normalized spectorgram data
        :return: preproc.scale(S)
        :			z-normalized ndarray
        '''
        import sklearn.preprocessing as preproc
        return preproc.scale(s)