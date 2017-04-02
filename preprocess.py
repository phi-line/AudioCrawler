'''
this is a class containing various process filters to apply to numpy data
'''

class preProcess:
    @staticmethod
    def z_norm(s):
        '''
        takes in spectrogram data and returns z-normalized spectorgram data
        :return: preproc.scale(S)
        :			z-normalized ndarray
        '''
        import sklearn.preprocessing as preproc
        return preproc.scale(s)

    @staticmethod
    def zerone(s):
    	'''
    	normalize data from 0 to 1 feature_range
    	:return: scaler.fit_transform(s)
    	:			returns scaled data
    	'''
    	from sklearn.preprocessing import MinMaxScaler
    	scaler = MinMaxScaler(feature_range = (0, 1))
    	return scaler.fit_transform(s)