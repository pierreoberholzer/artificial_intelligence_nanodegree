import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        
        best_BIC = float('Inf')
        best_model = None
          
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        
        for n in range(self.min_n_components,self.max_n_components+1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)
            
                #n_data_points = sum(self.lengths)
                n_data_points = model.n_features
                logN = np.log(len(self.X))
                n_free_params = n ** 2 + ( 2 * n * n_data_points - 1)
                BIC = (-2 *  logL) + (n_free_params * logN)
                
                if BIC < best_BIC:
                    best_model = model
                    best_BIC = BIC
            except:
                best_model = self.base_model(self.n_constant)
                
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def calc_DIC(self,n):
        other_words_logL = []
        model = self.base_model(n)
        for word,(X,lengths) in self.hwords.items():
            if word != self.this_word:
                logL = model.score(X, lengths)
                other_words_logL.append(logL)
        return model.score(self.X, self.lengths) - np.mean(other_words_logL)

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
               
        best_DIC = float('-Inf')
        best_model = None
          
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        try:
            for n in range(self.min_n_components,self.max_n_components+1):
                model = self.base_model(n)
                DIC = self.calc_DIC(n)             
                if DIC > best_DIC:
                    best_DIC = DIC  
                    best_model = model
        except:
            return self.base_model(self.n_constant)
    
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    
    def get_best_CV(self,n):
        
        split_method = KFold(n_splits = 2)
    
        best_logL_fold_ls = []
        
        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
    
            best_logL_fold = float('Inf')
    
            self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
            X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                  
            model = self.base_model(n)                   
            logL_fold = model.score(X_test, lengths_test)
            
            if logL_fold < best_logL_fold:
                best_logL_fold = logL_fold
            
            best_logL_fold_ls.append(best_logL_fold)
                 
        return np.mean(best_logL_fold_ls)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)      
        
        best_logL = float('Inf')
        best_model = None
        
        try:
            for n in range(self.min_n_components,self.max_n_components+1):
                logL = self.get_best_CV(n)
                model = self.base_model(n)
                if logL < best_logL :
                    best_model = model
                    best_logL = logL       
        
        except:
            return self.base_model(self.n_constant)
        
        return best_model
